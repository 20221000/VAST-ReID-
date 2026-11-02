#!/usr/bin/env python3
"""CLIP zero-shot으로 모든 이미지의 속성 확률을 예측하고, 조도별 속성 정확도를 측정.

    python tools/clip_attributes.py --limit 200          # 샘플로 시간 측정
    python tools/clip_attributes.py                      # 전체 → outputs/attr_probs/, results/attr_acc_clip.csv

정확도: 정답이 unknown이 아닌 경우만, 테스트 ID 이미지(query+gallery) 기준.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.clip_attr import ATTR_VALUES, ClipAttr  # noqa: E402
from vastreid.data import list_split  # noqa: E402
from vastreid.reid import gpu_temp  # noqa: E402


def probs_path(arch: str, level: str, split: str) -> Path:
    return ROOT / f"outputs/attr_probs/clip_{arch}__{level}__{split}.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/vast")
    ap.add_argument("--arch", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--levels", nargs="+", default=["normal", "low", "extreme"])
    ap.add_argument("--splits", nargs="+", default=["train", "query", "gallery"])
    ap.add_argument("--limit", type=int, default=None, help="split당 이미지 수 제한 (샘플 테스트, 저장 안 함)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/attr_acc_clip.csv")
    args = ap.parse_args()

    table = {int(k): v for k, v in json.loads(
        (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}
    clip = ClipAttr(args.arch, args.pretrained)
    print(f"[info] CLIP {args.arch}/{args.pretrained} GPU {gpu_temp()}°C")

    rows, t0 = [], time.time()
    for lv in args.levels:
        correct = {a: [0, 0] for a in ATTR_VALUES}
        for sp in args.splits:
            items = list_split(args.data, lv, sp)[: args.limit]
            t = time.time()
            probs = clip.predict([x.path for x in items])
            print(f"  {lv}/{sp}: {len(items)} imgs {time.time() - t:.1f}s  GPU {gpu_temp()}°C", flush=True)
            if not args.limit:
                pp = probs_path(args.arch, lv, sp)
                pp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pp, pids=np.array([x.pid for x in items]), **probs)
            if sp == "train":
                continue
            for a, vals in ATTR_VALUES.items():
                pred = probs[a].argmax(1)
                for i, it in enumerate(items):
                    p = table.get(it.pid)
                    if p is None or str(p[a]) == "unknown":
                        continue
                    correct[a][0] += int(vals[pred[i]] == str(p[a]))
                    correct[a][1] += 1
        accs = {a: c / max(n, 1) for a, (c, n) in correct.items()}
        rows.append({"model": f"clip_{args.arch}", "level": lv, **{a: round(v, 4) for a, v in accs.items()},
                     "mean_acc": round(float(np.mean(list(accs.values()))), 4)})
        print(f"  [{lv}] " + "  ".join(f"{a} {v*100:.1f}" for a, v in accs.items())
              + f"  | mean {np.mean(list(accs.values()))*100:.1f}")
    print(f"[done] {time.time() - t0:.1f}s  GPU {gpu_temp()}°C")

    # 참고: 다수 클래스(항상 가장 흔한 값)로 찍었을 때 정확도
    maj = {}
    for a, vals in ATTR_VALUES.items():
        labels = [str(p[a]) for p in table.values() if p["split"] == "test" and str(p[a]) != "unknown"]
        maj[a] = max(labels.count(v) for v in set(labels)) / len(labels)
    print("  [majority baseline, ID 기준] " + "  ".join(f"{a} {v*100:.1f}" for a, v in maj.items()))

    out = args.out if not args.limit else args.out.with_name(args.out.stem + "_sample.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.append({"model": "majority(ID)", "level": "-", **{a: round(v, 4) for a, v in maj.items()},
                 "mean_acc": round(float(np.mean(list(maj.values()))), 4)})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
