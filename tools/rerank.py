#!/usr/bin/env python3
"""속성 재순위화: score = cos(f_q, f_g) + λ · attr_agreement(q, g)

    python tools/rerank.py --attr-source oracle --out results/rerank_oracle.csv

- attr_agreement: 6개 속성 중 둘 다 알려진(unknown 아님) 속성의 일치 비율. 속성이 없는 사람(distractor)은 0.5(중립).
- λ는 **train 이미지로 만든 검증 분할**에서 mAP 최대가 되도록 고름 (테스트셋 미사용).
  검증 분할: train ID마다 카메라별 첫 이미지를 query, 나머지를 gallery.
- attr-source
    oracle : query·gallery 모두 정답 속성 (속성이 완벽히 주어졌을 때의 상한선, 실제 시스템 아님)
- 특징은 outputs/features/ 에 캐시 (L2 정규화 전 원본).
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
from vastreid.data import Item, list_split  # noqa: E402
from vastreid.metrics import evaluate  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

ATTRS = ["upper_color", "lower_color", "sleeve", "lower_type", "bag", "hat"]


def get_feats(model, data: Path, level: str, split: str, device: str, cache_dir: Path) -> tuple[list[Item], torch.Tensor]:
    items = list_split(data, level, split)
    cp = cache_dir / f"osnet_x1_0__none__{level}__{split}.pt"
    if cp.exists():
        d = torch.load(cp, weights_only=True)
        if d["n"] == len(items):
            return items, d["feats"]
    f, _ = extract(model, [x.path for x in items], device)
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n": len(items), "feats": f.cpu()}, cp)
    return items, f.cpu()


def attr_matrix(pids: list[int], table: dict[int, dict]) -> np.ndarray:
    """(N, 6) 정수 코드. 속성 없음/unknown = -1."""
    vocab: dict[tuple[str, str], int] = {}
    out = np.full((len(pids), len(ATTRS)), -1, np.int32)
    for i, pid in enumerate(pids):
        a = table.get(pid)
        if a is None:
            continue
        for j, k in enumerate(ATTRS):
            v = str(a[k])
            if v != "unknown":
                out[i, j] = vocab.setdefault((k, v), len(vocab))
    return out


def agreement(qa: np.ndarray, ga: np.ndarray, device: str) -> torch.Tensor:
    q = torch.from_numpy(qa).to(device)[:, None, :]
    g = torch.from_numpy(ga).to(device)[None, :, :]
    known = (q >= 0) & (g >= 0)
    match = ((q == g) & known).sum(-1).float()
    n = known.sum(-1).float()
    return torch.where(n > 0, match / n.clamp(min=1), torch.full_like(n, 0.5))


def run(qf, gf, qi, gi, qa, ga, lam, device) -> dict:
    qf = torch.nn.functional.normalize(qf.to(device), dim=1)
    gf = torch.nn.functional.normalize(gf.to(device), dim=1)
    score = qf @ gf.T
    if lam:
        score = score + lam * agreement(qa, ga, device)
    return evaluate((-score).cpu().numpy(), [x.pid for x in qi], [x.pid for x in gi],
                    [x.camid for x in qi], [x.camid for x in gi])


def val_split(items: list[Item]) -> tuple[list[int], list[int]]:
    seen, q, g = set(), [], []
    for i, it in enumerate(items):
        key = (it.pid, it.camid)
        (g if key in seen else q).append(i)
        seen.add(key)
    return q, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/vast")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/osnet_x1_0_market1501.pth")
    ap.add_argument("--attr-source", default="oracle", choices=["oracle"])
    ap.add_argument("--levels", nargs="+", default=["normal", "low", "extreme"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--out", type=Path, default=ROOT / "results/rerank_oracle.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    persons = json.loads((ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"]
    table = {int(k): v for k, v in persons.items()}
    cache = ROOT / "outputs/features"
    model = load_model("osnet_x1_0", args.weights, device)
    t0 = time.time()

    rows = []
    for lv in args.levels:
        pairs = [("same", lv, lv)] + ([("cross", lv, "normal")] if lv != "normal" else [])
        for proto, ql, gl in pairs:
            # ---- 검증 분할(train)에서 λ 선택
            tq_items, tq_f = get_feats(model, args.data, ql, "train", device, cache)
            tg_items, tg_f = get_feats(model, args.data, gl, "train", device, cache)
            qi_idx, gi_idx = val_split(tq_items)
            vq = [tq_items[i] for i in qi_idx]
            vg = [tg_items[i] for i in gi_idx]
            vqa = attr_matrix([x.pid for x in vq], table)
            vga = attr_matrix([x.pid for x in vg], table)
            val = {lam: run(tq_f[qi_idx], tg_f[gi_idx], vq, vg, vqa, vga, lam, device)["mAP"] for lam in args.lambdas}
            best = max(val, key=val.get)

            # ---- 테스트셋 평가 (λ=0, λ*)
            qi, qf = get_feats(model, args.data, ql, "query", device, cache)
            gi, gf = get_feats(model, args.data, gl, "gallery", device, cache)
            qa = attr_matrix([x.pid for x in qi], table)
            ga = attr_matrix([x.pid for x in gi], table)
            for tag, lam in [("cosine", 0.0), ("rerank", best)]:
                r = run(qf, gf, qi, gi, qa, ga, lam, device)
                rows.append({"attr_source": args.attr_source, "method": tag, "lambda": lam, "protocol": proto,
                             "query_level": ql, "gallery_level": gl,
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                             "val_mAP_at_lambda": round(val[lam], 4)})
                print(f"  {proto:5s} q={ql:7s} g={gl:7s} {tag:6s} λ={lam:<4} R1 {r['rank1']*100:5.1f}  "
                      f"mAP {r['mAP']*100:5.1f}  mINP {r['mINP']*100:5.1f}", flush=True)
            print("      val mAP by λ: " + ", ".join(f"{k}:{v*100:.1f}" for k, v in val.items()))
    print(f"[done] {time.time() - t0:.1f}s  GPU {gpu_temp()}°C")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
