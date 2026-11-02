#!/usr/bin/env python3
"""실패 원인 분석: Rank-1을 놓친 query에서 1위로 잘못 나온 gallery 이미지가 왜 나왔는지 분류.

    python tools/failure_analysis.py     # → results/failure_causes.csv, outputs/demo/hub_freq__*.npy

분류 (위에서부터 먼저 해당되는 것 하나):
  distractor : 사람이 아닌/부분 이미지 (Market-1501 pid 0)
  hub        : 어두운 query 1% 이상의 top-10에 반복 등장하는 gallery 이미지 (hubness)
  similar    : 정답 속성 6개 중 양쪽 모두 알려진 속성이 전부 일치 (비슷한 옷)
  other      : 외형 특징 손실 (위에 해당하지 않음)
설정: 외형만(OSNet, 코사인) / AdaBN + CSLS(k=50, β=1.0) / 미세조정 OSNet + CSLS(k=50, β=1.0, AdaBN 없음).
gallery는 normal(주간 기록).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.data import list_split  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = ROOT / "data/vast"
ATTRS = ["upper_color", "lower_color", "sleeve", "lower_type", "bag", "hat"]
CAUSES = ["distractor", "hub", "similar", "other"]


BASE, FT = "osnet_x1_0", "osnet_x1_0-ft"


def feats(level, split, feat, prefix=BASE):
    tag = f"adabn-{level}" if feat == "adabn" and level != "normal" else "none"
    d = torch.load(ROOT / f"outputs/features/{prefix}__{tag}__{level}__{split}.pt", weights_only=True)
    return torch.nn.functional.normalize(d["feats"].float(), dim=1).to(DEV)


def main():
    persons = {int(k): v for k, v in json.loads(
        (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}
    gi = list_split(DATA, "normal", "gallery")
    gp = torch.tensor([g.pid for g in gi], device=DEV)
    gc = torch.tensor([g.camid for g in gi], device=DEV)
    rows = []
    for setting, feat, beta, pf in [("OSNet", "none", 0.0, BASE), ("AdaBN+CSLS", "adabn", 1.0, BASE),
                                    ("FT+CSLS", "none", 1.0, FT)]:
        for ql in ["normal", "low", "extreme"]:
            qi = list_split(DATA, ql, "query")
            qp = torch.tensor([q.pid for q in qi], device=DEV)
            qc = torch.tensor([q.camid for q in qi], device=DEV)
            qf, gf = feats(ql, "query", feat, pf), feats("normal", "gallery", feat, pf)
            s = qf @ gf.T
            if beta:
                ref = feats(ql, "train", feat, pf)
                rg = torch.cat([(gf[i:i + 4096] @ ref.T).topk(50, dim=1).values.mean(1)
                                for i in range(0, len(gf), 4096)])
                s = s - beta * rg[None, :]
            s = s.masked_fill((gp[None, :] == qp[:, None]) & (gc[None, :] == qc[:, None]), -1e9)
            top = s.topk(10, dim=1).indices
            freq = np.bincount(top.flatten().cpu().numpy(), minlength=len(gi))
            pre = "" if pf == BASE else f"{pf}__"  # vastreid/demo_core.hub_freq와 같은 이름
            out = ROOT / f"outputs/demo/hub_freq__{pre}{feat}__{ql}__normal.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, freq)
            hub_thr = max(1, int(0.01 * len(qi)))
            top1 = top[:, 0].cpu().numpy()
            wrong = [(i, j) for i, j in enumerate(top1) if gi[j].pid != qi[i].pid]
            cnt = Counter()
            for i, j in wrong:
                gpid = gi[j].pid
                if gpid <= 0:
                    cnt["distractor"] += 1
                elif freq[j] >= hub_thr:
                    cnt["hub"] += 1
                else:
                    a, b = persons.get(qi[i].pid), persons.get(gpid)
                    known = [k for k in ATTRS if a and b and str(a[k]) != "unknown" and str(b[k]) != "unknown"]
                    if known and all(str(a[k]) == str(b[k]) for k in known):
                        cnt["similar"] += 1
                    else:
                        cnt["other"] += 1
            for c in CAUSES:
                rows.append({"setting": setting, "query_level": ql, "gallery_level": "normal", "cause": c,
                             "count": cnt[c], "ratio": round(cnt[c] / max(len(wrong), 1), 4),
                             "n_wrong_rank1": len(wrong), "n_query": len(qi), "hub_threshold": hub_thr})
            print(f"  {setting:11s} {ql:7s}→normal  Rank-1 오답 {len(wrong):4d}/{len(qi)}  "
                  + "  ".join(f"{c} {cnt[c] / max(len(wrong), 1) * 100:4.1f}%" for c in CAUSES))
    out = ROOT / "results/failure_causes.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
