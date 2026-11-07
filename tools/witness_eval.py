#!/usr/bin/env python3
"""목격자 진술(속성 재순위화)이 최종 모델에서도 의미가 있는지 평가.

    python tools/witness_eval.py   # → results/witness_eval.csv

데모와 같은 점수식(vastreid/demo_core.search)을 Market query 전체(3,368)에 적용:
  점수 = -(k-reciprocal 거리 + β·hub) + λ·(속성 일치도 × n/6)      (재순위화 끔: 코사인 − β·hub + λ·…)
진술 종류: 없음 / 맞는 6개 / 맞는 2개(상의 색·가방) / 틀린 상의 색 1개 / 맞는 1개 + 틀린 1개.
λ: 자동(train 검증 분할에서 고른 값) / 강하게 0.3 / 매우 강하게 0.9 (데모 "진술 반영 강도"와 같음).
query 쪽 진술은 사람이 본 정답 속성(annotations/person_attributes.json)으로 대신하고, gallery 쪽은 분류기 예측.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid import demo_core as core  # noqa: E402

PF = core.BACKBONES[core.DEFAULT_BACKBONE]
KEYS = list(core.ATTR_VALUES)


def testimony(pid: int, kind: str) -> dict:
    a = core.PERSONS.get(pid, {})
    ok = {k: str(a[k]) for k in KEYS if k in a and str(a[k]) in core.ATTR_VALUES[k] and str(a[k]) != "unknown"}
    wrong_upper = None
    if "upper_color" in ok:
        vals = [v for v in core.ATTR_VALUES["upper_color"] if v not in (ok["upper_color"], "unknown")]
        wrong_upper = vals[pid % len(vals)]
    if kind == "none":
        return {}
    if kind == "right6":
        return ok
    if kind == "right2":
        return {k: ok[k] for k in ("upper_color", "bag") if k in ok}
    if kind == "wrong1":
        return {"upper_color": wrong_upper} if wrong_upper else {}
    if kind == "right1_wrong1":
        d = {"upper_color": wrong_upper} if wrong_upper else {}
        if "bag" in ok:
            d["bag"] = ok["bag"]
        return d
    raise ValueError(kind)


def main():
    gi = core.items("normal", "gallery")
    gp = np.array([g.pid for g in gi])
    gc = np.array([g.camid for g in gi])
    gprob = core.probs("normal", "gallery", core.ATTR_PROBE[PF])
    rows = []
    for ql in ["low", "extreme"]:
        qi = core.items(ql, "query")
        qf, gf = core.feats(ql, "query", "none", PF), core.feats("normal", "gallery", "none", PF)
        kp, cp = core.kr_params(ql, "none", PF), core.csls_params(ql, "normal", "none", PF)
        hub_kr = core.hub_score(ql, "normal", "none", kp["k"], PF)
        hub_c = core.hub_score(ql, "normal", "none", cp["k"], PF)
        for path in ["kr+csls", "csls"]:
            auto = kp["lam"] if path == "kr+csls" else cp["lam"]
            for kind in ["none", "right6", "right2", "wrong1", "right1_wrong1"]:
                for strength, lam in [("auto", auto), ("strong", 0.3), ("max", 0.9)]:
                    if kind == "none" and strength != "auto":
                        continue
                    r1, aps = [], []
                    for i, q in enumerate(qi):
                        if path == "kr+csls":
                            s = -(core.kr_row(ql, "none", PF, i) + kp["beta"] * hub_kr)
                        else:
                            s = gf @ qf[i] - cp["beta"] * hub_c
                        agr = core._agreement(testimony(q.pid, kind), gprob)
                        if agr is not None:
                            s = s + lam * agr
                        valid = ~((gp == q.pid) & (gc == q.camid))
                        s = np.where(valid, s, -np.inf)
                        order = np.argsort(-s)
                        r1.append(gp[order[0]] == q.pid)
                        ap = core._ap(order, gp, valid, q.pid)
                        if ap is not None:
                            aps.append(ap)
                    row = {"backbone": PF, "query_level": ql, "path": path, "testimony": kind, "strength": strength,
                           "lambda": round(lam, 3), "rank1": round(float(np.mean(r1)), 4), "mAP": round(float(np.mean(aps)), 4)}
                    rows.append(row)
                    print(f"  {ql:7s} {path:7s} {kind:14s} {strength:6s} λ={lam:<5} R1 {row['rank1'] * 100:5.1f}  mAP {row['mAP'] * 100:5.1f}", flush=True)
    out = ROOT / "results/witness_eval.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
