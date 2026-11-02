#!/usr/bin/env python3
"""Hubness 완화(CSLS 방식) + 속성 재순위화 조합 평가. 학습 없음.

    python tools/hubness.py      # → results/hubness.csv

score(q, g) = cos(q, g) − β · r_g + λ · attr_agreement(q, g)
  r_g = gallery 이미지 g와 참조 집합 R의 코사인 유사도 상위 k개 평균 (g가 'query 쪽 이미지들 전반'에 얼마나 가까운가)
  R   = query와 같은 조도의 **train 이미지** (라벨 사용 안 함). 테스트 query는 사용하지 않음.
        검증 시에는 g와 같은 사람의 이미지를 R에서 제외 (테스트에서는 train과 test ID가 겹치지 않으므로 동일 조건).
  CSLS(Conneau et al., 2018)의 query 쪽 항 r_q는 query마다 상수라 순위에 영향 없으므로 생략.
- k, β, λ는 train 검증 분할(카메라별 첫 이미지 = query)에서 mAP 최대가 되도록 선택.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from rerank import agreement, clip_rep, get_feats, gt_rep, val_split  # noqa: E402
from vastreid.metrics import evaluate  # noqa: E402
from vastreid.reid import gpu_temp, load_model  # noqa: E402

KS = [10, 50]
BETAS = [0, 0.25, 0.5, 0.75, 1.0]
LAMBDAS = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
norm = lambda x: torch.nn.functional.normalize(x.float().to(DEV), dim=1)


def r_gallery(gf, ref, k, g_pids=None, r_pids=None):
    """gallery마다 참조 집합과의 상위 k 유사도 평균. pid가 주어지면 같은 사람은 제외."""
    out = []
    for s in range(0, len(gf), 4096):
        sim = gf[s:s + 4096] @ ref.T
        if g_pids is not None:
            sim = sim.masked_fill(g_pids[s:s + 4096, None] == r_pids[None, :], -2.0)
        out.append(sim.topk(k, dim=1).values.mean(1))
    return torch.cat(out)


def score_eval(qf, gf, rg, beta, agr, lam, qi, gi):
    s = qf @ gf.T - beta * rg[None, :]
    if lam:
        s = s + lam * agr
    r = evaluate((-s).cpu().numpy(), [x.pid for x in qi], [x.pid for x in gi],
                 [x.camid for x in qi], [x.camid for x in gi])
    return r, s


def hub_stats(s):
    top = s.topk(10, dim=1).indices.cpu().numpy().ravel()
    c = Counter(top.tolist())
    return len(c), sum(v for _, v in c.most_common(20)) / len(top)


def main():
    t0 = time.time()
    table = {int(k): v for k, v in json.loads(
        (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}
    data, cache = ROOT / "data/vast", ROOT / "outputs/features"
    model = load_model("osnet_x1_0", ROOT / "weights/osnet_x1_0_market1501.pth", DEV)
    rows = []
    conds = [("same", "normal", "normal"), ("same", "low", "low"), ("cross", "low", "normal"),
             ("same", "extreme", "extreme"), ("cross", "extreme", "normal")]
    for proto, ql, gl in conds:
        # ---------- 검증 분할
        tq, tqf = get_feats(model, data, ql, "train", DEV, cache)
        tg, tgf = get_feats(model, data, gl, "train", DEV, cache)
        qi_idx, gi_idx = val_split(tq)
        vq, vg = [tq[i] for i in qi_idx], [tg[i] for i in gi_idx]
        vqf, vgf, ref = norm(tqf[qi_idx]), norm(tgf[gi_idx]), norm(tqf)
        r_pids = torch.tensor([x.pid for x in tq], device=DEV)
        vg_pids = torch.tensor([x.pid for x in vg], device=DEV)
        v_rg = {k: r_gallery(vgf, ref, k, vg_pids, r_pids) for k in KS}
        best = max(((k, b) for k in KS for b in BETAS),
                   key=lambda kb: score_eval(vqf, vgf, v_rg[kb[0]], kb[1], None, 0, vq, vg)[0]["mAP"])
        k_star, b_star = best
        v_agr = {
            "probe": agreement(clip_rep(ql, "train", tq, qi_idx, "probe"), clip_rep(gl, "train", tg, gi_idx, "probe"), DEV),
            "witness-probe": agreement(gt_rep([x.pid for x in vq], table), clip_rep(gl, "train", tg, gi_idx, "probe"), DEV),
        }
        lam_star = {}
        for src, agr in v_agr.items():
            for beta in (0.0, b_star):
                lam_star[(src, beta)] = max(LAMBDAS, key=lambda lam: score_eval(
                    vqf, vgf, v_rg[k_star], beta, agr, lam, vq, vg)[0]["mAP"])

        # ---------- 테스트
        qi, qf = get_feats(model, data, ql, "query", DEV, cache)
        gi, gf = get_feats(model, data, gl, "gallery", DEV, cache)
        qf, gf = norm(qf), norm(gf)
        rg = r_gallery(gf, ref, k_star)  # 참조 = train (테스트와 ID 겹치지 않음)
        agr = {
            "probe": agreement(clip_rep(ql, "query", qi, pred="probe"), clip_rep(gl, "gallery", gi, pred="probe"), DEV),
            "witness-probe": agreement(gt_rep([x.pid for x in qi], table), clip_rep(gl, "gallery", gi, pred="probe"), DEV),
        }
        methods = [("cosine", 0.0, None, 0.0), ("csls", b_star, None, 0.0)]
        for src in ["probe", "witness-probe"]:
            methods.append((src, 0.0, src, lam_star[(src, 0.0)]))
            methods.append((f"csls+{src}", b_star, src, lam_star[(src, b_star)]))
        for name, beta, src, lam in methods:
            r, s = score_eval(qf, gf, rg, beta, agr[src] if src else None, lam, qi, gi)
            n_uniq, hub20 = hub_stats(s)
            rows.append({"method": name, "protocol": proto, "query_level": ql, "gallery_level": gl,
                         "k": k_star, "beta": beta, "lambda": lam,
                         **{m: round(v, 4) if isinstance(v, float) else v for m, v in r.items()},
                         "top10_unique_gallery": n_uniq, "top20_hub_share": round(hub20, 4)})
            print(f"  {proto:5s} {ql:7s}→{gl:7s} {name:20s} k={k_star} β={beta:<4} λ={lam:<4} "
                  f"R1 {r['rank1']*100:5.1f} mAP {r['mAP']*100:5.1f}  uniq {n_uniq:5d} hub20 {hub20*100:4.1f}%",
                  flush=True)
    print(f"[done] {time.time() - t0:.1f}s  GPU {gpu_temp()}°C")
    out = ROOT / "results/hubness.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
