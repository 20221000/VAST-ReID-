#!/usr/bin/env python3
"""k-reciprocal 재순위화(Zhong et al., CVPR 2017) 조합 평가. 학습 없음.

    python tools/kreciprocal.py      # → results/kreciprocal.csv

- k-reciprocal 파라미터는 논문 표준값 k1=20, k2=6, λ=0.3으로 고정 (테스트셋으로 조정하지 않음). 구현은 torchreid.
- 순서: **k-reciprocal을 먼저 원래 코사인 거리(q-g, q-q, g-g)에 적용**한 뒤, 그 거리에
    + β_kr · r_g            (CSLS hub 점수, 클수록 hub → 거리 증가)
    − λ_kr · 속성 일치도      (목격자 진술)
  을 더함. CSLS를 먼저 q-g 거리에만 적용하면 q-q·g-g 거리와 척도가 어긋나 k-reciprocal 이웃 계산이 깨짐
  (실제로 CSLS 단독과 결과가 같아짐) → 순서를 바꿈.
- r_g의 k는 tools/hubness.py가 고른 값을 쓰고, β_kr·λ_kr은 k-reciprocal 거리 척도에 맞춰 train 검증 분할에서 다시 선택.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch
from torchreid.utils.rerank import re_ranking

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from rerank import agreement, clip_rep, get_feats, gt_rep, val_split  # noqa: E402
from vastreid.metrics import evaluate_fast  # noqa: E402
from vastreid.reid import gpu_temp, load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K1, K2, LAM = 20, 6, 0.3
BETAS = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
LAMBDAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
norm = lambda x: torch.nn.functional.normalize(x.float().to(DEV), dim=1)


def r_gallery(gf, ref, k, g_pids=None, r_pids=None):
    out = []
    for s in range(0, len(gf), 4096):
        sim = gf[s:s + 4096] @ ref.T
        if g_pids is not None:
            sim = sim.masked_fill(g_pids[s:s + 4096, None] == r_pids[None, :], -2.0)
        out.append(sim.topk(k, dim=1).values.mean(1))
    return torch.cat(out).cpu().numpy()


def kr(qf, gf):
    return re_ranking((1 - qf @ gf.T).cpu().numpy(), (1 - qf @ qf.T).cpu().numpy(), (1 - gf @ gf.T).cpu().numpy(),
                      k1=K1, k2=K2, lambda_value=LAM)


def ev(dist, qi, gi):
    return evaluate_fast(dist, [x.pid for x in qi], [x.pid for x in gi], [x.camid for x in qi], [x.camid for x in gi])


def hub_k(feat, proto, ql):
    name = "hubness_adabn.csv" if feat == "adabn" else "hubness_wide.csv"
    with open(ROOT / "results" / name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] == "csls" and r["protocol"] == proto and r["query_level"] == ql:
                return int(r["k"])
    raise KeyError((feat, proto, ql))


def main():
    t0 = time.time()
    table = {int(k): v for k, v in json.loads(
        (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}
    data, cache = ROOT / "data/vast", ROOT / "outputs/features"
    model = load_model("osnet_x1_0", ROOT / "weights/osnet_x1_0_market1501.pth", DEV)
    conds = [("same", "normal", "normal"), ("same", "low", "low"), ("cross", "low", "normal"),
             ("same", "extreme", "extreme"), ("cross", "extreme", "normal")]
    rows = []
    for feat in ["none", "adabn"]:
        for proto, ql, gl in conds:
            k = hub_k(feat, proto, ql)
            # ---- 검증 분할: k-reciprocal 거리 위에서 β_kr, λ_kr 선택
            tq, tqf = get_feats(model, data, ql, "train", DEV, cache, feat)
            tg, tgf = get_feats(model, data, gl, "train", DEV, cache, feat)
            qi_idx, gi_idx = val_split(tq)
            vq, vg = [tq[i] for i in qi_idx], [tg[i] for i in gi_idx]
            vqf, vgf, ref = norm(tqf[qi_idx]), norm(tgf[gi_idx]), norm(tqf)
            r_p = torch.tensor([x.pid for x in tq], device=DEV)
            v_rg = r_gallery(vgf, ref, k, torch.tensor([x.pid for x in vg], device=DEV), r_p)
            v_d = kr(vqf, vgf)
            v_agr = agreement(gt_rep([x.pid for x in vq], table),
                              clip_rep(gl, "train", tg, gi_idx, "probe"), DEV).cpu().numpy()
            beta = max(BETAS, key=lambda b: ev(v_d + b * v_rg[None, :], vq, vg)["mAP"])
            lam = max(LAMBDAS, key=lambda l: ev(v_d + beta * v_rg[None, :] - l * v_agr, vq, vg)["mAP"])
            # ---- 테스트
            qi, qf = get_feats(model, data, ql, "query", DEV, cache, feat)
            gi, gf = get_feats(model, data, gl, "gallery", DEV, cache, feat)
            qf, gf = norm(qf), norm(gf)
            rg = r_gallery(gf, ref, k)[None, :]
            agr = agreement(gt_rep([x.pid for x in qi], table),
                            clip_rep(gl, "gallery", gi, pred="probe"), DEV).cpu().numpy()
            d = kr(qf, gf)
            for name, dist, b, l in [("kr", d, 0.0, 0.0), ("kr+csls", d + beta * rg, beta, 0.0),
                                     ("kr+csls+witness-probe", d + beta * rg - lam * agr, beta, lam)]:
                r = ev(dist, qi, gi)
                rows.append({"feat": feat, "method": name, "protocol": proto, "query_level": ql, "gallery_level": gl,
                             "k": k, "beta": b, "lambda": l, "k1": K1, "k2": K2, "kr_lambda": LAM,
                             **{m: round(v, 6) if isinstance(v, float) else v for m, v in r.items()}})
                print(f"  {feat:5s} {proto:5s} {ql:7s}→{gl:7s} {name:22s} β={b:<4} λ={l:<4} "
                      f"R1 {r['rank1']*100:5.1f}  mAP {r['mAP']*100:5.1f}  mINP {r['mINP']*100:5.1f}", flush=True)
    print(f"[done] {time.time() - t0:.0f}s  GPU {gpu_temp()}°C")
    out = ROOT / "results/kreciprocal.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
