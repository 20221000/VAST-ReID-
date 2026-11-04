#!/usr/bin/env python3
"""Night600(실제 야간) 정답률이 낮은 원인 분석.

    python tools/eval_night.py && python tools/night_analysis.py   # → results/night_analysis.csv

1) 이미지 통계: 밝기·검은 화소·번짐(과노출) 화소·고주파 노이즈·선명도·종횡비를 Market(합성 3조도)과 비교
2) 정답률 분해(CLIP 최종, CSLS): 쿼리 밝기 4분위별, 쿼리와 정답 갤러리의 밝기 차이별, 카메라별
3) hubness: 쿼리 top-10에 나온 서로 다른 갤러리 수, 상위 20장 점유율
4) 종횡비: 늘이기(기본) 대신 좌우 검은 여백으로 2:1을 맞춘 입력과 비교
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.data import list_night600, list_split  # noqa: E402
from vastreid.metrics import evaluate_fast  # noqa: E402
from vastreid.reid import MEAN, STD, load_model  # noqa: E402

DEV = "cuda"
PF = "clip_vit_b16-ftpair"
rows = []


def add(section, key, value, note=""):
    rows.append({"section": section, "key": key, "value": round(float(value), 4), "note": note})
    print(f"  [{section}] {key}: {value:.4f} {note}", flush=True)


def stats(paths):
    out = defaultdict(list)
    for p in paths:
        img = cv2.imread(str(p))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        out["luma"].append(g.mean())
        out["black"].append((g < 0.04).mean())
        out["glare"].append((g > 0.9).mean())
        out["noise"].append(np.abs(g - cv2.medianBlur((g * 255).astype(np.uint8), 3).astype(np.float32) / 255).mean()
                            / max(g.mean(), 1e-3))  # 밝기 대비 고주파 잔차 (입자 노이즈)
        out["sharp"].append(cv2.Laplacian(g / max(g.mean(), 1e-3), cv2.CV_32F).var())  # 밝기 정규화 선명도
        out["aspect"].append(img.shape[0] / img.shape[1])
    return {k: float(np.median(v)) for k, v in out.items()}


def feats(split):
    d = torch.load(ROOT / f"outputs/features/night__{PF}__{split}.pt", weights_only=True)
    return torch.nn.functional.normalize(d["feats"].float(), dim=1).to(DEV)


def ranks_of(s, qi, gi):
    gp = torch.tensor([x.pid for x in gi], device=DEV)
    gc = torch.tensor([x.camid for x in gi], device=DEV)
    out = []
    for i, q in enumerate(qi):
        v = ~((gp == q.pid) & (gc == q.camid))
        sc = s[i].masked_fill(~v, -1e9)
        pos = sc[(gp == q.pid) & v].max()
        out.append(int((sc > pos).sum()) + 1)
    return np.array(out)


@torch.no_grad()
def extract_letterbox(model, paths, batch=256):
    """세로:가로 = 2:1이 되도록 좌우(또는 위아래)에 검은 여백을 붙인 뒤 256×128로 축소."""
    mean = torch.tensor(MEAN, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=DEV).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(paths), batch):
        xs = []
        for p in paths[s:s + batch]:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            if h > 2 * w:
                pad = (h // 2 - w) // 2
                img = cv2.copyMakeBorder(img, 0, 0, pad, h // 2 - w - pad, cv2.BORDER_CONSTANT)
            else:
                pad = (2 * w - h) // 2
                img = cv2.copyMakeBorder(img, pad, 2 * w - h - pad, 0, 0, cv2.BORDER_CONSTANT)
            img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
            xs.append(torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()))
        x = torch.stack(xs).to(DEV).float() / 255
        out.append(model((x - mean) / std).float())
    return torch.nn.functional.normalize(torch.cat(out), dim=1)


def main():
    qi, gi = list_night600("query"), list_night600("gallery")
    # 1) 이미지 통계
    sets = {"Night600 query": [x.path for x in qi]}
    for lv in ["normal", "low", "extreme"]:
        its = list_split(ROOT / "data/vast", lv, "query")
        sets[f"Market {lv} query"] = [x.path for x in its[::3]]
    for name, ps in sets.items():
        for k, v in stats(ps).items():
            add("image_stats", f"{name} | {k}", v, "중앙값")

    # 2) 정답률 분해 (CLIP 최종 + CSLS, tools/eval_night.py와 같은 설정)
    qf, gf, rf = feats("query"), feats("gallery"), feats("ref")
    hub = (gf @ rf.T).topk(50, dim=1).values.mean(1)
    s = qf @ gf.T - hub[None, :]
    rk = ranks_of(s, qi, gi)
    add("overall", "Rank-1", (rk == 1).mean())
    add("overall", "Rank-10", (rk <= 10).mean())
    add("overall", "median rank of first correct", np.median(rk))
    lum_q = np.array([cv2.imread(str(x.path)).mean() / 255 for x in qi])
    qs = np.quantile(lum_q, [0.25, 0.5, 0.75])
    for b, (lo, hi) in enumerate(zip([0, *qs], [*qs, 1.01])):
        m = (lum_q >= lo) & (lum_q < hi)
        add("by_query_luma", f"Q{b + 1} ({lo:.3f}~{min(hi, 1):.3f})", (rk[m] == 1).mean(), f"n={m.sum()}")
    # 쿼리와 (다른 카메라) 정답 갤러리 중 가장 비슷한 밝기 차이
    lum_g = np.array([cv2.imread(str(x.path)).mean() / 255 for x in gi])
    gp, gc = np.array([x.pid for x in gi]), np.array([x.camid for x in gi])
    diff = np.array([np.min(np.abs(lum_g[(gp == q.pid) & (gc != q.camid)] - lum_q[i])) for i, q in enumerate(qi)])
    for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 1)]:
        m = (diff >= lo) & (diff < hi)
        add("by_luma_gap", f"쿼리-정답 밝기 차 {lo}~{hi}", (rk[m] == 1).mean(), f"n={m.sum()}")
    for c in sorted({x.camid for x in qi}):
        m = np.array([x.camid == c for x in qi])
        add("by_camera", f"cam{c + 1}", (rk[m] == 1).mean(), f"n={m.sum()}, 평균 밝기 {lum_q[m].mean():.3f}")

    # 3) hubness
    top = s.topk(10, dim=1).indices.flatten().cpu().numpy()
    cnt = Counter(top)
    add("hubness", "top-10 서로 다른 갤러리 수", len(cnt), f"gallery {len(gi)}")
    add("hubness", "상위 20장 점유율", sum(v for _, v in cnt.most_common(20)) / len(top))
    # 같은 사람끼리 vs 다른 사람끼리 유사도 분포가 얼마나 겹치는지
    cos = (qf @ gf.T).cpu().numpy()
    qp = np.array([x.pid for x in qi])
    same = cos[(qp[:, None] == gp[None, :]) & (np.array([x.camid for x in qi])[:, None] != gc[None, :])]
    diffp = cos[qp[:, None] != gp[None, :]]
    add("similarity", "같은 사람(다른 카메라) 코사인 평균", same.mean())
    add("similarity", "다른 사람 코사인 평균", diffp.mean())
    add("similarity", "다른 사람 코사인 상위 1%", np.quantile(diffp, 0.99))

    # 4) 종횡비: 늘이기 vs 여백
    m = load_model(None, ROOT / "weights/clip_vit_b16_vast_lowlight_ft_pair.pth", DEV)
    lq, lg = extract_letterbox(m, [x.path for x in qi]), extract_letterbox(m, [x.path for x in gi])
    r = evaluate_fast(-(lq @ lg.T), [x.pid for x in qi], [x.pid for x in gi], [x.camid for x in qi], [x.camid for x in gi])
    add("aspect", "여백으로 2:1 맞춤 (코사인) Rank-1", r["rank1"])
    r0 = evaluate_fast(-(qf @ gf.T), [x.pid for x in qi], [x.pid for x in gi], [x.camid for x in qi], [x.camid for x in gi])
    add("aspect", "늘이기 (기본, 코사인) Rank-1", r0["rank1"])

    with open(ROOT / "results/night_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[saved] results/night_analysis.csv")


if __name__ == "__main__":
    main()
