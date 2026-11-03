#!/usr/bin/env python3
"""다른 데이터셋(MSMT17)에서의 저조도 검증.

    python tools/eval_msmt.py      # → data/vast_msmt/{level}/query, results/eval_msmt.csv

- MSMT17_V1 공식 test 분할: query 11,659장 / gallery 82,161장, 3,060명, 카메라 15대 (Market과 장소·카메라·사람이 모두 다름).
- query만 configs/lowlight.yaml(Market과 같은 합성)로 low·extreme을 만들고 gallery는 원본(주간) → "야간 쿼리 → 주간 기록" 교차 조건.
- 표준 프로토콜(같은 사람·같은 카메라 제외), 코사인. Market으로만 학습한 모델에게는 처음 보는 데이터셋.
  MSMT17 train을 함께 학습한 모델(--msmt)은 같은 데이터셋이므로 표에서 구분.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from synthesize_lowlight import degrade, make_rng, sample_params  # noqa: E402
from vastreid.data import list_msmt  # noqa: E402
from vastreid.metrics import evaluate_fast  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

DEV = "cuda"
OUT = ROOT / "data/vast_msmt"
MODELS = {"사전학습 (Market)": "weights/osnet_x1_0_market1501.pth",
          "미세조정 고정 합성": "weights/osnet_x1_0_vast_lowlight_ft.pth",
          "미세조정 온라인 합성": "weights/osnet_x1_0_vast_lowlight_ft_online.pth",
          "미세조정 온라인 + MSMT17 학습*": "weights/osnet_x1_0_vast_lowlight_ft_online_msmt.pth",
          "최종 + 12 에폭 추가*": "weights/osnet_x1_0_vast_lowlight_ft_online_msmt_24.pth",
          "짝 일관성 + triplet (12 에폭)*": "weights/osnet_x1_0_vast_lowlight_ft_pair.pth",
          "짝 + triplet, extreme 2/3 (12 에폭)*": "weights/osnet_x1_0_vast_lowlight_ft_pair_ext.pth"}


def synth_queries(items, level, cfg):
    if level == "normal":
        return [x.path for x in items]
    out = []
    for x in items:
        dst = OUT / level / "query" / x.path.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            p = sample_params(cfg["levels"][level], make_rng(42, level, x.path.name))
            img, _ = degrade(cv2.imread(str(x.path)), p, make_rng(42, level + "/noise", x.path.name))
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        out.append(dst)
    return out


def ev(qf, gf, qi, gi, chunk=2000):
    """query를 나눠 평가하고 유효 query 수로 가중 평균 (82k gallery 거리 행렬을 한 번에 만들지 않기 위해)."""
    gp, gc = [x.pid for x in gi], [x.camid for x in gi]
    acc, n = {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "mINP": 0.0}, 0
    for s in range(0, len(qi), chunk):
        q = qi[s:s + chunk]
        r = evaluate_fast(-(qf[s:s + chunk] @ gf.T), [x.pid for x in q], gp, [x.camid for x in q], gc)
        k = r["num_valid_query"]
        for m in acc:
            acc[m] += r[m] * k
        n += k
    return {m: v / n for m, v in acc.items()} | {"num_valid_query": n}


def main():
    cfg = yaml.safe_load((ROOT / "configs/lowlight.yaml").read_text(encoding="utf-8"))
    qi, gi = list_msmt("query"), list_msmt("gallery")
    print(f"  MSMT17 query {len(qi)}, gallery {len(gi)}", flush=True)
    qpaths = {lv: synth_queries(qi, lv, cfg) for lv in ["normal", "low", "extreme"]}
    rows = []
    for name, w in MODELS.items():
        if not (ROOT / w).exists():
            print(f"  [skip] {name}: {w} 없음")
            continue
        m = load_model("osnet_x1_0", ROOT / w, DEV)
        gf, _ = extract(m, [x.path for x in gi], DEV)
        gf = torch.nn.functional.normalize(gf.float().to(DEV), dim=1)
        for lv, ps in qpaths.items():
            qf, _ = extract(m, ps, DEV)
            qf = torch.nn.functional.normalize(qf.float().to(DEV), dim=1)
            r = ev(qf, gf, qi, gi)
            rows.append({"model": name, "query_level": lv, "gallery_level": "normal",
                         **{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}})
            print(f"  {name:26s} {lv:7s}→normal  R1 {r['rank1']*100:5.1f}  mAP {r['mAP']*100:5.1f}  GPU {gpu_temp()}°C",
                  flush=True)
        del m, gf
        torch.cuda.empty_cache()
    with open(ROOT / "results/eval_msmt.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("[saved] results/eval_msmt.csv")


if __name__ == "__main__":
    main()
