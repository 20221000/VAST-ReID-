#!/usr/bin/env python3
"""학습 때 본 적 없는 저조도(분포 이동)로 일반화 시험.

    python tools/eval_shift.py      # → data/vast_shift/{shift}/query, results/eval_shift.csv

configs/lowlight.yaml 범위를 벗어난 합성으로 query만 새로 만들고(gallery = 주간 normal), 모델별 코사인 성능 비교.
- shift_noise: Extreme보다 노출은 약간 밝지만 노이즈가 훨씬 강하고 JPEG 화질이 낮음
- shift_color: Low 수준 밝기에 강한 주황 색 틀어짐(나트륨 가로등)
미세조정이 합성 노이즈 자체를 외웠다면 여기서 크게 떨어짐. 테스트셋으로 모델을 고르지 않고 비교만 함.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from synthesize_lowlight import degrade, make_rng, sample_params  # noqa: E402
from vastreid.data import list_split  # noqa: E402
from vastreid.metrics import evaluate_fast  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

DEV = "cuda"
SHIFTS = {
    "shift_noise": {"exposure": [0.05, 0.10], "shot_noise": [0.020, 0.040], "read_noise": [0.010, 0.020],
                    "blur_sigma": [0.0, 0.4], "color_gain_r": [1.0, 1.1], "color_gain_b": [0.85, 1.0],
                    "jpeg_quality": [35, 60]},
    "shift_color": {"exposure": [0.15, 0.30], "shot_noise": [0.004, 0.010], "read_noise": [0.002, 0.005],
                    "blur_sigma": [0.0, 0.6], "color_gain_r": [1.35, 1.6], "color_gain_b": [0.45, 0.65],
                    "jpeg_quality": [70, 90]},
}
MODELS = {"osnet_x1_0 (사전학습)": "weights/osnet_x1_0_market1501.pth",
          "미세조정 (고정 합성)": "weights/osnet_x1_0_vast_lowlight_ft.pth",
          "미세조정 (온라인 합성)": "weights/osnet_x1_0_vast_lowlight_ft_online.pth",
          "미세조정 (온라인 + MSMT17)": "weights/osnet_x1_0_vast_lowlight_ft_online_msmt.pth",
          "최종 + 12 에폭 추가": "weights/osnet_x1_0_vast_lowlight_ft_online_msmt_24.pth"}


def synth(shift: str, cfg: dict) -> list[Path]:
    out = ROOT / f"data/vast_shift/{shift}/query"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for it in list_split(ROOT / "data/vast", "normal", "query"):
        dst = out / it.path.name
        if not dst.exists():
            p = sample_params(cfg, make_rng(7, shift, it.path.name))
            img, _ = degrade(cv2.imread(str(it.path)), p, make_rng(7, shift + "/noise", it.path.name))
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        paths.append(dst)
    return paths


def main():
    qi = list_split(ROOT / "data/vast", "normal", "query")
    gi = list_split(ROOT / "data/vast", "normal", "gallery")
    qpaths = {s: synth(s, c) for s, c in SHIFTS.items()}
    lum = {s: float(np.mean([cv2.imread(str(p)).mean() / 255 for p in ps[::20]])) for s, ps in qpaths.items()}
    print("  평균 밝기", {s: round(v, 3) for s, v in lum.items()}, flush=True)
    rows = []
    for name, w in MODELS.items():
        if not (ROOT / w).exists():
            continue
        m = load_model("osnet_x1_0", ROOT / w, DEV)
        gf, _ = extract(m, [x.path for x in gi], DEV)
        gf = torch.nn.functional.normalize(gf.float().to(DEV), dim=1)
        for s, ps in [("extreme(기존)", [x.path for x in list_split(ROOT / "data/vast", "extreme", "query")]),
                      *qpaths.items()]:
            qf, _ = extract(m, ps, DEV)
            qf = torch.nn.functional.normalize(qf.float().to(DEV), dim=1)
            r = evaluate_fast(-(qf @ gf.T), [x.pid for x in qi], [x.pid for x in gi],
                              [x.camid for x in qi], [x.camid for x in gi])
            rows.append({"model": name, "query": s, "gallery": "normal",
                         **{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}})
            print(f"  {name:22s} {s:14s} R1 {r['rank1']*100:5.1f}  mAP {r['mAP']*100:5.1f}  GPU {gpu_temp()}°C", flush=True)
    with open(ROOT / "results/eval_shift.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("[saved] results/eval_shift.csv")


if __name__ == "__main__":
    main()
