#!/usr/bin/env python3
"""CPU 추론 속도·크기 비교 (경량화 확인용): CLIP ViT-B/16 vs OSNet x1.0.

    python tools/bench_cpu.py   # → results/bench_cpu.csv

256×128 입력, torch CPU, 스레드 수는 기본값. 사진 1장(데모 쿼리)과 64장 묶음(갤러리 색인)의 장당 시간.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.reid import load_model  # noqa: E402

MODELS = {"CLIP ViT-B/16 (최종, 선생님)": "weights/clip_vit_b16_vast_lowlight_ft_night.pth",
          "OSNet x1.0 (증류 학생)": "weights/osnet_x1_0_distill_clipnight.pth"}


@torch.inference_mode()
def per_image_ms(m, n, reps):
    x = torch.randn(n, 3, 256, 128)
    m(x)  # 워밍업
    t = time.perf_counter()
    for _ in range(reps):
        m(x)
    return (time.perf_counter() - t) / (reps * n) * 1000


def main():
    rows = []
    for name, w in MODELS.items():
        if not (ROOT / w).exists():
            print(f"  [skip] {name}: {w} 없음")
            continue
        m = load_model(None, ROOT / w, "cpu")
        params = sum(p.numel() for p in m.parameters()) / 1e6
        one, batch = per_image_ms(m, 1, 20), per_image_ms(m, 64, 3)
        size = (ROOT / w).stat().st_size / 2 ** 20
        rows.append({"model": name, "params_M": round(params, 1), "file_MB": round(size, 1),
                     "ms_per_image_b1": round(one, 1), "ms_per_image_b64": round(batch, 1)})
        print(f"  {name:28s} 파라미터 {params:5.1f}M  파일 {size:6.1f}MB  1장 {one:6.1f}ms  64장 묶음 장당 {batch:5.1f}ms", flush=True)
    with open(ROOT / "results/bench_cpu.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] results/bench_cpu.csv  (torch {torch.__version__}, threads {torch.get_num_threads()})")


if __name__ == "__main__":
    main()
