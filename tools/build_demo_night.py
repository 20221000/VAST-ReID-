#!/usr/bin/env python3
"""웹 데모용 Night600(실제 야간) k-reciprocal 거리 캐시.

    python tools/eval_night.py && python tools/build_demo_night.py   # → outputs/demo/krfull_night__{prefix}.npy

tools/eval_night.py가 저장한 특징(outputs/features/night__{prefix}__{query,gallery}.pt)으로 전체 query(2,180) × gallery(14,351)의
k-reciprocal 거리(k1=20, k2=6, λ=0.3)를 미리 계산. 데모 기본 백본과 OSNet 최종 백본만.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchreid.utils.rerank import re_ranking

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid import demo_core as core  # noqa: E402


def main():
    for pf in core.TRAINED_MSMT:
        if core.night_feats(pf, "gallery") is None:
            print(f"  [skip] {pf}: 특징 없음 (tools/eval_night.py 먼저)")
            continue
        out = ROOT / f"outputs/demo/krfull_night__{pf}.npy"
        t = time.time()
        qf = torch.from_numpy(core.night_feats(pf, "query")).to(core.DEV)
        gf = torch.from_numpy(core.night_feats(pf, "gallery")).to(core.DEV)
        d = re_ranking((1 - qf @ gf.T).cpu().numpy(), (1 - qf @ qf.T).cpu().numpy(), (1 - gf @ gf.T).cpu().numpy(),
                       k1=20, k2=6, lambda_value=0.3)
        np.save(out, d.astype(np.float16))
        print(f"  [krfull_night] {pf} {d.shape} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
