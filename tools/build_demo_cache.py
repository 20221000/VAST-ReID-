#!/usr/bin/env python3
"""웹 데모용 k-reciprocal 거리 캐시 (샘플 쿼리 24개 × 쿼리 조도 3 × AdaBN on/off).

    python tools/build_demo_cache.py                          # → outputs/demo/kr__{q_level}__{feat}.npy
    python tools/build_demo_cache.py --prefix osnet_x1_0-ft   # → outputs/demo/kr__osnet_x1_0-ft__{q_level}__{feat}.npy

k-reciprocal은 gallery 전체 이웃 관계가 필요해 쿼리마다 계산하면 20초 이상 걸리므로 미리 계산.
tools/kreciprocal.py와 같은 순서: 원래 코사인 거리에 k-reciprocal(k1=20, k2=6, λ=0.3)만 적용해 저장하고,
CSLS·속성은 데모에서 results/kreciprocal.csv의 β·λ로 더함. gallery = normal.
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=core.BASE)
    pf = ap.parse_args().prefix
    pre = "" if pf == core.BASE else f"{pf}__"
    out = ROOT / "outputs/demo"
    out.mkdir(parents=True, exist_ok=True)
    idx = core.sample_idx()
    for feat in ["none", "adabn"]:
        gf = torch.from_numpy(core.feats("normal", "gallery", feat, pf)).to(core.DEV)
        gg = (1 - gf @ gf.T).cpu().numpy()
        for ql in core.LEVELS:
            qf = torch.from_numpy(core.feats(ql, "query", feat, pf)[idx]).to(core.DEV)
            qq = (1 - qf @ qf.T).cpu().numpy()
            t = time.time()
            d = re_ranking((1 - qf @ gf.T).cpu().numpy(), qq, gg, k1=20, k2=6, lambda_value=0.3)
            np.save(out / f"kr__{pre}{ql}__{feat}.npy", d.astype(np.float16))
            print(f"  kr {ql:7s} {feat:5s} {d.shape} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
