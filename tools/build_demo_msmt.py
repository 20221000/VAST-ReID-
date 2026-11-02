#!/usr/bin/env python3
"""데모 무작위 샘플용 캐시.

    python tools/build_demo_msmt.py

1) MSMT17 특징 (백본 3종): gallery(주간 82,161장), query(normal·low·extreme, data/vast_msmt), CSLS 참조용 train 일부
   (3,000장을 같은 방식으로 합성, 라벨 사용 안 함) → outputs/features/msmt__{prefix}__{level}__{split}.pt
2) Market 최종 모델의 k-reciprocal 거리 (query 3,368개 전체 × gallery, 쿼리 조도별, AdaBN 없음)
   → outputs/demo/krfull__{prefix}__{level}__none.npy (float16). 무작위로 고른 Market 인물에도 재순위화 적용.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchreid.utils.rerank import re_ranking

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from synthesize_lowlight import degrade, make_rng, sample_params  # noqa: E402
from vastreid import demo_core as core  # noqa: E402
from vastreid.data import list_msmt  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

DEV = "cuda"
WEIGHTS = {core.BASE: "weights/osnet_x1_0_market1501.pth", core.FT: "weights/osnet_x1_0_vast_lowlight_ft.pth",
           core.FTM: "weights/osnet_x1_0_vast_lowlight_ft_online_msmt_24.pth"}
N_REF = 3000


def synth(items, level, sub, cfg):
    if level == "normal":
        return [x.path for x in items]
    out = []
    for x in items:
        dst = ROOT / f"data/vast_msmt/{level}/{sub}/{x.path.name}"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            p = sample_params(cfg["levels"][level], make_rng(42, level, x.path.name))
            img, _ = degrade(cv2.imread(str(x.path)), p, make_rng(42, level + "/noise", x.path.name))
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        out.append(dst)
    return out


def main():
    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs/lowlight.yaml").read_text(encoding="utf-8"))
    qi, gi = list_msmt("query"), list_msmt("gallery")
    tr = list_msmt("train")
    ref = [tr[i] for i in sorted(random.Random(0).sample(range(len(tr)), N_REF))]
    paths = {("normal", "gallery"): [x.path for x in gi]}
    for lv in core.LEVELS:
        paths[(lv, "query")] = synth(qi, lv, "query", cfg)
        paths[(lv, "ref")] = synth(ref, lv, "train_ref", cfg)
    print(f"  [synth] {time.time() - t0:.0f}s", flush=True)
    cache = ROOT / "outputs/features"
    for pf, w in WEIGHTS.items():
        m = load_model("osnet_x1_0", ROOT / w, DEV)
        for (lv, sp), ps in paths.items():
            f, _ = extract(m, ps, DEV)
            torch.save({"n": len(ps), "feats": f.half().cpu()}, cache / f"msmt__{pf}__{lv}__{sp}.pt")
        print(f"  [msmt feat] {pf} {time.time() - t0:.0f}s GPU {gpu_temp()}°C", flush=True)
        del m
        torch.cuda.empty_cache()
    # Market 최종 모델 전체 쿼리 k-reciprocal
    gf = torch.from_numpy(core.feats("normal", "gallery", "none", core.FTM)).to(DEV)
    gg = (1 - gf @ gf.T).cpu().numpy()
    for lv in core.LEVELS:
        qf = torch.from_numpy(core.feats(lv, "query", "none", core.FTM)).to(DEV)
        t = time.time()
        d = re_ranking((1 - qf @ gf.T).cpu().numpy(), (1 - qf @ qf.T).cpu().numpy(), gg, k1=20, k2=6, lambda_value=0.3)
        np.save(ROOT / f"outputs/demo/krfull__{core.FTM}__{lv}__none.npy", d.astype(np.float16))
        print(f"  [krfull] {lv} {d.shape} {time.time() - t:.0f}s GPU {gpu_temp()}°C", flush=True)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
