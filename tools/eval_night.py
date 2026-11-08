#!/usr/bin/env python3
"""실제 야간 데이터셋(Night600)으로 검증. 어떤 모델도 Night600으로 학습하지 않음 (zero-shot).

    python tools/eval_night.py      # → results/eval_night.csv, outputs/features/night__{prefix}__{split}.pt

- † 표시 모델만 Night600 train 300명으로 학습 (test 300명과 사람이 겹치지 않음). 나머지는 zero-shot.
- Night600 (IDF, IEEE TMM 2024, github.com/Alexadlu/IDF): 600명 중 test 300명, query 2,180장 / gallery 14,351장, 카메라 8대.
  query·gallery 모두 실제 야간 사진 (합성 아님). 표준 프로토콜(같은 사람·같은 카메라 제외), 코사인.
- CSLS(hub 보정) 참조는 train 이미지(라벨 미사용) 3,000장, k=50·β=1 고정. kr+csls: k-reciprocal(k1=20, k2=6, λ=0.3) 위 β=0.1(Market 검증값).
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import torch
from torchreid.utils.rerank import re_ranking

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.data import list_night600  # noqa: E402
from vastreid.metrics import evaluate_fast  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

DEV = "cuda"
N_REF = 3000
MODELS = {"사전학습 OSNet (Market)": ("osnet_x1_0", "weights/osnet_x1_0_market1501.pth"),
          "OSNet 미세조정 (Market, 고정 합성)": ("osnet_x1_0-ft", "weights/osnet_x1_0_vast_lowlight_ft.pth"),
          "OSNet 최종 (Market + MSMT17, 24 에폭)": ("osnet_x1_0-ftmsmt24", "weights/osnet_x1_0_vast_lowlight_ft_online_msmt_24.pth"),
          "OSNet 짝 + triplet (24 에폭)": ("osnet_x1_0-ftpairext24", "weights/osnet_x1_0_vast_lowlight_ft_pair_ext_24.pth"),
          "OSNet 증류 학생 (최종 CLIP → OSNet)†": ("osnet_x1_0-distill", "weights/osnet_x1_0_distill_clipnight.pth"),
          "CLIP ViT-B/16 짝 + triplet (12 에폭)": ("clip_vit_b16-ftpair", "weights/clip_vit_b16_vast_lowlight_ft_pair.pth"),
          "CLIP ViT-B/16 + 노이즈 종류 다양화 (12 에폭)": ("clip_vit_b16-ftnoise", "weights/clip_vit_b16_vast_lowlight_ft_noise.pth"),
          "CLIP ViT-B/16 짝 + triplet (24 에폭)": ("clip_vit_b16-ftpair24", "weights/clip_vit_b16_vast_lowlight_ft_pair_24.pth"),
          "CLIP ViT-B/16 + 카메라 처리 합성 (12 에폭)": ("clip_vit_b16-ftcam", "weights/clip_vit_b16_vast_lowlight_ft_cam.pth"),
          "CLIP ViT-B/16 + Night600 train (24 에폭)†": ("clip_vit_b16-ftnight", "weights/clip_vit_b16_vast_lowlight_ft_night.pth")}


def main():
    qi, gi, tr = list_night600("query"), list_night600("gallery"), list_night600("train")
    ref = [tr[i] for i in sorted(random.Random(0).sample(range(len(tr)), N_REF))]
    print(f"  Night600 query {len(qi)}, gallery {len(gi)}, ref {len(ref)}", flush=True)
    cache = ROOT / "outputs/features"
    rows = []
    for name, (prefix, w) in MODELS.items():
        if not (ROOT / w).exists():
            print(f"  [skip] {name}: {w} 없음")
            continue
        m = load_model(None, ROOT / w, DEV)
        f = {}
        for sp, its in [("query", qi), ("gallery", gi), ("ref", ref)]:
            x, _ = extract(m, [it.path for it in its], DEV)
            torch.save({"n": len(its), "feats": x.half().cpu()}, cache / f"night__{prefix}__{sp}.pt")
            f[sp] = torch.nn.functional.normalize(x.float().to(DEV), dim=1)
        del m
        torch.cuda.empty_cache()
        cos = f["query"] @ f["gallery"].T
        hub = (f["gallery"] @ f["ref"].T).topk(50, dim=1).values.mean(1)
        kr = -torch.from_numpy(re_ranking((1 - cos).cpu().numpy(), (1 - f["query"] @ f["query"].T).cpu().numpy(),
                                           (1 - f["gallery"] @ f["gallery"].T).cpu().numpy(), k1=20, k2=6,
                                           lambda_value=0.3)).to(DEV)
        for method, s in [("cosine", cos), ("csls", cos - hub[None, :]), ("kr+csls", kr - 0.1 * hub[None, :])]:
            r = evaluate_fast(-s, [x.pid for x in qi], [x.pid for x in gi], [x.camid for x in qi], [x.camid for x in gi])
            rows.append({"model": name, "prefix": prefix, "method": method,
                         **{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}})
            print(f"  {name:34s} {method:6s} R1 {r['rank1']*100:5.1f}  R5 {r['rank5']*100:5.1f}  mAP {r['mAP']*100:5.1f}  "
                  f"GPU {gpu_temp()}°C", flush=True)
    with open(ROOT / "results/eval_night.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("[saved] results/eval_night.csv")


if __name__ == "__main__":
    main()
