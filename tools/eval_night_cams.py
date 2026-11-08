#!/usr/bin/env python3
"""처음 보는 야간 카메라 검증 (Night600 카메라 분할).

    python tools/eval_night_cams.py   # → results/eval_night_cams.csv

최종 모델은 Night600 train(test와 다른 300명)을 카메라 8대 전부에서 학습했으므로, test 카메라가 학습 때 본 카메라다.
처음 보는 카메라에서도 통하는지 확인하려고 train을 카메라 1~4 사진만으로 다시 학습하고(`finetune.py --night-cams 1,2,3,4`),
test는 카메라 5~8끼리(query·gallery 모두 카메라 5~8)만 평가한다. 비교를 위해 카메라 1~4끼리도 평가.

- 평가 프로토콜은 tools/eval_night.py와 같음(같은 사람·같은 카메라 제외, 코사인 / CSLS k=50 β=1 / k-reciprocal + CSLS β=0.1).
- CSLS 참조는 eval_night.py와 같은 train 3,000장(라벨 미사용) 중 해당 카메라 그룹 사진만 사용.
- eval_night.py가 저장한 특징(outputs/features/night__{prefix}__{split}.pt)을 다시 쓰고, 없으면 추출해 저장.
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
from vastreid.reid import extract, load_model  # noqa: E402

DEV = "cuda"
N_REF = 3000  # eval_night.py와 같은 참조 표본
GROUPS = {"카메라 5~8 (학습 때 안 본 카메라)": {4, 5, 6, 7}, "카메라 1~4 (학습 때 본 카메라)": {0, 1, 2, 3}}
MODELS = {"CLIP 합성만 (Night600 학습 없음)": ("clip_vit_b16-ftpair", "weights/clip_vit_b16_vast_lowlight_ft_pair.pth"),
          "CLIP + Night600 train 카메라 1~4만": ("clip_vit_b16-ftnight-c1234", "weights/clip_vit_b16_vast_lowlight_ft_night_c1234.pth"),
          "카메라 1~4 학습 + 5~8 라벨 없이 적응": ("clip_vit_b16-ftnight-c1234-adapt58", "weights/clip_vit_b16_vast_lowlight_ft_night_c1234_adapt58.pth"),
          "CLIP + Night600 train 카메라 8대 전부 (최종)": ("clip_vit_b16-ftnight", "weights/clip_vit_b16_vast_lowlight_ft_night.pth")}


def feats(prefix: str, w: str, splits: dict) -> dict:
    cache, out, m = ROOT / "outputs/features", {}, None
    for sp, its in splits.items():
        p = cache / f"night__{prefix}__{sp}.pt"
        if p.exists() and torch.load(p, weights_only=True)["n"] == len(its):
            x = torch.load(p, weights_only=True)["feats"]
        else:
            m = m or load_model(None, ROOT / w, DEV)
            x, _ = extract(m, [it.path for it in its], DEV)
            torch.save({"n": len(its), "feats": x.half().cpu()}, p)
        out[sp] = torch.nn.functional.normalize(x.float().to(DEV), dim=1)
    if m is not None:
        del m
        torch.cuda.empty_cache()
    return out


def main():
    qi, gi, tr = list_night600("query"), list_night600("gallery"), list_night600("train")
    ref = [tr[i] for i in sorted(random.Random(0).sample(range(len(tr)), N_REF))]
    rows = []
    for name, (prefix, w) in MODELS.items():
        if not (ROOT / w).exists():
            print(f"  [skip] {name}: {w} 없음")
            continue
        f = feats(prefix, w, {"query": qi, "gallery": gi, "ref": ref})
        for gname, cams in GROUPS.items():
            qm = torch.tensor([x.camid in cams for x in qi], device=DEV)
            gm = torch.tensor([x.camid in cams for x in gi], device=DEV)
            rm = torch.tensor([x.camid in cams for x in ref], device=DEV)
            q, g, r = f["query"][qm], f["gallery"][gm], f["ref"][rm]
            qs = [x for x, k in zip(qi, qm.tolist()) if k]
            gs = [x for x, k in zip(gi, gm.tolist()) if k]
            cos = q @ g.T
            hub = (g @ r.T).topk(50, dim=1).values.mean(1)
            kr = -torch.from_numpy(re_ranking((1 - cos).cpu().numpy(), (1 - q @ q.T).cpu().numpy(),
                                               (1 - g @ g.T).cpu().numpy(), k1=20, k2=6, lambda_value=0.3)).to(DEV)
            for method, s in [("cosine", cos), ("csls", cos - hub[None, :]), ("kr+csls", kr - 0.1 * hub[None, :])]:
                res = evaluate_fast(-s, [x.pid for x in qs], [x.pid for x in gs], [x.camid for x in qs], [x.camid for x in gs])
                rows.append({"model": name, "prefix": prefix, "cams": gname, "method": method, "n_query": len(qs),
                             "n_gallery": len(gs), **{k: round(v, 6) if isinstance(v, float) else v for k, v in res.items()}})
                print(f"  {name:32s} {gname:22s} {method:7s} R1 {res['rank1'] * 100:5.1f}  R5 {res['rank5'] * 100:5.1f}  "
                      f"mAP {res['mAP'] * 100:5.1f}  (query {len(qs)}, gallery {len(gs)})", flush=True)
    with open(ROOT / "results/eval_night_cams.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("[saved] results/eval_night_cams.csv")


if __name__ == "__main__":
    main()
