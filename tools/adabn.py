#!/usr/bin/env python3
"""AdaBN: 저조도 이미지로 BatchNorm 통계(평균·분산)만 다시 계산. 가중치 업데이트·라벨 없음.

    python tools/adabn.py        # → results/adabn.csv, outputs/features/osnet_x1_0__adabn-{level}__*.pt

- 조도 L(low/extreme)용 모델: 원래 OSNet을 복사한 뒤, L 조도의 train 이미지(라벨 미사용)를 통과시켜
  모든 BatchNorm의 running mean/var를 누적 평균으로 다시 추정 (Li et al., "Revisiting Batch Normalization
  for Practical Domain Adaptation", 2016).
- 특징 추출: L 조도 이미지는 L용 모델, normal 이미지는 원래 모델.
  same(L→L): query·gallery 모두 L용 모델 / cross(L→normal): query는 L용, gallery는 원래 모델.
- 조정할 하이퍼파라미터가 없어 검증 분할 선택 과정이 필요 없음. 테스트 이미지는 통계 추정에 사용하지 않음.
"""
from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.data import list_split  # noqa: E402
from vastreid.metrics import evaluate  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = ROOT / "data/vast"
CACHE = ROOT / "outputs/features"


@torch.no_grad()
def adapt_bn(base, level: str):
    m = copy.deepcopy(base)
    bns = [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # 누적 평균
    m.eval()
    for bn in bns:
        bn.train()  # BN만 통계 갱신 모드 (dropout 등은 eval 유지)
    paths = [x.path for x in list_split(DATA, level, "train")]
    extract(m, paths, DEV)  # forward만 수행 → running stats 갱신
    m.eval()
    return m, len(bns), len(paths)


def feats(model, tag, level, split):
    items = list_split(DATA, level, split)
    cp = CACHE / f"osnet_x1_0__{tag}__{level}__{split}.pt"
    if cp.exists():
        d = torch.load(cp, weights_only=True)
        if d["n"] == len(items):
            return items, d["feats"]
    f, _ = extract(model, [x.path for x in items], DEV)
    torch.save({"n": len(items), "feats": f.cpu()}, cp)
    return items, f.cpu()


def ev(qi, qf, gi, gf, metric):
    if metric == "cosine":
        qf, gf = torch.nn.functional.normalize(qf, dim=1), torch.nn.functional.normalize(gf, dim=1)
        dist = -(qf @ gf.T)
    else:
        dist = torch.cdist(qf, gf)
    return evaluate(dist.numpy(), [x.pid for x in qi], [x.pid for x in gi],
                    [x.camid for x in qi], [x.camid for x in gi])


def main():
    t0 = time.time()
    base = load_model("osnet_x1_0", ROOT / "weights/osnet_x1_0_market1501.pth", DEV)
    rows = []
    for lv in ["low", "extreme"]:
        t = time.time()
        m, n_bn, n_img = adapt_bn(base, lv)
        print(f"[adabn] {lv}: BN {n_bn}개 통계 재추정 ({n_img} train imgs, {time.time() - t:.0f}s)  GPU {gpu_temp()}°C",
              flush=True)
        tag = f"adabn-{lv}"
        feats(m, tag, lv, "train")  # 검증 분할·hub 참조 집합용 (tools/hubness.py --feat adabn)
        for proto, gl in [("same", lv), ("cross", "normal")]:
            qi, qf0 = feats(base, "none", lv, "query")
            qi, qf1 = feats(m, tag, lv, "query")
            gi, gf0 = feats(base, "none", gl, "gallery")
            gi, gf1 = (feats(m, tag, gl, "gallery") if gl == lv else (gi, gf0))
            for metric in ["euclidean", "cosine"]:
                for name, (a, b) in [("baseline", (qf0, gf0)), ("adabn", (qf1, gf1))]:
                    r = ev(qi, a, gi, b, metric)
                    rows.append({"method": name, "metric": metric, "protocol": proto, "query_level": lv,
                                 "gallery_level": gl, **{k: round(v, 4) if isinstance(v, float) else v
                                                         for k, v in r.items()}})
                    print(f"  {proto:5s} {lv:7s}→{gl:7s} {metric:9s} {name:8s} R1 {r['rank1']*100:5.1f}  "
                          f"mAP {r['mAP']*100:5.1f}  mINP {r['mINP']*100:5.1f}", flush=True)
    print(f"[done] {time.time() - t0:.1f}s  GPU {gpu_temp()}°C")
    out = ROOT / "results/adabn.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
