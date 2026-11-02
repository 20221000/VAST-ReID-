#!/usr/bin/env python3
"""속성 재순위화: score = cos(f_q, f_g) + λ · attr_agreement(q, g)

    python tools/rerank.py --attr-source oracle --out results/rerank_oracle.csv

- attr_agreement: 속성별로 두 확률분포의 내적(= 같은 값일 확률)을 구해, 양쪽 모두 알려진 속성에 대해 평균.
  정답 속성은 one-hot(unknown은 제외), 예측 속성은 CLIP 확률. 비교할 속성이 없으면(distractor 등) 0.5(중립).
- λ는 **train 이미지로 만든 검증 분할**에서 mAP 최대가 되도록 고름 (테스트셋 미사용).
  검증 분할: train ID마다 카메라별 첫 이미지를 query, 나머지를 gallery.
- attr-source
    oracle  : query·gallery 모두 정답 속성 (속성이 완벽히 주어졌을 때의 상한선, 실제 시스템 아님)
    clip    : query·gallery 모두 CLIP zero-shot 예측 (완전 자동)
    witness : query는 사람이 묘사한 속성(정답으로 대체), gallery는 CLIP 예측 (목격자 진술 + 영상 검색 시나리오)
    probe / witness-probe : 위와 같되 예측을 CLIP 특징 위 선형 분류기(tools/attr_probe.py)로
- 특징은 outputs/features/ 에 캐시 (L2 정규화 전 원본).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.clip_attr import ATTR_VALUES  # noqa: E402
from vastreid.data import Item, list_split  # noqa: E402
from vastreid.metrics import evaluate  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

ATTRS = ["upper_color", "lower_color", "sleeve", "lower_type", "bag", "hat"]


def get_feats(model, data: Path, level: str, split: str, device: str, cache_dir: Path,
              tag: str = "none", prefix: str = "osnet_x1_0") -> tuple[list[Item], torch.Tensor]:
    """tag='adabn' 이면 저조도 이미지는 AdaBN 특징 캐시 사용 (normal은 원래 모델).
    prefix: 특징 캐시의 모델 이름 (기본 osnet_x1_0, 저조도 미세조정은 osnet_x1_0-ft — tools/finetune.py가 생성)."""
    items = list_split(data, level, split)
    if tag == "adabn" and level != "normal":
        cp = cache_dir / f"{prefix}__adabn-{level}__{level}__{split}.pt"
        if not cp.exists():
            raise FileNotFoundError(f"{cp} 없음 → tools/adabn.py 또는 tools/finetune.py 먼저 실행")
        return items, torch.load(cp, weights_only=True)["feats"]
    cp = cache_dir / f"{prefix}__none__{level}__{split}.pt"
    if prefix != "osnet_x1_0" and not cp.exists():  # 다른 모델의 특징을 기본 모델로 계산하지 않도록
        raise FileNotFoundError(f"{cp} 없음 → tools/finetune.py 먼저 실행")
    if cp.exists():
        d = torch.load(cp, weights_only=True)
        if d["n"] == len(items):
            return items, d["feats"]
    f, _ = extract(model, [x.path for x in items], device)
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n": len(items), "feats": f.cpu()}, cp)
    return items, f.cpu()


AttrRep = dict  # attr -> (probs (N, V) float32, mask (N,) bool)


def gt_rep(pids: list[int], table: dict[int, dict]) -> AttrRep:
    rep = {}
    for k, vals in ATTR_VALUES.items():
        P = np.zeros((len(pids), len(vals)), np.float32)
        M = np.zeros(len(pids), bool)
        for i, pid in enumerate(pids):
            a = table.get(pid)
            if a is None or str(a[k]) == "unknown":
                continue
            P[i, vals.index(str(a[k]))] = 1.0
            M[i] = True
        rep[k] = (P, M)
    return rep


def clip_rep(level: str, split: str, items: list[Item], idx=None, pred: str = "clip") -> AttrRep:
    name = "clip_ViT-B-16" if pred == "clip" else pred  # "probe" / "probe-ft" (tools/attr_probe.py --name)
    d = np.load(ROOT / f"outputs/attr_probs/{name}__{level}__{split}.npz")
    if not np.array_equal(d["pids"], np.array([x.pid for x in items])):
        raise RuntimeError(f"CLIP 확률과 이미지 목록 불일치: {level}/{split} (tools/clip_attributes.py 재실행)")
    sel = slice(None) if idx is None else np.asarray(idx)
    return {k: (d[k][sel].astype(np.float32), np.ones(len(d[k][sel]), bool)) for k in ATTR_VALUES}


def agreement(qa: AttrRep, ga: AttrRep, device: str) -> torch.Tensor:
    num = den = None
    for k in ATTR_VALUES:
        qp, qm = (torch.from_numpy(x).to(device) for x in qa[k])
        gp, gm = (torch.from_numpy(x).to(device) for x in ga[k])
        both = qm.float()[:, None] * gm.float()[None, :]
        term = (qp @ gp.T) * both
        num = term if num is None else num + term
        den = both if den is None else den + both
    return torch.where(den > 0, num / den.clamp(min=1), torch.full_like(den, 0.5))


def run(qf, gf, qi, gi, qa, ga, lam, device) -> dict:
    qf = torch.nn.functional.normalize(qf.to(device), dim=1)
    gf = torch.nn.functional.normalize(gf.to(device), dim=1)
    score = qf @ gf.T
    if lam:
        score = score + lam * agreement(qa, ga, device)
    return evaluate((-score).cpu().numpy(), [x.pid for x in qi], [x.pid for x in gi],
                    [x.camid for x in qi], [x.camid for x in gi])


def val_split(items: list[Item]) -> tuple[list[int], list[int]]:
    seen, q, g = set(), [], []
    for i, it in enumerate(items):
        key = (it.pid, it.camid)
        (g if key in seen else q).append(i)
        seen.add(key)
    return q, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/vast")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/osnet_x1_0_market1501.pth")
    ap.add_argument("--attr-source", default="oracle", choices=["oracle", "clip", "witness", "probe", "witness-probe"])
    ap.add_argument("--levels", nargs="+", default=["normal", "low", "extreme"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--out", type=Path, default=None, help="기본: results/rerank_<attr-source>.csv")
    args = ap.parse_args()
    args.out = args.out or ROOT / f"results/rerank_{args.attr_source}.csv"
    pred = "probe" if "probe" in args.attr_source else "clip"
    gt_query = args.attr_source in ("oracle", "witness", "witness-probe")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    persons = json.loads((ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"]
    table = {int(k): v for k, v in persons.items()}
    cache = ROOT / "outputs/features"
    model = load_model("osnet_x1_0", args.weights, device)
    t0 = time.time()

    rows = []
    for lv in args.levels:
        pairs = [("same", lv, lv)] + ([("cross", lv, "normal")] if lv != "normal" else [])
        for proto, ql, gl in pairs:
            # ---- 검증 분할(train)에서 λ 선택
            tq_items, tq_f = get_feats(model, args.data, ql, "train", device, cache)
            tg_items, tg_f = get_feats(model, args.data, gl, "train", device, cache)
            qi_idx, gi_idx = val_split(tq_items)
            vq = [tq_items[i] for i in qi_idx]
            vg = [tg_items[i] for i in gi_idx]
            vqa = (gt_rep([x.pid for x in vq], table) if gt_query
                   else clip_rep(ql, "train", tq_items, qi_idx, pred))
            vga = (gt_rep([x.pid for x in vg], table) if args.attr_source == "oracle"
                   else clip_rep(gl, "train", tg_items, gi_idx, pred))
            val = {lam: run(tq_f[qi_idx], tg_f[gi_idx], vq, vg, vqa, vga, lam, device)["mAP"] for lam in args.lambdas}
            best = max(val, key=val.get)

            # ---- 테스트셋 평가 (λ=0, λ*)
            qi, qf = get_feats(model, args.data, ql, "query", device, cache)
            gi, gf = get_feats(model, args.data, gl, "gallery", device, cache)
            qa = (gt_rep([x.pid for x in qi], table) if gt_query
                  else clip_rep(ql, "query", qi, pred=pred))
            ga = (gt_rep([x.pid for x in gi], table) if args.attr_source == "oracle"
                  else clip_rep(gl, "gallery", gi, pred=pred))
            for tag, lam in [("cosine", 0.0), ("rerank", best)]:
                r = run(qf, gf, qi, gi, qa, ga, lam, device)
                rows.append({"attr_source": args.attr_source, "method": tag, "lambda": lam, "protocol": proto,
                             "query_level": ql, "gallery_level": gl,
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                             "val_mAP_at_lambda": round(val[lam], 4)})
                print(f"  {proto:5s} q={ql:7s} g={gl:7s} {tag:6s} λ={lam:<4} R1 {r['rank1']*100:5.1f}  "
                      f"mAP {r['mAP']*100:5.1f}  mINP {r['mINP']*100:5.1f}", flush=True)
            print("      val mAP by λ: " + ", ".join(f"{k}:{v*100:.1f}" for k, v in val.items()))
    print(f"[done] {time.time() - t0:.1f}s  GPU {gpu_temp()}°C")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
