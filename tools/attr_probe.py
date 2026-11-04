#!/usr/bin/env python3
"""CLIP 이미지 특징 위 선형 분류기(로지스틱 회귀)로 속성 예측 정확도 개선.

    python tools/attr_probe.py      # → outputs/attr_probs/probe__*.npz, results/attr_acc_probe.csv
    python tools/attr_probe.py --src clip+osnet-ft --name probe-ft   # 미세조정 OSNet 특징 결합

- CLIP은 고정(학습 안 함). 속성마다 로지스틱 회귀 하나 (CPU, 수 초).
- 학습 데이터: train split의 normal·low·extreme 이미지 전부 (조도에 강하도록), 정답이 unknown인 속성은 제외.
- train 이미지에 대한 예측은 **ID 기준 5-fold 교차 예측** (그 ID를 빼고 학습한 분류기로 예측).
  재순위화의 λ 검증이 train으로 이뤄지므로, 이미 본 이미지를 예측하면 속성이 완벽해 보여 λ가 부풀려짐.
- query·gallery 예측은 train 전체로 학습한 분류기 사용.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.clip_attr import ATTR_VALUES  # noqa: E402
from vastreid.data import list_split  # noqa: E402

LEVELS = ["normal", "low", "extreme"]
SRC = ROOT / "outputs/attr_probs"
C = 1.0
FOLDS = 5


SOURCE = "clip"


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load(level: str, split: str):
    """특징 출처: clip(CLIP ViT-B/16) / osnet-ft(저조도 미세조정 OSNet) / clip+osnet-ft(둘을 L2 정규화 후 이어 붙임)
    / clip+clipreid(CLIP + 저조도 미세조정 CLIP-ReID, 최종 백본)."""
    d = np.load(SRC / f"clip_ViT-B-16__{level}__{split}.npz")
    pids, parts = d["pids"], []
    if "clip" in SOURCE:
        c = d["feat"].astype(np.float32)
        parts.append(c if SOURCE == "clip" else _l2(c))  # clip 단독은 기존 결과 재현을 위해 그대로
    if "osnet-ft" in SOURCE:
        f = torch.load(ROOT / f"outputs/features/osnet_x1_0-ft__none__{level}__{split}.pt", weights_only=True)["feats"]
        if not np.array_equal(pids, np.array([x.pid for x in list_split(ROOT / "data/vast", level, split)])):
            raise RuntimeError(f"이미지 순서 불일치: {level}/{split}")
        parts.append(_l2(f.float().numpy()))
    if "clipreid" in SOURCE:
        f = torch.load(ROOT / f"outputs/features/clip_vit_b16-ftpair__none__{level}__{split}.pt", weights_only=True)["feats"]
        if len(f) != len(pids):
            raise RuntimeError(f"이미지 수 불일치: {level}/{split}")
        parts.append(_l2(f.float().numpy()))
    return pids, np.concatenate(parts, axis=1)


def labels(pids, table, attr):
    vals = ATTR_VALUES[attr]
    y = np.full(len(pids), -1)
    for i, pid in enumerate(pids):
        p = table.get(int(pid))
        if p is not None and str(p[attr]) != "unknown":
            y[i] = vals.index(str(p[attr]))
    return y


def fit_predict(Xtr, ytr, Xte, n_cls):
    m = ytr >= 0
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Xtr[m], ytr[m])
    P = np.zeros((len(Xte), n_cls), np.float32)
    P[:, clf.classes_] = clf.predict_proba(Xte)  # 학습에 없던 클래스는 확률 0
    return P


def main():
    global SOURCE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="clip", choices=["clip", "osnet-ft", "clip+osnet-ft", "clip+clipreid"])
    ap.add_argument("--name", default="probe", help="출력 이름: outputs/attr_probs/{name}__*.npz, results/attr_acc_{name}.csv")
    args = ap.parse_args()
    SOURCE = args.src
    t0 = time.time()
    table = {int(k): v for k, v in json.loads(
        (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}

    train = {lv: load(lv, "train") for lv in LEVELS}
    Xtr = np.concatenate([train[lv][1] for lv in LEVELS])
    ptr = np.concatenate([train[lv][0] for lv in LEVELS])
    uniq = np.unique(ptr)
    rng = np.random.default_rng(42)
    fold_of_pid = dict(zip(uniq, rng.permutation(len(uniq)) % FOLDS))
    fold = np.array([fold_of_pid[p] for p in ptr])

    probs = {(lv, sp): {} for lv in LEVELS for sp in ["train", "query", "gallery"]}
    test = {(lv, sp): load(lv, sp) for lv in LEVELS for sp in ["query", "gallery"]}
    offsets = np.cumsum([0] + [len(train[lv][0]) for lv in LEVELS])

    oof_acc = {}
    for attr, vals in ATTR_VALUES.items():
        y = labels(ptr, table, attr)
        # train: ID 기준 교차 예측
        oof = np.zeros((len(Xtr), len(vals)), np.float32)
        for k in range(FOLDS):
            te = fold == k
            oof[te] = fit_predict(Xtr[~te], y[~te], Xtr[te], len(vals))
        m = y >= 0
        oof_acc[attr] = float((oof[m].argmax(1) == y[m]).mean())  # 특징 출처 선택용 (테스트셋 대신)
        for i, lv in enumerate(LEVELS):
            probs[(lv, "train")][attr] = oof[offsets[i]:offsets[i + 1]]
        # query·gallery: train 전체로 학습
        m = y >= 0
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr[m], y[m])
        for key, (_, X) in test.items():
            P = np.zeros((len(X), len(vals)), np.float32)
            P[:, clf.classes_] = clf.predict_proba(X)
            probs[key][attr] = P
        print(f"  {attr} done {time.time() - t0:.0f}s", flush=True)

    for (lv, sp), d in probs.items():
        pids = train[lv][0] if sp == "train" else test[(lv, sp)][0]
        np.savez_compressed(SRC / f"{args.name}__{lv}__{sp}.npz", pids=pids, **d)
    print("  [train 교차 예측] " + "  ".join(f"{a} {v*100:.1f}" for a, v in oof_acc.items())
          + f"  | mean {np.mean(list(oof_acc.values()))*100:.1f}")

    # 정확도 (query+gallery, 정답 unknown 제외) — clip_attributes.py 와 같은 기준
    rows = []
    for lv in LEVELS:
        accs = {}
        for attr, vals in ATTR_VALUES.items():
            c = n = 0
            for sp in ["query", "gallery"]:
                y = labels(test[(lv, sp)][0], table, attr)
                pred = probs[(lv, sp)][attr].argmax(1)
                m = y >= 0
                c += int((pred[m] == y[m]).sum())
                n += int(m.sum())
            accs[attr] = c / max(n, 1)
        rows.append({"model": {"clip": "clip_ViT-B-16", "osnet-ft": "osnet_x1_0-ft",
                               "clip+osnet-ft": "clip_ViT-B-16+osnet_x1_0-ft",
                               "clip+clipreid": "clip_ViT-B-16+clip_vit_b16-ftpair"}[SOURCE] + "+linear_probe", "level": lv, **{a: round(v, 4) for a, v in accs.items()},
                     "mean_acc": round(float(np.mean(list(accs.values()))), 4)})
        print(f"  [{lv}] " + "  ".join(f"{a} {v*100:.1f}" for a, v in accs.items())
              + f"  | mean {np.mean(list(accs.values()))*100:.1f}")
    out = ROOT / f"results/attr_acc_{args.name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {time.time() - t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
