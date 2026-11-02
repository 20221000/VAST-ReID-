#!/usr/bin/env python3
"""Hugging Face Spaces 공개판 데이터 팩 생성 (샘플 갤러리만, 키·모델 가중치 없음).

    python tools/build_space.py      # → space/ (app.py, requirements.txt, README.md, data/)

- query: 데모와 같은 샘플 24명 (normal/low/extreme)
- gallery: 샘플 인물의 gallery 이미지 전부 + 무작위 다른 이미지 N장 (시드 고정) — 조도 3단계
- 특징(원래 모델, AdaBN), 속성 확률(선형 분류기), hub 점수(전체 train 참조로 미리 계산)를 npz로 저장
  → Space에서는 numpy 내적만 수행 (PyTorch 불필요)
- 결과 CSV(results/)는 벤치마크 탭용으로 복사
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.clip_attr import ATTR_VALUES  # noqa: E402
from vastreid.data import list_split  # noqa: E402

DATA = ROOT / "data/vast"
OUT = ROOT / "space"
LEVELS = ["normal", "low", "extreme"]
N_RANDOM = 800
N_SAMPLES = 24
K_HUB = 50


def feats(level, split, feat):
    tag = f"adabn-{level}" if feat == "adabn" and level != "normal" else "none"
    d = torch.load(ROOT / f"outputs/features/osnet_x1_0__{tag}__{level}__{split}.pt", weights_only=True)
    return torch.nn.functional.normalize(d["feats"].float(), dim=1)


def main():
    persons = json.loads((ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"]
    q_items = list_split(DATA, "normal", "query")
    g_items = list_split(DATA, "normal", "gallery")

    # 데모와 같은 샘플 선택 규칙
    seen, q_idx = set(), []
    for i in range(0, len(q_items), 37):
        if q_items[i].pid not in seen and f"{q_items[i].pid:04d}" in persons:
            seen.add(q_items[i].pid)
            q_idx.append(i)
        if len(q_idx) >= N_SAMPLES:
            break
    pids = {q_items[i].pid for i in q_idx}
    own = [j for j, g in enumerate(g_items) if g.pid in pids]
    rng = np.random.default_rng(42)
    rest = [j for j, g in enumerate(g_items) if g.pid not in pids]
    g_idx = sorted(own + rng.choice(rest, size=N_RANDOM, replace=False).tolist())
    print(f"[space] query {len(q_idx)}, gallery {len(g_idx)} (샘플 인물 {len(own)} + 무작위 {N_RANDOM})")

    if OUT.exists():
        shutil.rmtree(OUT / "data", ignore_errors=True)
    pack = {"q_names": np.array([q_items[i].path.name for i in q_idx]),
            "q_pids": np.array([q_items[i].pid for i in q_idx]),
            "q_cams": np.array([q_items[i].camid for i in q_idx]),
            "g_names": np.array([g_items[j].path.name for j in g_idx]),
            "g_pids": np.array([g_items[j].pid for j in g_idx]),
            "g_cams": np.array([g_items[j].camid for j in g_idx])}
    for lv in LEVELS:
        for sp, idx, items in [("query", q_idx, q_items), ("gallery", g_idx, g_items)]:
            dst = OUT / "data/images" / lv / sp
            dst.mkdir(parents=True, exist_ok=True)
            for i in idx:
                shutil.copyfile(DATA / lv / sp / items[i].path.name, dst / items[i].path.name)
        z = np.load(ROOT / f"outputs/attr_probs/probe__{lv}__gallery.npz")
        for k in ATTR_VALUES:
            pack[f"gprob__{lv}__{k}"] = z[k][g_idx].astype(np.float16)
        for feat in ["none", "adabn"]:
            qf, gf = feats(lv, "query", feat)[q_idx], feats(lv, "gallery", feat)[g_idx]
            pack[f"qf__{lv}__{feat}"] = qf.numpy().astype(np.float16)
            pack[f"gf__{lv}__{feat}"] = gf.numpy().astype(np.float16)
    # hub 점수: 서브셋 gallery(조도 g)와 전체 train(조도 q) 참조, tools/hubness.py와 같은 정의
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ql in LEVELS:
        for gl in LEVELS:
            for feat in ["none", "adabn"]:
                ref = feats(ql, "train", feat).to(dev)
                gf = feats(gl, "gallery", feat)[g_idx].to(dev)
                pack[f"hub__{ql}__{gl}__{feat}"] = (gf @ ref.T).topk(K_HUB, dim=1).values.mean(1).cpu().numpy()
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "data/index.npz", **pack)
    sub = {f"{p:04d}": persons[f"{p:04d}"] for p in sorted(set(pack["g_pids"].tolist()) | pids) if f"{p:04d}" in persons}
    (OUT / "data/persons.json").write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    (OUT / "data/results").mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "results").glob("*.csv"):
        shutil.copyfile(f, OUT / "data/results" / f.name)
    size = sum(f.stat().st_size for f in (OUT / "data").rglob("*") if f.is_file())
    print(f"[space] data pack {size / 2**20:.1f} MB → {OUT / 'data'}")


if __name__ == "__main__":
    main()
