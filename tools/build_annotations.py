#!/usr/bin/env python3
"""Market-1501 Attribute → VAST-ReID 속성 6개 + 템플릿 캡션 + 이미지별 어노테이션.

    python tools/build_annotations.py --attr data/raw/market_attribute/market_attribute.mat --vast data/vast

출력
  annotations/person_attributes.json          (커밋용, ID 1,501명 단위 속성·캡션)
  data/vast/annotations/{train,query,gallery}.json  (이미지×조도 단위, HANDOFF §4-5 스키마, 용량이 커서 gitignore)

윤리 원칙: gender, age, hair(성별을 간접적으로 드러낼 수 있음)는 사용하지 않음.
원본 라벨 값: 이진 속성은 1=없음, 2=있음 / up: 1=긴소매, 2=짧은소매 / down: 1=긴 하의, 2=짧은 하의 / clothes: 1=치마·원피스, 2=바지
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.data import list_split  # noqa: E402

EXCLUDED = ["gender", "age", "hair"]
UP_COLORS = ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"]
DOWN_COLORS = ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"]
ATTRIBUTES = {
    "upper_color": UP_COLORS + ["unknown"],
    "lower_color": DOWN_COLORS + ["unknown"],
    "sleeve": ["long", "short"],
    "lower_type": ["pants", "shorts", "skirt"],
    "bag": ["backpack", "handbag", "bag", "none"],
    "hat": [False, True],
}
KO = {
    "black": "검은", "white": "흰", "red": "빨간", "purple": "보라", "yellow": "노란", "gray": "회색",
    "blue": "파란", "green": "초록", "pink": "분홍", "brown": "갈색", "unknown": "",
    "long": "긴팔", "short": "반팔", "pants": "긴 바지", "shorts": "반바지", "skirt": "치마",
    "backpack": "백팩을 멘", "handbag": "핸드백을 든", "bag": "가방을 든", "none": "",
}


def color_of(rec, prefix: str, colors: list[str]) -> str:
    on = [c for c in colors if int(getattr(rec, prefix + c, 1)) == 2]
    return on[0] if len(on) == 1 else "unknown"  # 없거나 여러 개면 unknown


def map_person(rec, i: int) -> dict:
    g = lambda f: int(np.asarray(getattr(rec, f))[i])
    up = {c: g("up" + c) for c in UP_COLORS}
    down = {c: g("down" + c) for c in DOWN_COLORS if hasattr(rec, "down" + c)}
    upper = [c for c, v in up.items() if v == 2]
    lower = [c for c, v in down.items() if v == 2]
    if g("clothes") == 1:
        lower_type = "skirt"
    else:
        lower_type = "pants" if g("down") == 1 else "shorts"
    bag = "none"
    for b in ["backpack", "handbag", "bag"]:
        if g(b) == 2:
            bag = b
            break
    return {
        "upper_color": upper[0] if len(upper) == 1 else "unknown",
        "lower_color": lower[0] if len(lower) == 1 else "unknown",
        "sleeve": "long" if g("up") == 1 else "short",
        "lower_type": lower_type,
        "bag": bag,
        "hat": g("hat") == 2,
    }


def caption(a: dict) -> str:
    up = " ".join(x for x in [KO[a["upper_color"]], KO[a["sleeve"]], "상의"] if x)
    low = " ".join(x for x in [KO[a["lower_color"]], KO[a["lower_type"]]] if x)
    s = f"{up}와 {low}를 입은"
    if a["bag"] != "none":
        s = f"{up}와 {low}를 입고 {KO[a['bag']]}"
    s += " 사람"
    if a["hat"]:
        s = "모자를 쓴 " + s
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", type=Path, default=ROOT / "data/raw/market_attribute/market_attribute.mat")
    ap.add_argument("--vast", type=Path, default=ROOT / "data/vast")
    ap.add_argument("--out", type=Path, default=ROOT / "annotations/person_attributes.json")
    args = ap.parse_args()

    ma = sio.loadmat(args.attr, squeeze_me=True, struct_as_record=False)["market_attribute"]
    persons = {}
    for split in ["train", "test"]:
        rec = getattr(ma, split)
        for i, pid in enumerate(np.asarray(rec.image_index)):
            a = map_person(rec, i)
            persons[int(pid)] = {"split": split, **a, "caption": caption(a)}
    print(f"[attr] {len(persons)} persons")
    for k in ATTRIBUTES:
        print(f"  {k:12s}", dict(Counter(str(p[k]) for p in persons.values()).most_common()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": "Market-1501_Attribute (Lin et al.), github.com/vana77/Market-1501_Attribute",
        "excluded_attributes": EXCLUDED,
        "attribute_values": {k: [str(v) for v in vs] for k, vs in ATTRIBUTES.items()},
        "persons": {f"{pid:04d}": v for pid, v in sorted(persons.items())},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {args.out}")

    # 이미지 × 조도 어노테이션 (HANDOFF §4-5 스키마)
    log = {}
    with open(args.vast / "synthesis_log.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            log[(r["level"], r["file"])] = r
    out_dir = args.vast / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "query", "gallery"]:
        entries, missing = [], 0
        for level in ["normal", "low", "extreme"]:
            for it in list_split(args.vast, level, split, drop_junk=False):
                rel = f"{split}/{it.path.name}"
                lg = log[(level, rel)]
                p = persons.get(it.pid)
                if it.pid > 0 and p is None:
                    missing += 1
                entries.append({
                    "image_id": f"{it.path.stem}__{level}",
                    "file": f"{level}/{rel}",
                    "person_id": it.pid,
                    "camera_id": it.camid + 1,
                    "split": split,
                    "illumination": level,
                    "exposure_scale": float(lg["exposure_scale"]),
                    "noise_sigma": float(lg["noise_sigma"]),
                    "mean_luminance": float(lg["mean_luminance"]),
                    "attributes": None if p is None else {k: p[k] for k in ATTRIBUTES},
                    "caption": None if p is None else p["caption"],
                })
        (out_dir / f"{split}.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {out_dir / (split + '.json')}: {len(entries)} entries, 속성 없는 실제 ID {missing}")


if __name__ == "__main__":
    main()
