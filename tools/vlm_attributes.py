#!/usr/bin/env python3
"""비전 LLM(Grok, GPT)의 조도별 속성 인식 정확도 — CLIP zero-shot / 선형 분류기와 같은 샘플에서 비교.

    python tools/vlm_attributes.py --dry-run                 # 호출 수만 계산 (API 호출 없음)
    python tools/vlm_attributes.py --n 3 --providers xai openai   # 시험 호출로 토큰 측정
    python tools/vlm_attributes.py                           # 전체 (50명 × 3조도 × 모델 수)

- 샘플: 테스트 query에서 서로 다른 사람 N명 (시드 고정), 같은 이미지를 normal/low/extreme으로 질의
- 반 공용 API 키 → vastreid.llm 캐시·호출 상한·사용량 로그 사용. 같은 요청은 다시 호출하지 않음.
- 민감 속성(성별·나이)은 묻지 않고, 판단하지 말라고 명시.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.clip_attr import ATTR_VALUES  # noqa: E402
from vastreid.data import list_split  # noqa: E402
from vastreid.llm import LLM  # noqa: E402

LEVELS = ["normal", "low", "extreme"]
SYSTEM = ("You are a careful visual attribute annotator for pedestrian images from surveillance cameras. "
          "Describe only clothing and carried items. Never infer or mention gender, age, race, or identity. "
          "Answer with a single JSON object only.")
USER = (
    "Look at this pedestrian image (it may be very dark or noisy). Choose exactly one value per field.\n"
    '- "upper_color": one of black, white, red, purple, yellow, gray, blue, green, unknown\n'
    '- "lower_color": one of black, white, pink, purple, yellow, gray, blue, green, brown, unknown\n'
    '- "sleeve": one of long, short, unknown\n'
    '- "lower_type": one of pants, shorts, skirt, unknown\n'
    '- "bag": one of backpack, handbag, bag, none, unknown   (bag = shoulder/other bag)\n'
    '- "hat": one of true, false, unknown\n'
    'Use "unknown" only if it is truly impossible to tell. Return JSON with exactly these six keys.'
)


def sample_items(n: int, seed: int = 42):
    table = {int(k): v for k, v in json.loads(
        (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}
    q = list_split(ROOT / "data/vast", "normal", "query")
    first = {}
    for i, it in enumerate(q):
        if it.pid in table and it.pid not in first:
            first[it.pid] = i
    pids = sorted(first)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(pids, size=min(n, len(pids)), replace=False).tolist())
    return [(first[p], q[first[p]]) for p in chosen], table


def parse(text: str | None) -> dict:
    if not text:
        return {}
    try:
        d = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return {}
    out = {}
    for k, vals in ATTR_VALUES.items():
        v = str(d.get(k, "unknown")).strip().lower()
        if k == "hat":
            v = {"true": "True", "false": "False"}.get(v, "unknown")
        out[k] = v if v in vals else "unknown"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--providers", nargs="+", default=["xai", "openai"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=160, help="provider당 실제 호출 상한")
    ap.add_argument("--out", type=Path, default=ROOT / "results/attr_acc_vlm.csv")
    args = ap.parse_args()

    items, table = sample_items(args.n)
    print(f"[info] {len(items)} persons × {len(LEVELS)} levels = {len(items) * len(LEVELS)} images per provider")

    preds = {}  # (model, level) -> list[dict]
    for prov in args.providers:
        llm = LLM(provider=prov, dry_run=args.dry_run, max_calls=args.max_calls)
        model = llm.pcfg["vision_model"]
        for lv in LEVELS:
            res = []
            for _, it in items:
                path = str(it.path).replace("/normal/", f"/{lv}/")
                res.append(parse(llm.chat(SYSTEM, USER, images=[path], json_mode=True, tag=f"vlm_attr/{lv}")))
            preds[(f"{prov}:{model}", lv)] = res
        print(llm.summary())
    if args.dry_run:
        return

    # 같은 샘플에서 CLIP zero-shot / 선형 분류기 예측
    for name, prefix in [("clip_zero_shot", "clip_ViT-B-16"), ("clip_linear_probe", "probe")]:
        for lv in LEVELS:
            d = np.load(ROOT / f"outputs/attr_probs/{prefix}__{lv}__query.npz")
            preds[(name, lv)] = [{k: vals[int(d[k][i].argmax())] for k, vals in ATTR_VALUES.items()}
                                 for i, _ in items]

    rows = []
    for (model, lv), res in preds.items():
        accs, unk = {}, 0
        for k in ATTR_VALUES:
            c = n = 0
            for (_, it), r in zip(items, res):
                gt = str(table[it.pid][k])
                if gt == "unknown":
                    continue
                n += 1
                c += int(r.get(k) == gt)  # unknown/파싱 실패는 오답 처리
                unk += int(r.get(k, "unknown") == "unknown")
            accs[k] = c / max(n, 1)
        rows.append({"model": model, "level": lv, "n_images": len(res), **{k: round(v, 4) for k, v in accs.items()},
                     "mean_acc": round(float(np.mean(list(accs.values()))), 4), "unknown_answers": unk})
    rows.sort(key=lambda r: (r["model"], LEVELS.index(r["level"])))
    for r in rows:
        print(f"  {r['model']:32s} {r['level']:8s} mean {r['mean_acc']*100:5.1f}  "
              + "  ".join(f"{k} {r[k]*100:.0f}" for k in ATTR_VALUES) + f"  unk {r['unknown_answers']}")
    out = args.out if args.n >= 50 else args.out.with_name(args.out.stem + "_pilot.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
