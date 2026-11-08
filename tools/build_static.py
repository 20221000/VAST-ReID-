#!/usr/bin/env python3
"""GitHub Pages 공개판 데이터 생성: 데모 API 응답을 미리 계산해 정적 JSON·이미지로 저장.

    python tools/build_static.py                 # → demo/static/ (git에 포함, 약 수십 MB)
    python tools/build_static.py --site _site    # + 화면(demo/web)과 합쳐 로컬 확인용 사이트 조립
    python -m http.server -d _site 8000          # http://localhost:8000

GitHub에서는 .github/workflows/pages.yml이 main push마다 demo/web + demo/static을 합쳐 Pages에 배포한다.
화면(HTML·CSS·JS)만 고치면 push만으로 바로 반영되고, 검색 결과·벤치마크 수치를 바꿨을 때만 이 스크립트를 다시 실행해 커밋한다.

공개 범위 (사용자 결정, 11/8):
- 사진: Market-1501 기본 샘플 24명과 Night600 query 8명, 그 검색 결과에 나온 사진만. **MSMT17은 재배포 금지라 제외.**
- 키·모델 가중치 없음. AI 설명·캡션, LLM 속성 추출, 목격자 진술 필터, 자유 문장 텍스트 검색은 로컬 데모 전용
  (텍스트 검색은 예시 문장만 미리 계산).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid import demo_core as core  # noqa: E402

OUT = ROOT / "demo/static"
N_NIGHT = 8
MODELS = {"ck": (True, True), "c": (True, False), "k": (False, True), "p": (False, False)}  # 거리 보정: csls, kr
TEXT_EXAMPLES = ["흰색 반팔 상의, 검은 반바지, 백팩을 멘 사람", "검은 긴팔 상의에 청바지, 가방 없음",
                 "빨간 상의와 검은 긴 바지", "회색 상의, 핸드백을 든 사람", "모자를 쓰고 흰 상의를 입은 사람"]
TRACK_STEP = 8
TRACK_LAST = 520


class Images:
    """응답 JSON의 이미지 URL을 정적 경로로 바꾸고 해당 파일을 복사."""

    def __init__(self, out: Path):
        self.out, self.n = out, 0

    def __call__(self, url: str) -> str:
        if url.startswith("/img_night/"):
            _, _, split, stem = url.split("/")
            rel, src = f"img_night/{split}/{stem}.jpg", core.night_path(split, stem)
        elif url.startswith("/img/"):
            _, _, level, split, name = url.split("/")
            rel, src = f"img/{level}/{split}/{name}", core.DATA / level / split / name
        else:
            raise ValueError(url)
        dst = self.out / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            self.n += 1
        return rel

    def fix(self, obj):
        if isinstance(obj, dict):
            return {k: (self(v) if k in ("url", "original") and isinstance(v, str) and v.startswith("/img") else self.fix(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.fix(v) for v in obj]
        return obj


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=None, help="화면과 합친 확인용 사이트 폴더 (예: _site)")
    ap.add_argument("--skip-data", action="store_true", help="데이터는 그대로 두고 사이트만 조립")
    args = ap.parse_args()
    if not args.skip_data:
        build_data()
    if args.site:
        assemble(args.site)


def build_data():
    t0 = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    img = Images(OUT)
    api = OUT / "api"
    dump(api / "attr_values.json", {"values": core.ATTR_VALUES, "attr_ko": core.ATTR_KO, "value_ko": core.VALUE_KO})
    dump(api / "bench.json", core.bench())

    nq = core.night_items("query")
    night_idx = sorted(random.Random(0).sample(range(len(nq)), N_NIGHT))
    market_idx = core.sample_idx()
    for lv in core.LEVELS:
        q = core.items(lv, "query")
        lst = [{"ds": "market", "idx": i, "name": q[i].path.stem, "url": core.img_url(lv, "query", q[i].path.name)} for i in market_idx]
        night = [{"ds": "night", "idx": i, "name": nq[i].path.stem, "url": core.night_url("query", nq[i].path.name)} for i in night_idx]
        for k, x in enumerate(night):  # 4명마다 실제 야간 1명씩 섞어 보여줌
            lst.insert(min(len(lst), k * 4 + 3), x)
        dump(api / f"samples_{lv}.json", img.fix(lst))

    jobs = [("market", i, lv) for lv in core.LEVELS for i in market_idx] + [("night", i, "x") for i in night_idx]
    n = 0
    for ds, i, lv in jobs:
        for bb in core.BACKBONES:
            for m, (csls, kr) in MODELS.items():
                for adabn in (0, 1):
                    r = core.search(lv if ds == "market" else "normal", None, "normal", bool(adabn), csls, 1.0, 0.5, {},
                                    None, None, kr, bb, "auto", ds, i)
                    dump(api / "search" / f"{ds}_{i}_{lv}_{bb}_{m}_{adabn}.json", img.fix(r))
                    n += 1
        print(f"  search {ds} {i} {lv}: 누적 {n}개 응답, 이미지 {img.n}장  ({time.time() - t0:.0f}s)", flush=True)

    for k, text in enumerate(TEXT_EXAMPLES):
        for lv in core.LEVELS:
            dump(api / "text" / f"{k}_{lv}.json", img.fix(core.text_search(text, lv)))
    dump(api / "text_examples.json", TEXT_EXAMPLES)

    for lv in core.LEVELS:  # 트래킹 프레임 (MOT17, CC BY-NC-SA) — 폭 640으로 줄여 저장
        for i in range(1, TRACK_LAST + 1, TRACK_STEP):
            buf = core.track_frame(lv, i)
            im = cv2.imdecode(__import__("numpy").frombuffer(buf, "uint8"), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (640, int(im.shape[0] * 640 / im.shape[1])), interpolation=cv2.INTER_AREA)
            p = OUT / "track" / lv / f"{i}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 72])
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2 ** 20
    print(f"[static] {OUT}: 검색 응답 {n}개, 사진 {img.n}장, 합계 {size:.1f}MB  ({time.time() - t0:.0f}s)")


def assemble(site: Path):
    """.github/workflows/pages.yml과 같은 조립: index.html(정적 모드 표시) + web/ + static 데이터."""
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(OUT, site)
    shutil.copytree(ROOT / "demo/web", site / "web")
    html = (ROOT / "demo/web/index.html").read_text(encoding="utf-8").replace('<html lang="ko">', '<html lang="ko" data-static="1">', 1)
    (site / "index.html").write_text(html, encoding="utf-8")
    (site / ".nojekyll").write_text("", encoding="utf-8")
    print(f"[site] {site} 조립 완료")


if __name__ == "__main__":
    main()
