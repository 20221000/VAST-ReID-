#!/usr/bin/env python3
"""GitHub Pages 공개판 데이터 생성: 데모 API 응답을 미리 계산해 정적 JSON·이미지로 저장.

    python tools/build_static.py                 # → demo/static/ (git에 포함, 약 수십 MB)
    python tools/build_static.py --site _site    # + 화면(demo/web)과 합쳐 로컬 확인용 사이트 조립
    python -m http.server -d _site 8000          # http://localhost:8000

GitHub에서는 .github/workflows/pages.yml이 main push마다 demo/web + demo/static을 합쳐 Pages에 배포한다.
화면(HTML·CSS·JS)만 고치면 push만으로 바로 반영되고, 검색 결과·벤치마크 수치를 바꿨을 때만 이 스크립트를 다시 실행해 커밋한다.

공개 범위 (사용자 결정, 11/8):
- 사진: Market-1501 기본 샘플 24명과 Night600 query 8명, 그 검색 결과에 나온 사진만. **MSMT17은 재배포 금지라 제외.**
- 키·모델 가중치 없음. LLM 속성 추출, 목격자 진술 필터, 자유 문장 텍스트 검색은 로컬 데모 전용
  (텍스트 검색은 예시 문장만 미리 계산).
- `--ai`: 오답 카드의 AI 오답 설명과 텍스트 결과의 AI 캡션을 **로컬에서 미리 생성**해 JSON에 텍스트로만 넣음
  (키는 .env에만, 공개 사이트는 API를 부르지 않음). 응답은 outputs/api_cache에 캐시되어 다시 실행하면 무료.
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
    ap.add_argument("--ai", action="store_true", help="AI 오답 설명·캡션을 미리 생성해 넣음 (.env의 API 키 필요, 캐시 재사용)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not args.skip_data:
        build_data()
        if args.ai:
            build_ai(args.workers)
        check_no_secrets()
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
                for adabn in ((0, 1) if bb == "base" else (0,)):  # 저조도 보정은 사전학습 OSNet에서만 켤 수 있음
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


def build_ai(workers: int):
    """검색 JSON의 오답마다 x["ai"], 텍스트 결과마다 x["ai_caption"]을 채움. 같은 (쿼리, 오답) 쌍은 한 번만 호출."""
    from concurrent.futures import ThreadPoolExecutor
    core.AI_MAX_CALLS = 10000  # 미리 생성용 (서버 실행 상한 120과 별개)
    t0 = time.time()
    files = sorted((OUT / "api/search").glob("*.json")) + sorted((OUT / "api/text").glob("*.json"))
    docs = {f: json.loads(f.read_text(encoding="utf-8")) for f in files}
    explain, caption = {}, {}
    for f, r in docs.items():
        if "query" in r:
            q = r["query"]
            for x in r["results"]:
                if not x["ok"]:
                    explain[(q["lvl"], q["split"], q["name"], x["glvl"], x["name"], tuple(x["causes"]))] = None
        else:
            for x in r["results"]:
                caption[(r["g_level"], x["name"])] = None

    def run(job):
        kind, key = job
        try:
            out = core.ai_explain(*key[:5], list(key[5])) if kind == "e" else core.ai_caption(key[0], "gallery", key[1])
            return job, out["text"], out["model"]
        except Exception as e:  # 한 건 실패는 건너뜀 (화면에는 안내 문구)
            return job, None, f"{type(e).__name__}"
    jobs = [("e", k) for k in explain] + [("c", k) for k in caption]
    print(f"  [ai] 오답 설명 {len(explain)}개 + 캡션 {len(caption)}개 생성 (캐시 재사용)", flush=True)
    fail, model = 0, None
    with ThreadPoolExecutor(workers) as ex:
        for n, ((kind, key), text, m) in enumerate(ex.map(run, jobs), 1):
            (explain if kind == "e" else caption)[key] = text
            fail += text is None
            model = model or (m if text else None)
            if n % 200 == 0:
                print(f"  [ai] {n}/{len(jobs)}  실패 {fail}  ({time.time() - t0:.0f}s)", flush=True)
    for f, r in docs.items():
        for x in r["results"]:
            if "query" in r:
                if not x["ok"]:
                    x["ai"] = explain[(r["query"]["lvl"], r["query"]["split"], r["query"]["name"], x["glvl"], x["name"], tuple(x["causes"]))]
            else:
                x["ai_caption"] = caption[(r["g_level"], x["name"])]
        r["ai_model"] = model
        dump(f, r)
    print(f"  [ai] 완료: 실패 {fail}건, 모델 {model}, {core._llm().summary()}  ({time.time() - t0:.0f}s)")


def check_no_secrets():
    """공개 데이터에 API 키 형태의 문자열이나 .env 값이 섞이지 않았는지 확인 (있으면 중단)."""
    import re
    env = ROOT / ".env"
    vals = [v.split("=", 1)[1].strip().strip("\"'") for v in env.read_text(encoding="utf-8").splitlines()
            if "=" in v and not v.lstrip().startswith("#")] if env.exists() else []
    vals = [v for v in vals if len(v) >= 12]
    pat = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|xai-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{16,})")
    for f in OUT.rglob("*.json"):
        t = f.read_text(encoding="utf-8")
        if pat.search(t) or any(v in t for v in vals):
            raise SystemExit(f"[static] 키로 보이는 문자열 발견: {f} — 공개 데이터에서 제거 필요")
    print("[static] 키 검사 통과 (공개 데이터에 API 키 없음)")


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
