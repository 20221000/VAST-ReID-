#!/usr/bin/env python3
"""VAST-ReID 웹 데모 서버 (디자인 시안 3화면 구현).

    python demo/server.py          # http://127.0.0.1:7860

화면: demo/web/ (HTML/CSS/JS), 로직: vastreid/demo_core.py, 수치: results/*.csv
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid import demo_core as core  # noqa: E402

app = FastAPI(title="VAST-ReID Demo")
WEB = Path(__file__).resolve().parent / "web"
NAME = re.compile(r"^-?\d{1,4}_c\ds\d_\d{6}_\d{2}\.jpg$")
ATTR_KEYS = list(core.ATTR_VALUES)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/img/{level}/{split}/{name}")
def image(level: str, split: str, name: str):
    if level not in core.LEVELS or split not in ("query", "gallery") or not NAME.match(name):
        raise HTTPException(404)
    p = core.DATA / level / split / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/img_night/{split}/{stem}")
def image_night(split: str, stem: str):
    if split not in ("query", "gallery"):
        raise HTTPException(404)
    try:
        p = core.night_path(split, stem)  # 이름 형식·목록에 있는 파일만 허용
    except ValueError:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/img_msmt/{level}/{split}/{stem}")
def image_msmt(level: str, split: str, stem: str):
    if split not in ("query", "gallery"):
        raise HTTPException(404)
    try:
        p = core.msmt_path(level, split, stem)  # 이름 형식·목록에 있는 파일만 허용
    except ValueError:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def _level(v: str) -> str:
    if v not in core.LEVELS:
        raise HTTPException(400, "level")
    return v


@app.get("/api/samples")
def samples(level: str = "low", seed: int | None = Query(None, ge=0, le=2**31)):
    return core.samples(_level(level), seed)


@app.get("/api/search")
def search(q_level: str = "low", sample: int | None = None, g_level: str = "normal", adabn: bool = True,
           csls: bool = True, kr: bool = False, beta: float = 1.0, lam: float = 0.5, src_level: str | None = None,
           src_gallery: int | None = None, upper_color: str = "", lower_color: str = "", sleeve: str = "",
           lower_type: str = "", bag: str = "", hat: str = "", backbone: str = core.DEFAULT_BACKBONE,
           ds: str = "market", qidx: int | None = None,
           attr_strength: str = "auto"):
    if ds not in ("market", "msmt", "night"):
        raise HTTPException(400, "ds")
    if sample is None and src_gallery is None and qidx is None:
        raise HTTPException(400, "sample, qidx or src_gallery")
    if sample is not None and not 0 <= sample < len(core.sample_idx()):
        raise HTTPException(400, "sample")
    if qidx is not None:
        n = {"msmt": lambda: len(core.msmt_items("query")), "night": lambda: len(core.night_items("query"))}.get(
            ds, lambda: len(core.items(_level(q_level), "query")))()
        if not 0 <= qidx < n or (ds == "msmt" and not core.msmt_ready()) or (ds == "night" and not core.night_ready()):
            raise HTTPException(400, "qidx")
    attrs = {"upper_color": upper_color, "lower_color": lower_color, "sleeve": sleeve,
             "lower_type": lower_type, "bag": bag, "hat": hat}
    return core.search(_level(q_level), sample, _level(g_level), adabn, csls, beta, lam, attrs,
                       _level(src_level) if src_level else None, src_gallery, kr, backbone if backbone in core.BACKBONES else core.DEFAULT_BACKBONE,
                       attr_strength if attr_strength in ("auto", "strong") else "auto", ds, qidx)


@app.get("/api/text")
def text(q: str = Query("", max_length=200), g_level: str = "normal", llm: bool = False,
         upper_color: str = "", lower_color: str = "", sleeve: str = "", lower_type: str = "", bag: str = "",
         hat: str = ""):
    extra = {"upper_color": upper_color, "lower_color": lower_color, "sleeve": sleeve, "lower_type": lower_type,
             "bag": bag, "hat": hat}
    return core.text_search(q, _level(g_level), llm, extra)


@app.get("/api/attr_values")
def attr_values():
    return {"values": core.ATTR_VALUES, "attr_ko": core.ATTR_KO, "value_ko": core.VALUE_KO}


@app.get("/api/bench")
def bench():
    return core.bench()


@app.get("/api/ai_caption")
def ai_caption(level: str, split: str, name: str):
    try:
        return core.ai_caption(level, split, name)
    except ValueError:
        raise HTTPException(400, "image")
    except Exception as e:  # 키 없음, 호출 상한 등
        return {"text": "", "error": f"{type(e).__name__}: {str(e)[:120]}"}


@app.get("/api/ai_explain")
def ai_explain(q_level: str, q_split: str, q_name: str, g_level: str, g_name: str,
               causes: str = Query("", max_length=400)):
    try:
        return core.ai_explain(q_level, q_split, q_name, g_level, g_name, [c for c in causes.split("|") if c])
    except ValueError:
        raise HTTPException(400, "image")
    except Exception as e:
        return {"text": "", "error": f"{type(e).__name__}: {str(e)[:120]}"}


@app.get("/api/track_frame")
def track_frame(level: str = "extreme", i: int = 1, tracker: str = "botsort-osnet-adabn"):
    if tracker not in ("bytetrack", "botsort-osnet", "botsort-osnet-adabn"):
        raise HTTPException(400, "tracker")
    return Response(core.track_frame(_level(level), i, tracker), media_type="image/jpeg")


app.mount("/web", StaticFiles(directory=WEB), name="web")
# 영상에서 찾기(브라우저 계산): ONNX 모델과 예제 영상을 공개판과 같은 경로로 제공
for _d in ("models", "video"):
    if (ROOT / "demo/static" / _d).is_dir():
        app.mount(f"/{_d}", StaticFiles(directory=ROOT / "demo/static" / _d), name=_d)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 7860)))
