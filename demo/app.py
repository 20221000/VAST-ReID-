#!/usr/bin/env python3
"""VAST-ReID Gradio 데모 (3탭: 이미지 검색 / 텍스트 검색 / 벤치마크).

    python demo/app.py                 # 로컬 (http://127.0.0.1:7860)

필요 파일 (로컬에서 앞 단계 스크립트로 생성):
  data/vast/, outputs/features/osnet_x1_0__none__*.pt, outputs/attr_probs/probe__*.npz,
  annotations/person_attributes.json, results/*.csv
수치는 모두 results/*.csv 에서 읽음 (하드코딩 없음).
"""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.clip_attr import ATTR_VALUES  # noqa: E402
from vastreid.data import list_split  # noqa: E402

DATA = Path(os.environ.get("VAST_DATA", ROOT / "data/vast"))
LEVELS = ["normal", "low", "extreme"]
LEVEL_KO = {"normal": "Normal (주간)", "low": "Low (저조도)", "extreme": "Extreme (극저조도)"}
ATTR_KO = {"upper_color": "상의 색", "lower_color": "하의 색", "sleeve": "소매", "lower_type": "하의 종류",
           "bag": "가방", "hat": "모자"}
VALUE_KO = {"black": "검정", "white": "흰색", "red": "빨강", "purple": "보라", "yellow": "노랑", "gray": "회색",
            "blue": "파랑", "green": "초록", "pink": "분홍", "brown": "갈색", "long": "긴팔", "short": "반팔",
            "pants": "긴 바지", "shorts": "반바지", "skirt": "치마", "backpack": "백팩", "handbag": "핸드백",
            "bag": "가방", "none": "없음", "True": "씀", "False": "안 씀"}
ANY = "모름"
N_SAMPLES = 24
PERSONS = {int(k): v for k, v in json.loads(
    (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}


# ------------------------------------------------------------------ 인덱스
@lru_cache(maxsize=None)
def index(level: str, split: str):
    items = list_split(DATA, level, split)
    d = torch.load(ROOT / f"outputs/features/osnet_x1_0__none__{level}__{split}.pt", weights_only=True)
    feats = torch.nn.functional.normalize(d["feats"].float(), dim=1).numpy()
    probs = None
    pp = ROOT / f"outputs/attr_probs/probe__{level}__{split}.npz"
    if pp.exists():
        z = np.load(pp)
        probs = {k: z[k] for k in ATTR_VALUES}
    return items, feats, probs


@lru_cache(maxsize=1)
def reid_model():
    from vastreid.reid import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model("osnet_x1_0", ROOT / "weights/osnet_x1_0_market1501.pth", dev), dev


def sample_queries():
    items, _, _ = index("normal", "query")
    seen, out = set(), []
    for i in range(0, len(items), 37):
        if items[i].pid not in seen and items[i].pid in PERSONS:
            seen.add(items[i].pid)
            out.append(i)
        if len(out) >= N_SAMPLES:
            break
    return out


SAMPLE_IDX = sample_queries()


def witness_vec(choices: dict[str, str]):
    """사용자가 고른 속성 → (속성별 one-hot, 선택 여부)."""
    rep = {}
    for k, vals in ATTR_VALUES.items():
        v = choices.get(k, ANY)
        if v == ANY or v is None:
            continue
        rep[k] = np.eye(len(vals), dtype=np.float32)[vals.index(v)]
    return rep


def agreement(rep: dict, gprobs: dict) -> np.ndarray:
    if not rep or gprobs is None:
        return None
    return np.mean([gprobs[k] @ v for k, v in rep.items()], axis=0)


def caption_of(pid: int) -> str:
    p = PERSONS.get(pid)
    return p["caption"] if p else "속성 정보 없음 (distractor)"


# ------------------------------------------------------------------ 탭 1: 이미지 검색
def show_samples(level):
    items, _, _ = index(level, "query")
    return [(str(items[i].path), f"#{n}") for n, i in enumerate(SAMPLE_IDX)]


def image_search(sel, upload, q_level, g_level, lam, *attr_choices):
    gi, gf, gp = index(g_level, "gallery")
    if upload is not None:
        from vastreid.reid import extract
        import cv2
        tmp = ROOT / "outputs/demo_upload.jpg"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp), cv2.cvtColor(np.asarray(upload), cv2.COLOR_RGB2BGR))
        model, dev = reid_model()
        f, _ = extract(model, [tmp], dev, batch=1, workers=0)
        qf = torch.nn.functional.normalize(f, dim=1).cpu().numpy()[0]
        q_pid, q_cam, q_path = None, None, str(tmp)
    else:
        if sel is None:
            return None, [], "왼쪽에서 query 샘플을 고르거나 이미지를 업로드하세요."
        qi, qfs, _ = index(q_level, "query")
        idx = SAMPLE_IDX[sel]
        qf, q_pid, q_cam, q_path = qfs[idx], qi[idx].pid, qi[idx].camid, str(qi[idx].path)

    score = gf @ qf
    rep = witness_vec(dict(zip(ATTR_VALUES, attr_choices)))
    agr = agreement(rep, gp)
    if agr is not None:
        score = score + lam * agr
    if q_pid is not None:  # 표준 프로토콜: 같은 사람·같은 카메라 제외
        same = np.array([(g.pid == q_pid and g.camid == q_cam) for g in gi])
        score = np.where(same, -np.inf, score)
    order = np.argsort(-score)
    top = []
    for r, j in enumerate(order[:10], 1):
        g = gi[j]
        badge = "" if q_pid is None else (" ✅ 정답" if g.pid == q_pid else " ❌")
        top.append((str(g.path), f"#{r} · 점수 {score[j]:.3f} · cam{g.camid + 1}{badge}"))

    msg = [f"**gallery**: {LEVEL_KO[g_level]} {len(gi):,}장"]
    if rep:
        msg.append("**목격자 속성 재순위화**: " + ", ".join(f"{ATTR_KO[k]}={VALUE_KO.get(v, v)}"
                                                for k, v in zip(ATTR_VALUES, attr_choices) if v != ANY)
                   + f" (λ={lam})")
    if q_pid is not None:
        valid = [k for k in order if np.isfinite(score[k])]
        first = next((r for r, k in enumerate(valid, 1) if gi[k].pid == q_pid), None)
        msg.append(f"**정답 인물 첫 등장 순위**: {first if first else '없음'}")
        msg.append(f"**query 인물 (참고용 정답 속성)**: {caption_of(q_pid)}")
    return q_path, top, "  \n".join(msg)


# ------------------------------------------------------------------ 탭 2: 텍스트 검색
KEYWORDS = {
    "upper_color": {"black": ["검은 상의", "검정 상의", "검은색 상의", "black shirt", "black top"],
                    "white": ["흰 상의", "흰색 상의", "하얀 상의", "white shirt", "white top"],
                    "red": ["빨간 상의", "red shirt"], "purple": ["보라 상의", "purple shirt"],
                    "yellow": ["노란 상의", "yellow shirt"], "gray": ["회색 상의", "gray shirt", "grey shirt"],
                    "blue": ["파란 상의", "blue shirt"], "green": ["초록 상의", "green shirt"]},
    "lower_color": {"black": ["검은 바지", "검정 바지", "검은 하의", "검은 치마", "black pants"],
                    "white": ["흰 바지", "흰 하의", "흰 치마", "white pants"],
                    "blue": ["파란 바지", "청바지", "blue pants", "jeans"],
                    "gray": ["회색 바지", "gray pants"], "brown": ["갈색 바지", "brown pants"],
                    "pink": ["분홍 치마", "분홍 바지", "pink"], "green": ["초록 바지", "green pants"]},
    "sleeve": {"long": ["긴팔", "long sleeve", "long-sleeve"], "short": ["반팔", "short sleeve", "t-shirt"]},
    "lower_type": {"pants": ["긴 바지", "청바지", "pants", "jeans", "trousers"], "shorts": ["반바지", "shorts"],
                   "skirt": ["치마", "원피스", "skirt", "dress"]},
    "bag": {"backpack": ["백팩", "배낭", "backpack"], "handbag": ["핸드백", "handbag"],
            "bag": ["가방", "숄더백", "shoulder bag"], "none": ["가방 없", "no bag"]},
    "hat": {"True": ["모자를 쓴", "모자 쓴", "모자", "hat", "cap"], "False": ["모자 안", "no hat"]},
}


def parse_rules(text: str) -> dict:
    t = text.lower()
    out = {}
    for attr, table in KEYWORDS.items():
        for v, kws in table.items():
            if any(k in t for k in kws):
                out[attr] = v
                break
    # "흰색 상의, 검은 바지"처럼 '색 + 상의/바지'가 떨어져 있어도 잡기
    colors = {"검": "black", "흰": "white", "하얀": "white", "빨": "red", "보라": "purple", "노란": "yellow",
              "회색": "gray", "파란": "blue", "초록": "green", "분홍": "pink", "갈색": "brown"}
    pat = (r"(검|흰|하얀|빨|보라|노란|회색|파란|초록|분홍|갈색)\S*\s+(?:(?:긴팔|반팔|긴|짧은)\s*)?"
           r"(상의|티셔츠|티|셔츠|반바지|바지|하의|치마)")
    for m in re.finditer(pat, text):
        attr = "upper_color" if m.group(2) in ("상의", "티셔츠", "티", "셔츠") else "lower_color"
        c = colors[m.group(1)]
        if c in ATTR_VALUES[attr]:
            out.setdefault(attr, c)
    return out


def parse_llm(text: str) -> dict:
    from vastreid.llm import LLM
    llm = LLM(max_calls=5)
    sys_p = ("Extract clothing and carried-item attributes of a pedestrian from a description. "
             "Never infer gender or age. Answer with one JSON object only.")
    user = ("Description: " + text + "\nReturn JSON with any of these keys you can determine "
            "(omit unknown keys): " + json.dumps({k: v for k, v in ATTR_VALUES.items()}))
    raw = llm.chat(sys_p, user, json_mode=True, tag="demo_text_search") or "{}"
    try:
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        return {}
    out = {}
    for k, vals in ATTR_VALUES.items():
        v = str(d.get(k, "")).strip()
        v = {"true": "True", "false": "False"}.get(v.lower(), v.lower() if k != "hat" else v)
        if v in vals:
            out[k] = v
    return out


def text_search(text, g_level, use_llm):
    if not text.strip():
        return "설명을 입력하세요.", []
    how = "규칙 기반"
    attrs = {}
    if use_llm:
        try:
            attrs, how = parse_llm(text), "LLM"
        except Exception as e:  # 키 없음 등 → 규칙 기반으로 대체
            how = f"규칙 기반 (LLM 사용 불가: {type(e).__name__})"
    if not attrs:
        attrs = parse_rules(text)
    if not attrs:
        return f"인식된 속성이 없습니다 ({how}). 예: 흰색 상의, 검은 바지, 백팩", []
    gi, _, gp = index(g_level, "gallery")
    rep = {k: np.eye(len(ATTR_VALUES[k]), dtype=np.float32)[ATTR_VALUES[k].index(v)] for k, v in attrs.items()}
    score = agreement(rep, gp)
    order = np.argsort(-score)
    seen, res = set(), []
    for j in order:
        g = gi[j]
        if g.pid in seen or g.pid <= 0:
            continue
        seen.add(g.pid)
        res.append((str(g.path), f"{score[j]:.2f} · {caption_of(g.pid)}"))
        if len(res) >= 20:
            break
    chips = " ".join(f"`{ATTR_KO[k]}: {VALUE_KO.get(v, v)}`" for k, v in attrs.items())
    return f"**추출된 속성** ({how}): {chips}  \n결과는 사람마다 대표 이미지 1장, 설명은 참고용 정답 속성.", res


# ------------------------------------------------------------------ 탭 3: 벤치마크
def read(name):
    p = ROOT / "results" / name
    return pd.read_csv(p) if p.exists() else None


def bench_rerank(metric):
    col = {"Rank-1": "rank1", "mAP": "mAP"}[metric]
    rows = []
    names = {"oracle": "정답 속성 (상한선)", "clip": "CLIP zero-shot 속성", "probe": "선형 분류기 속성",
             "witness-probe": "목격자 + 선형 분류기"}
    base_done = set()
    for src, label in names.items():
        df = read(f"rerank_{src}.csv")
        if df is None:
            continue
        for _, r in df.iterrows():
            cond = f"{r.query_level}→{r.gallery_level}"
            if r.method == "cosine" and cond not in base_done:
                rows.append({"조건": cond, "방법": "외형만 (OSNet)", metric: r[col] * 100})
                base_done.add(cond)
            if r.method == "rerank":
                rows.append({"조건": cond, "방법": label, metric: r[col] * 100})
    return pd.DataFrame(rows)


def bench_enhance(metric):
    col = {"Rank-1": "rank1", "mAP": "mAP"}[metric]
    rows = []
    for name, tag in [("enhance.csv", "전부 적용"), ("enhance_v2.csv", "게이팅")]:
        df = read(name)
        if df is None:
            continue
        for _, r in df.iterrows():
            if r.protocol != "same":
                continue
            den = r.get("denoise", "none")
            label = "개선 없음" if r.enhance == "none" and den in ("none", np.nan) else \
                f"{tag}: {'NLM+' if den == 'nlm' else ''}{r.enhance}"
            rows.append({"조도": r.query_level, "방법": label, metric: r[col] * 100})
    return pd.DataFrame(rows).drop_duplicates(["조도", "방법"])


def bench_attr():
    rows = []
    for name in ["attr_acc_vlm.csv"]:
        df = read(name)
        if df is None:
            continue
        for _, r in df.iterrows():
            rows.append({"조도": r.level, "모델": r.model.split(":")[-1], "평균 정확도": r.mean_acc * 100})
    return pd.DataFrame(rows)


def bench_cards():
    b = read("baseline.csv")
    p = read("rerank_witness-probe.csv")
    if b is None:
        return "results/baseline.csv 가 없습니다."
    get = lambda df, **kw: df.loc[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
    n = get(b, protocol="same", query_level="normal")
    lo = get(b, protocol="same", query_level="low")
    cr = get(b, protocol="cross", query_level="low")
    s = (f"### 핵심 지표 (results/*.csv)\n"
         f"- OSNet 베이스라인(유클리드 거리): Normal Rank-1 **{n.rank1*100:.1f}%** → Low **{lo.rank1*100:.1f}%** "
         f"(하락 {(n.rank1-lo.rank1)*100:.1f}%p), 야간 query → 주간 gallery **{cr.rank1*100:.1f}%**\n")
    if p is not None:
        c0 = get(p, protocol="cross", query_level="low", method="cosine")
        c1 = get(p, protocol="cross", query_level="low", method="rerank")
        s += (f"- 목격자 속성 + 선형 분류기 재순위화(코사인 기준): 야간→주간 Rank-1 "
              f"**{c0.rank1*100:.1f}% → {c1.rank1*100:.1f}%** (+{(c1.rank1-c0.rank1)*100:.1f}%p)\n")
    s += "- 저조도 개선(감마·CLAHE·Zero-DCE)은 ReID 성능을 높이지 못함 (아래 그래프)"
    return s


# ------------------------------------------------------------------ UI
CSS = """
:root, .gradio-container { --vast-navy: #1B2A4A; --vast-amber: #F5B547; }
.gradio-container { font-family: 'IBM Plex Sans KR', sans-serif !important; }
#title h1 { color: var(--vast-navy); margin-bottom: 0; }
#title p { color: #5b6475; }
.primary-btn { background: var(--vast-navy) !important; color: white !important; }
.gallery-item, .thumbnail-item { background: #0d1117 !important; }
"""
HEAD = ('<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&display=swap" '
        'rel="stylesheet">')


def build():
    with gr.Blocks(title="VAST-ReID Demo") as demo:
        gr.Markdown("# VAST-ReID\n저조도 사람 재식별(Re-ID) 벤치마크 데모 · Market-1501 기반 저조도 합성 데이터",
                    elem_id="title")
        with gr.Tabs():
            with gr.Tab("이미지 검색"):
                with gr.Row():
                    with gr.Column(scale=2):
                        q_level = gr.Radio(LEVELS, value="low", label="query 조도")
                        samples = gr.Gallery(show_samples("low"), label="query 샘플 (클릭해서 선택)",
                                             columns=8, height=300, allow_preview=False)
                        sel = gr.State(None)
                        upload = gr.Image(label="또는 이미지 업로드 (선택)", type="numpy", height=200)
                        g_level = gr.Radio(LEVELS, value="normal", label="gallery 조도 (기록 영상)")
                        with gr.Accordion("목격자 속성 (재순위화)", open=True):
                            attr_inputs = [gr.Dropdown([ANY] + ATTR_VALUES[k], value=ANY, label=ATTR_KO[k])
                                           for k in ATTR_VALUES]
                            lam = gr.Slider(0, 1, value=0.5, step=0.05, label="λ (속성 가중치)")
                        btn = gr.Button("검색", elem_classes="primary-btn")
                    with gr.Column(scale=3):
                        with gr.Row():
                            q_view = gr.Image(label="query", height=256, interactive=False)
                            info = gr.Markdown()
                        results = gr.Gallery(label="Top-10 결과", columns=5, height=560)
                q_level.change(show_samples, q_level, samples)
                def on_select(evt: gr.SelectData):
                    return evt.index
                samples.select(on_select, None, sel)
                btn.click(image_search, [sel, upload, q_level, g_level, lam, *attr_inputs], [q_view, results, info])

            with gr.Tab("텍스트 검색"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text = gr.Textbox(label="인상착의", value="흰색 반팔 상의, 검은 반바지, 백팩을 멘 사람")
                        tg_level = gr.Radio(LEVELS, value="normal", label="gallery 조도")
                        use_llm = gr.Checkbox(False, label="LLM으로 속성 추출 (로컬 전용, API 사용)")
                        tbtn = gr.Button("검색", elem_classes="primary-btn")
                        tinfo = gr.Markdown()
                    with gr.Column(scale=3):
                        tres = gr.Gallery(label="속성 일치 순 결과 (사람별 1장)", columns=5, height=600)
                tbtn.click(text_search, [text, tg_level, use_llm], [tinfo, tres])

            with gr.Tab("벤치마크"):
                gr.Markdown(bench_cards())
                metric = gr.Radio(["Rank-1", "mAP"], value="Rank-1", label="지표")
                p1 = gr.BarPlot(bench_rerank("Rank-1"), x="조건", y="Rank-1", color="방법",
                                title="속성 재순위화 (OSNet + 속성)", height=360)
                p2 = gr.BarPlot(bench_enhance("Rank-1"), x="방법", y="Rank-1", color="조도",
                                title="저조도 개선 전처리 (same 프로토콜)", height=360, x_label_angle=-30)
                p3 = gr.BarPlot(bench_attr(), x="모델", y="평균 정확도", color="조도",
                                title="조도별 속성 인식 정확도 (50명 샘플)", height=360)

                def upd(m):
                    return (gr.BarPlot(bench_rerank(m), y=m), gr.BarPlot(bench_enhance(m), y=m))
                metric.change(upd, metric, [p1, p2])
                _ = p3
    return demo


if __name__ == "__main__":
    build().launch(css=CSS, head=HEAD, theme=gr.themes.Soft(primary_hue="blue"),
                   server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
