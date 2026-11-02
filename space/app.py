"""VAST-ReID 공개 데모 (Hugging Face Spaces).

- 소형 샘플 갤러리만 포함 (tools/build_space.py로 생성). 특징·속성 확률·hub 점수는 미리 계산되어 있어 numpy만 사용.
- API 키, 모델 가중치, 이미지 업로드 기능 없음. 텍스트 검색은 키워드 규칙 기반.
- 수치는 data/results/*.csv (전체 벤치마크 실제 실행 결과)에서 읽음.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent / "data"
Z = np.load(D / "index.npz")
PERSONS = {int(k): v for k, v in json.loads((D / "persons.json").read_text(encoding="utf-8")).items()}
LEVELS = ["normal", "low", "extreme"]
ATTR_VALUES = {
    "upper_color": ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"],
    "lower_color": ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"],
    "sleeve": ["long", "short"],
    "lower_type": ["pants", "shorts", "skirt"],
    "bag": ["backpack", "handbag", "bag", "none"],
    "hat": ["False", "True"],
}
ATTR_KO = {"upper_color": "상의 색", "lower_color": "하의 색", "sleeve": "소매", "lower_type": "하의 종류",
           "bag": "가방", "hat": "모자"}
VALUE_KO = {"black": "검정", "white": "흰색", "red": "빨강", "purple": "보라", "yellow": "노랑", "gray": "회색",
            "blue": "파랑", "green": "초록", "pink": "분홍", "brown": "갈색", "long": "긴팔", "short": "반팔",
            "pants": "긴 바지", "shorts": "반바지", "skirt": "치마", "backpack": "백팩", "handbag": "핸드백",
            "bag": "가방", "none": "없음", "True": "씀", "False": "안 씀"}
ANY = "모름"
QP, QC, GP, GC = Z["q_pids"], Z["q_cams"], Z["g_pids"], Z["g_cams"]


def img(level, split, name):
    return str(D / "images" / level / split / name)


def caption_of(pid):
    p = PERSONS.get(int(pid))
    return p["caption"] if p else "속성 정보 없음 (distractor)"


@lru_cache(maxsize=None)
def arr(key):
    return Z[key].astype(np.float32)


def gprobs(level):
    return {k: arr(f"gprob__{level}__{k}") for k in ATTR_VALUES}


def agreement(choices: dict, level: str):
    gp = gprobs(level)
    terms = [gp[k] @ np.eye(len(v), dtype=np.float32)[v.index(choices[k])]
             for k, v in ATTR_VALUES.items() if choices.get(k, ANY) not in (ANY, None)]
    return np.mean(terms, axis=0) if terms else None


# ------------------------------------------------------------------ 이미지 검색
def show_samples(level):
    return [(img(level, "query", n), f"#{i}") for i, n in enumerate(Z["q_names"])]


def image_search(sel, q_level, g_level, use_adabn, use_csls, beta, lam, *attrs):
    if sel is None:
        return None, [], "왼쪽에서 query 샘플을 클릭하세요."
    feat = "adabn" if use_adabn else "none"
    qf = arr(f"qf__{q_level}__{feat}")[sel]
    score = arr(f"gf__{g_level}__{feat}") @ qf
    if use_csls:
        score = score - beta * arr(f"hub__{q_level}__{g_level}__{feat}")
    choices = dict(zip(ATTR_VALUES, attrs))
    agr = agreement(choices, g_level)
    if agr is not None:
        score = score + lam * agr
    score = np.where((GP == QP[sel]) & (GC == QC[sel]), -np.inf, score)  # 같은 사람·같은 카메라 제외
    order = np.argsort(-score)
    top = [(img(g_level, "gallery", Z["g_names"][j]),
            f"#{r} · 점수 {score[j]:.3f} · cam{GC[j] + 1} " + ("✅ 정답" if GP[j] == QP[sel] else "❌"))
           for r, j in enumerate(order[:10], 1)]
    valid = [j for j in order if np.isfinite(score[j])]
    first = next((r for r, j in enumerate(valid, 1) if GP[j] == QP[sel]), None)
    msg = [f"**gallery**: {g_level} 샘플 {len(GP):,}장 (공개용 축소판이라 실제 벤치마크보다 쉬움)",
           f"**정답 인물 첫 등장 순위**: {first}",
           f"**query 인물 (참고용 정답 속성)**: {caption_of(QP[sel])}"]
    if use_adabn:
        msg.insert(1, "**AdaBN** 적용")
    if use_csls:
        msg.insert(1, f"**Hub 보정(CSLS)** β={beta}")
    return img(q_level, "query", Z["q_names"][sel]), top, "  \n".join(msg)


# ------------------------------------------------------------------ 텍스트 검색 (규칙 기반)
KEYWORDS = {
    "sleeve": {"long": ["긴팔", "long sleeve"], "short": ["반팔", "short sleeve", "t-shirt"]},
    "lower_type": {"pants": ["긴 바지", "청바지", "pants", "jeans"], "shorts": ["반바지", "shorts"],
                   "skirt": ["치마", "원피스", "skirt", "dress"]},
    "bag": {"backpack": ["백팩", "배낭", "backpack"], "handbag": ["핸드백", "handbag"],
            "bag": ["가방", "숄더백", "shoulder bag"]},
    "hat": {"True": ["모자", "hat", "cap"]},
}
COLORS = {"검": "black", "흰": "white", "하얀": "white", "빨": "red", "보라": "purple", "노란": "yellow",
          "회색": "gray", "파란": "blue", "초록": "green", "분홍": "pink", "갈색": "brown"}
PAT = (r"(검|흰|하얀|빨|보라|노란|회색|파란|초록|분홍|갈색)\S*\s+(?:(?:긴팔|반팔|긴|짧은)\s*)?"
       r"(상의|티셔츠|티|셔츠|반바지|바지|하의|치마)")


def parse_rules(text):
    t, out = text.lower(), {}
    for attr, table in KEYWORDS.items():
        for v, kws in table.items():
            if any(k in t for k in kws):
                out[attr] = v
                break
    for m in re.finditer(PAT, text):
        attr = "upper_color" if m.group(2) in ("상의", "티셔츠", "티", "셔츠") else "lower_color"
        if COLORS[m.group(1)] in ATTR_VALUES[attr]:
            out.setdefault(attr, COLORS[m.group(1)])
    if "청바지" in text:
        out.setdefault("lower_color", "blue")
    return out


def text_search(text, g_level):
    attrs = parse_rules(text or "")
    if not attrs:
        return "인식된 속성이 없습니다. 예: 흰색 반팔 상의, 검은 반바지, 백팩", []
    score = agreement(attrs, g_level)
    seen, res = set(), []
    for j in np.argsort(-score):
        if GP[j] in seen or GP[j] <= 0:
            continue
        seen.add(GP[j])
        res.append((img(g_level, "gallery", Z["g_names"][j]), f"{score[j]:.2f} · {caption_of(GP[j])}"))
        if len(res) >= 20:
            break
    chips = " ".join(f"`{ATTR_KO[k]}: {VALUE_KO.get(v, v)}`" for k, v in attrs.items())
    return f"**추출된 속성** (키워드 규칙): {chips}", res


# ------------------------------------------------------------------ 벤치마크
def read(name):
    p = D / "results" / name
    return pd.read_csv(p) if p.exists() else None


def bench_overall(metric):
    col = {"Rank-1": "rank1", "mAP": "mAP"}[metric]
    label = {("none", "cosine"): "1. 외형만", ("none", "csls"): "2. + CSLS",
             ("none", "csls+witness-probe"): "3. + CSLS + 목격자 속성", ("adabn", "cosine"): "4. AdaBN",
             ("adabn", "csls+witness-probe"): "5. AdaBN + CSLS + 목격자 속성"}
    rows = []
    for name, feat in [("hubness_wide.csv", "none"), ("hubness_adabn.csv", "adabn")]:
        df = read(name)
        if df is None:
            continue
        for _, r in df.iterrows():
            if (feat, r.method) in label and r.query_level != "normal":
                rows.append({"조건": f"{r.query_level}→{r.gallery_level}", "방법": label[(feat, r.method)],
                             metric: r[col] * 100})
    return pd.DataFrame(rows)


def bench_tracking():
    rows = []
    for name in ["tracking.csv", "tracking_yolo_adabn.csv"]:
        df = read(name)
        if df is None:
            continue
        for _, r in df.iterrows():
            det = r.get("detector", "yolo11m")
            rows.append({"조도": r.level, "방법": f"{r.tracker} ({det})", "HOTA": r.HOTA * 100})
    return pd.DataFrame(rows)


def bench_attr():
    df = read("attr_acc_vlm.csv")
    if df is None:
        return pd.DataFrame()
    return pd.DataFrame([{"조도": r.level, "모델": r.model.split(":")[-1], "평균 정확도": r.mean_acc * 100}
                         for _, r in df.iterrows()])


CSS = """
.gradio-container { font-family: 'IBM Plex Sans KR', sans-serif !important; }
#title h1 { color: #1B2A4A; margin-bottom: 0; }
.primary-btn { background: #1B2A4A !important; color: white !important; }
"""
HEAD = ('<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&display=swap" '
        'rel="stylesheet">')

with gr.Blocks(title="VAST-ReID Demo") as demo:
    gr.Markdown("# VAST-ReID\n저조도 사람 재식별(Re-ID) 벤치마크 공개 데모 · Market-1501 기반 저조도 합성 · "
                "[GitHub](https://github.com/20221000/VAST-ReID-)  \n"
                "공개용 축소 샘플 갤러리입니다. 연구·교육 목적에 한해 사용하며, 성별·나이 등 민감 속성은 다루지 않습니다.",
                elem_id="title")
    with gr.Tabs():
        with gr.Tab("이미지 검색"):
            with gr.Row():
                with gr.Column(scale=2):
                    q_level = gr.Radio(LEVELS, value="extreme", label="query 조도")
                    samples = gr.Gallery(show_samples("extreme"), label="query 샘플 (클릭해서 선택)",
                                         columns=8, height=300, allow_preview=False)
                    sel = gr.State(None)
                    g_level = gr.Radio(LEVELS, value="normal", label="gallery 조도")
                    use_adabn = gr.Checkbox(True, label="AdaBN 사용")
                    with gr.Accordion("Hub 보정 (CSLS)", open=False):
                        use_csls = gr.Checkbox(True, label="Hub 보정 사용")
                        beta = gr.Slider(0, 2, value=1.0, step=0.25, label="β")
                    with gr.Accordion("목격자 속성 (재순위화)", open=False):
                        attr_inputs = [gr.Dropdown([ANY] + v, value=ANY, label=ATTR_KO[k])
                                       for k, v in ATTR_VALUES.items()]
                        lam = gr.Slider(0, 1, value=0.5, step=0.05, label="λ")
                    btn = gr.Button("검색", elem_classes="primary-btn")
                with gr.Column(scale=3):
                    with gr.Row():
                        q_view = gr.Image(label="query", height=256, interactive=False)
                        info = gr.Markdown()
                    results = gr.Gallery(label="Top-10", columns=5, height=560)

            def on_select(evt: gr.SelectData):
                return evt.index
            q_level.change(show_samples, q_level, samples)
            samples.select(on_select, None, sel)
            btn.click(image_search, [sel, q_level, g_level, use_adabn, use_csls, beta, lam, *attr_inputs],
                      [q_view, results, info])
        with gr.Tab("텍스트 검색"):
            with gr.Row():
                with gr.Column(scale=2):
                    text = gr.Textbox(label="인상착의", value="흰색 반팔 상의, 검은 반바지, 백팩을 멘 사람")
                    tg_level = gr.Radio(LEVELS, value="normal", label="gallery 조도")
                    tbtn = gr.Button("검색", elem_classes="primary-btn")
                    tinfo = gr.Markdown()
                with gr.Column(scale=3):
                    tres = gr.Gallery(label="속성 일치 순 (사람별 1장)", columns=5, height=600)
            tbtn.click(text_search, [text, tg_level], [tinfo, tres])
        with gr.Tab("벤치마크"):
            gr.Markdown("전체 벤치마크(query 3,368 × gallery 15,913)의 실제 실행 결과입니다.")
            metric = gr.Radio(["Rank-1", "mAP"], value="Rank-1", label="지표")
            p0 = gr.BarPlot(bench_overall("Rank-1"), x="조건", y="Rank-1", color="방법",
                            title="ReID: 방법별 비교 (코사인 기준)", height=380)
            gr.BarPlot(bench_tracking(), x="방법", y="HOTA", color="조도", title="트래킹: 저조도 MOT17 HOTA",
                       height=380, x_label_angle=-30)
            gr.BarPlot(bench_attr(), x="모델", y="평균 정확도", color="조도",
                       title="조도별 속성 인식 정확도 (50명 샘플)", height=360)
            metric.change(lambda m: gr.BarPlot(bench_overall(m), y=m), metric, p0)

if __name__ == "__main__":
    demo.launch(css=CSS, head=HEAD, theme=gr.themes.Soft(primary_hue="blue"))
