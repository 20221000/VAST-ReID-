"""웹 데모(demo/server.py)용 검색·분석 로직. 모든 수치는 캐시된 특징과 results/*.csv에서 계산."""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from vastreid.clip_attr import ATTR_VALUES
from vastreid.data import list_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/vast"
LEVELS = ["normal", "low", "extreme"]
LEVEL_LABEL = {"normal": "Normal", "low": "Low", "extreme": "Extreme"}
ATTR_KO = {"upper_color": "상의", "lower_color": "하의 색", "sleeve": "소매", "lower_type": "하의",
           "bag": "가방", "hat": "모자"}
VALUE_KO = {"black": "검정", "white": "흰색", "red": "빨강", "purple": "보라", "yellow": "노랑", "gray": "회색",
            "blue": "파랑", "green": "초록", "pink": "분홍", "brown": "갈색", "unknown": "알 수 없음",
            "long": "긴팔", "short": "반팔", "pants": "긴 바지", "shorts": "반바지", "skirt": "치마",
            "backpack": "백팩", "handbag": "핸드백", "bag": "가방", "none": "없음", "True": "씀", "False": "없음"}
PERSONS = {int(k): v for k, v in json.loads(
    (ROOT / "annotations/person_attributes.json").read_text(encoding="utf-8"))["persons"].items()}
N_SAMPLES = 24
BASE, FT = "osnet_x1_0", "osnet_x1_0-ft"  # 사전학습 / 저조도 미세조정(tools/finetune.py) 특징 캐시
ATTR_STRENGTH = {"strong": 0.3, "max": 0.9}
# 갤러리 속성 예측: 미세조정 백본은 CLIP + 미세조정 OSNet 특징 결합 분류기 (tools/attr_probe.py --src clip+osnet-ft)
ATTR_PROBE = {BASE: "probe", FT: "probe-clip+osnet-ft"}  # λ (속성 하나당 가중치 = λ/6 → 0.05, 0.15)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ 인덱스
@lru_cache(maxsize=None)
def items(level: str, split: str):
    return list_split(DATA, level, split)


@lru_cache(maxsize=None)
def feats(level: str, split: str, feat: str = "none", prefix: str = BASE) -> np.ndarray:
    tag = f"adabn-{level}" if feat == "adabn" and level != "normal" else "none"
    d = torch.load(ROOT / f"outputs/features/{prefix}__{tag}__{level}__{split}.pt", weights_only=True)
    return torch.nn.functional.normalize(d["feats"].float(), dim=1).numpy()


@lru_cache(maxsize=None)
def probs(level: str, split: str, name: str = "probe") -> dict:
    z = np.load(ROOT / f"outputs/attr_probs/{name}__{level}__{split}.npz")
    return {k: z[k].astype(np.float32) for k in ATTR_VALUES}


@lru_cache(maxsize=None)
def hub_score(q_level: str, g_level: str, feat: str, k: int = 50, prefix: str = BASE) -> np.ndarray:
    g = torch.from_numpy(feats(g_level, "gallery", feat, prefix)).to(DEV)
    r = torch.from_numpy(feats(q_level, "train", feat, prefix)).to(DEV)
    return torch.cat([(g[s:s + 4096] @ r.T).topk(k, dim=1).values.mean(1)
                      for s in range(0, len(g), 4096)]).cpu().numpy()


@lru_cache(maxsize=None)
def hub_freq(feat: str, q_level: str, prefix: str = BASE) -> np.ndarray | None:
    pre = "" if prefix == BASE else f"{prefix}__"
    p = ROOT / f"outputs/demo/hub_freq__{pre}{feat}__{q_level}__normal.npy"
    return np.load(p) if p.exists() else None


@lru_cache(maxsize=1)
def luminance() -> dict:
    out = {}
    with open(DATA / "synthesis_log.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["level"], r["file"])] = float(r["mean_luminance"])
    return out


@lru_cache(maxsize=1)
def sample_idx() -> list[int]:
    q, seen, out = items("normal", "query"), set(), []
    for i in range(0, len(q), 37):
        if q[i].pid not in seen and q[i].pid in PERSONS:
            seen.add(q[i].pid)
            out.append(i)
        if len(out) >= N_SAMPLES:
            break
    return out


def img_url(level: str, split: str, name: str) -> str:
    return f"/img/{level}/{split}/{name}"


def attrs_ko(pid: int) -> dict | None:
    p = PERSONS.get(pid)
    if not p:
        return None
    return {k: VALUE_KO.get(str(p[k]), str(p[k])) for k in ATTR_VALUES}


def caption(pid: int) -> str:
    p = PERSONS.get(pid)
    return p["caption"] if p else "속성 정보 없음 (사람이 아니거나 일부만 찍힌 이미지)"


# ------------------------------------------------------------------ 이미지 검색
def samples(level: str) -> list[dict]:
    q = items(level, "query")
    return [{"idx": n, "name": q[i].path.stem, "url": img_url(level, "query", q[i].path.name)}
            for n, i in enumerate(sample_idx())]


def _agreement(choices: dict, gp: dict) -> np.ndarray | None:
    """목격자 진술 속성 일치도. λ는 속성 6개 평균 기준으로 검증했으므로, 입력한 속성 수 n에 맞춰 n/6을 곱해
    속성 하나당 가중치를 일정하게 유지 (속성 1~2개만 입력해도 그 하나가 점수를 좌우하지 않도록)."""
    terms = []
    for k, v in choices.items():
        if v and v in ATTR_VALUES.get(k, []):
            terms.append(gp[k][:, ATTR_VALUES[k].index(v)])
    return np.mean(terms, axis=0) * len(terms) / len(ATTR_VALUES) if terms else None


_HUB_CSV = {(BASE, "none"): "hubness_wide.csv", (BASE, "adabn"): "hubness_adabn.csv",
            (FT, "none"): "hubness_ft_pft.csv", (FT, "adabn"): "hubness_ft_adabn_pft.csv"}


@lru_cache(maxsize=None)
def csls_params(q_level: str, g_level: str, feat: str, prefix: str = BASE) -> dict | None:
    """tools/hubness.py가 train 검증 분할에서 고른 k, β, λ (CSLS + 목격자 속성)."""
    proto = "same" if q_level == g_level else "cross"
    r = _pick(_csv(_HUB_CSV[(prefix, feat)]), method="csls+witness-probe", protocol=proto,
              query_level=q_level, gallery_level=g_level)
    return None if r is None else {"k": int(r["k"]), "beta": float(r["beta"]), "lam": float(r["lambda"])}


def _ap(order: np.ndarray, gpids: np.ndarray, valid: np.ndarray, pid: int) -> float | None:
    m = (gpids[order] == pid)[valid[order]]
    if not m.any():
        return None
    hits = np.cumsum(m)
    pos = np.where(m)[0]
    return float((hits[pos] / (pos + 1)).mean())


def _failure_causes(q_pid: int, g_item, j: int, q_level: str, feat: str, use_csls: bool,
                    prefix: str = BASE) -> list[str]:
    causes = []
    if g_item.pid <= 0:
        causes.append("사람이 아니거나 일부만 찍힌 이미지")
    freq = hub_freq(feat, q_level, prefix) if q_level != "normal" else None  # tools/failure_analysis.py가 저장한 빈도
    if freq is not None and freq[j] >= 34:
        causes.append(f"hub 이미지 · 어두운 query {int(freq[j]):,}개의 Top-10에 등장")
    a, b = PERSONS.get(q_pid), PERSONS.get(g_item.pid)
    if a and b:
        same = [k for k in ATTR_VALUES if str(a[k]) == str(b[k]) and str(a[k]) not in ("unknown", "False", "none")]
        if len(same) >= 3:
            causes.append("유사 의복 · " + " + ".join(f"{ATTR_KO[k]} {VALUE_KO.get(str(a[k]))}" for k in same[:3]))
    if q_level == "extreme":
        causes.append("Extreme 조도 (평균 밝기 약 0.08)")
    elif q_level == "low":
        causes.append("Low 조도")
    return causes or ["외형 특징 손실"]


@lru_cache(maxsize=None)
def kr_rows(q_level: str, feat: str, prefix: str = BASE) -> np.ndarray | None:
    """tools/build_demo_cache.py가 만든 샘플 쿼리의 k-reciprocal 거리 (24, Ng)."""
    pre = "" if prefix == BASE else f"{prefix}__"
    p = ROOT / f"outputs/demo/kr__{pre}{q_level}__{feat}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


@lru_cache(maxsize=None)
def kr_params(q_level: str, feat: str, prefix: str = BASE) -> dict | None:
    """results/kreciprocal.csv에서 검증 분할로 고른 k, β_kr, λ_kr (gallery = normal)."""
    proto = "same" if q_level == "normal" else "cross"
    rows = [r for r in _csv("kreciprocal.csv" if prefix == BASE else "kreciprocal_ft_pft.csv") if r["feat"] == feat and r["protocol"] == proto
            and r["query_level"] == q_level]
    get = lambda m: next((r for r in rows if r["method"] == m), None)
    a, b = get("kr+csls"), get("kr+csls+witness-probe")
    if not (a and b):
        return None
    return {"k": int(a["k"]), "beta": float(a["beta"]), "lam": float(b["lambda"])}


def search(q_level: str, sample: int | None, g_level: str = "normal", adabn: bool = True, csls: bool = True,
           beta: float = 1.0, lam: float = 0.5, attrs: dict | None = None,
           src_level: str | None = None, src_gallery: int | None = None, kr: bool = False,
           ft: bool = False, attr_strength: str = "auto") -> dict:
    feat = "adabn" if adabn else "none"
    pf = FT if ft else BASE
    gi, gf = items(g_level, "gallery"), feats(g_level, "gallery", feat, pf)
    if src_gallery is not None:  # 텍스트 검색 결과를 query로 재검색
        lv = src_level or g_level
        qi_ = items(lv, "gallery")[src_gallery]
        qf = feats(lv, "gallery", feat, pf)[src_gallery]
        q_level, q_split = lv, "gallery"
    else:
        qi_ = items(q_level, "query")[sample_idx()[sample]]
        qf = feats(q_level, "query", feat, pf)[sample_idx()[sample]]
        q_split = "query"
    cos = gf @ qf
    kr_used, kr_note = False, None
    rows = kr_rows(q_level, feat, pf) if kr else None
    kp = kr_params(q_level, feat, pf) if kr else None
    if kr and (src_gallery is not None or g_level != "normal" or rows is None or kp is None):
        kr_note = "재순위화는 샘플 쿼리(주간 gallery)에만 미리 계산되어 있어 이번 검색에는 적용하지 않았습니다"
    elif kr:
        d = rows[sample].copy()  # k-reciprocal 거리 (작을수록 가까움)
        if csls:
            d = d + kp["beta"] * hub_score(q_level, g_level, feat, kp["k"], pf)
        score, kr_used, lam = -d, True, kp["lam"]  # 속성 가중치도 k-reciprocal 척도에 맞춘 값 사용
    if not kr_used:
        score = cos.copy()
        cp = csls_params(q_level, g_level, feat, pf)
        if cp:  # 화면에서 넘어온 고정값(β=1, λ=0.5) 대신 검증 분할에서 고른 값 사용
            beta, lam = cp["beta"], cp["lam"]
        if csls:
            score = score - beta * hub_score(q_level, g_level, feat, cp["k"] if cp else 50, pf)
    agr = _agreement(attrs or {}, probs(g_level, "gallery", ATTR_PROBE[pf]))
    # 진술 반영 강도: auto = 검증 분할에서 고른 λ (정답률 최적), strong/max = 시연용으로 진술을 크게 반영
    lam = ATTR_STRENGTH.get(attr_strength, lam)
    if agr is not None:
        score = score + lam * agr
    gp = np.array([g.pid for g in gi])
    gc = np.array([g.camid for g in gi])
    valid = ~((gp == qi_.pid) & (gc == qi_.camid))
    if q_split == "gallery":
        valid &= np.array([g.path.name != qi_.path.name for g in gi])
    score = np.where(valid, score, -np.inf)
    order = np.argsort(-score)
    res = []
    for r, j in enumerate(order[:10], 1):
        g = gi[j]
        ok = bool(g.pid == qi_.pid)
        res.append({"rank": r, "url": img_url(g_level, "gallery", g.path.name), "name": g.path.stem, "glvl": g_level,
                    "score": round(float(score[j]), 3), "sim": round(float(cos[j]), 3), "pid": int(g.pid),
                    "cam": f"cam{g.camid + 1:02d}",
                    "illum": LEVEL_LABEL[g_level], "ok": ok,
                    "causes": [] if ok else _failure_causes(qi_.pid, g, int(j), q_level, feat, csls, pf)})
    ap = _ap(order, gp, valid, qi_.pid)
    ranks = [r for r, j in enumerate(order[np.isfinite(score[order])], 1) if gp[j] == qi_.pid]
    lum = luminance().get((q_level, f"{q_split}/{qi_.path.name}"))
    return {
        "query": {"url": img_url(q_level, q_split, qi_.path.name), "name": qi_.path.stem,
                  "level": LEVEL_LABEL[q_level], "luminance": lum, "pid": int(qi_.pid),
                  "caption": caption(qi_.pid), "original": img_url("normal", q_split, qi_.path.name),
                  "split": q_split, "lvl": q_level},
        "gallery_size": len(gi), "g_level": LEVEL_LABEL[g_level],
        "rank1": bool(res and res[0]["ok"]), "ap": None if ap is None else round(ap, 3),
        "first_correct": ranks[0] if ranks else None, "results": res, "kr": kr_used, "kr_note": kr_note,
        "ft": ft, "attr_weight": round(lam / len(ATTR_VALUES), 3), "attr_strength": attr_strength,
    }


# ------------------------------------------------------------------ 텍스트 검색
_COLORS = {"검": "black", "흰": "white", "하얀": "white", "빨": "red", "보라": "purple", "노란": "yellow",
           "회색": "gray", "파란": "blue", "초록": "green", "분홍": "pink", "갈색": "brown"}
_PAT = (r"(검|흰|하얀|빨|보라|노란|회색|파란|초록|분홍|갈색)\S*\s+(?:(?:긴팔|반팔|긴|짧은)\s*)?"
        r"(상의|티셔츠|티|셔츠|반바지|바지|하의|치마)")
_KW = {
    "sleeve": {"long": ["긴팔", "long sleeve"], "short": ["반팔", "short sleeve", "t-shirt"]},
    "lower_type": {"pants": ["긴 바지", "긴바지", "청바지", "pants", "jeans"], "shorts": ["반바지", "shorts"],
                   "skirt": ["치마", "원피스", "skirt", "dress"]},
    "bag": {"backpack": ["백팩", "배낭", "backpack"], "handbag": ["핸드백", "handbag"],
            "bag": ["가방", "숄더백", "shoulder bag"]},
    "hat": {"True": ["모자", "hat", "cap"]},
}


def parse_rules(text: str) -> dict:
    t, out = text.lower(), {}
    for attr, table in _KW.items():
        for v, kws in table.items():
            if any(k in t for k in kws):
                out[attr] = v
                break
    for m in re.finditer(_PAT, text):
        attr = "upper_color" if m.group(2) in ("상의", "티셔츠", "티", "셔츠") else "lower_color"
        if _COLORS[m.group(1)] in ATTR_VALUES[attr]:
            out.setdefault(attr, _COLORS[m.group(1)])
    if "바지" in text and "lower_type" not in out:
        out["lower_type"] = "pants"
    return out


def parse_llm(text: str) -> dict:
    from vastreid.llm import LLM
    llm = LLM(max_calls=3)
    raw = llm.chat("Extract clothing and carried-item attributes of a pedestrian from a description. "
                   "Never infer gender or age. Answer with one JSON object only.",
                   "Description: " + text + "\nReturn JSON with any of these keys you can determine "
                   "(omit unknown keys): " + json.dumps(ATTR_VALUES), json_mode=True, tag="web_text_search") or "{}"
    d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1] or "{}")
    out = {}
    for k, vals in ATTR_VALUES.items():
        v = str(d.get(k, "")).strip()
        v = {"true": "True", "false": "False"}.get(v.lower(), v.lower() if k != "hat" else v)
        if v in vals:
            out[k] = v
    return out


def text_search(text: str, g_level: str = "normal", use_llm: bool = False, extra: dict | None = None,
                n: int = 12) -> dict:
    how = "키워드 규칙"
    attrs = {}
    if use_llm:
        try:
            attrs, how = parse_llm(text), "LLM"
        except Exception as e:  # 키 없음, 호출 상한 등 → 규칙 기반
            how = f"키워드 규칙 (LLM 사용 불가: {type(e).__name__})"
    if not attrs:
        attrs = parse_rules(text)
    attrs.update({k: v for k, v in (extra or {}).items() if v in ATTR_VALUES.get(k, [])})
    chips = [{"key": k, "value": v, "label": f"{ATTR_KO[k]} · {VALUE_KO.get(v, v)}"} for k, v in attrs.items()]
    if not attrs:
        return {"how": how, "chips": [], "results": []}
    gi = items(g_level, "gallery")
    score = _agreement(attrs, probs(g_level, "gallery", ATTR_PROBE[FT]))  # 더 정확한 결합 분류기
    seen, res = set(), []
    for j in np.argsort(-score):
        g = gi[j]
        if g.pid <= 0 or g.pid in seen:
            continue
        seen.add(g.pid)
        res.append({"gidx": int(j), "url": img_url(g_level, "gallery", g.path.name), "name": g.path.stem,
                    "score": round(float(score[j]), 2), "pid": int(g.pid), "cam": f"cam{g.camid + 1:02d}",
                    "illum": LEVEL_LABEL[g_level], "caption": caption(g.pid), "attrs": attrs_ko(g.pid)})
        if len(res) >= n:
            break
    return {"how": how, "chips": chips, "results": res, "g_level": g_level}


# ------------------------------------------------------------------ 벤치마크
def _csv(name: str) -> list[dict]:
    p = ROOT / "results" / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick(rows, **kw):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    return None


def bench() -> dict:
    base, wide, ada = _csv("baseline.csv"), _csv("hubness_wide.csv"), _csv("hubness_adabn.csv")
    enh2, krc = _csv("enhance_v2.csv"), _csv("kreciprocal.csv")
    ft, krf = _csv("finetune.csv"), _csv("kreciprocal_ft.csv")

    def triple(rows, method, extra=None):
        out = {}
        for m in ["rank1", "mAP"]:
            vals = []
            for ql in LEVELS:
                proto = "same" if ql == "normal" else "cross"
                r = _pick(rows, method=method, protocol=proto, query_level=ql, **(extra or {}))
                vals.append(None if r is None else round(float(r[m]) * 100, 1))
            out[m] = vals
        return out

    zd = {}
    for m in ["rank1", "mAP"]:
        zd[m] = [None if r is None else round(float(r[m]) * 100, 1) for r in [
            _pick(enh2, enhance="zerodce", denoise="none", protocol="same", query_level="normal"),
            _pick(enh2, enhance="zerodce", denoise="none", protocol="cross", query_level="low"),
            _pick(enh2, enhance="zerodce", denoise="none", protocol="cross", query_level="extreme")]]
    rows = [
        {"name": "OSNet (외형만)", **triple(wide, "cosine")},
        {"name": "OSNet + Zero-DCE (게이팅)", **zd, "note": "유클리드 거리"},
        {"name": "OSNet + CSLS + 목격자 속성", **triple(wide, "csls+witness-probe")},
        {"name": "OSNet + AdaBN", **triple(ada, "cosine")},
        {"name": "AdaBN + CSLS + 목격자 속성", **triple(ada, "csls+witness-probe")},
    ]
    if krc:
        rows.append({"name": "AdaBN + CSLS + 재순위화 + 목격자 속성",
                     **triple(krc, "kr+csls+witness-probe", {"feat": "adabn"}), "note": "k-reciprocal 표준값"})
    if ft:
        rows.append({"name": "저조도 미세조정 OSNet", **triple(ft, "cosine", {"feat": "none"}),
                     "note": "train 3조도 12 에폭"})
    if krf:
        rows.append({"name": "미세조정 + CSLS + 재순위화 + 목격자 속성",
                     **triple(krf, "kr+csls+witness-probe", {"feat": "none"})})
    b_n = _pick(base, protocol="same", query_level="normal")
    b_e = _pick(base, protocol="cross", query_level="extreme")
    z_l = _pick(enh2, enhance="zerodce", denoise="none", protocol="cross", query_level="low")
    c_l = _pick(wide, method="cosine", protocol="cross", query_level="low")
    best = _pick(ada, method="csls+witness-probe", protocol="cross", query_level="low")
    best_kr = _pick(krc, feat="adabn", method="kr+csls+witness-probe", protocol="cross", query_level="low")
    best_ft = _pick(krf, feat="none", method="kr+csls+witness-probe", protocol="cross", query_level="low")
    for cand in (best_kr, best_ft):
        if cand and (not best or float(cand["rank1"]) > float(best["rank1"])):
            best = cand
    cards = {
        "dataset": {"persons": len(PERSONS), "images": 36036, "levels": 3},
        "drop": None if not (b_n and b_e) else round((float(b_e["rank1"]) - float(b_n["rank1"])) * 100, 1),
        "zerodce_low": None if not (z_l and c_l) else round((float(z_l["rank1"]) - float(c_l["rank1"])) * 100, 1),
        "best_gain": None if not (best and c_l) else round((float(best["rank1"]) - float(c_l["rank1"])) * 100, 1),
        "best_value": None if not best else round(float(best["rank1"]) * 100, 1),
    }
    fail = {}
    for r in _csv("failure_causes.csv"):
        fail.setdefault(f"{r['setting']}|{r['query_level']}", []).append(
            {"cause": r["cause"], "ratio": round(float(r["ratio"]) * 100, 1), "count": int(r["count"]),
             "n_wrong": int(r["n_wrong_rank1"])})
    trk = {}
    for name, det in [("tracking.csv", "yolo11m"), ("tracking_yolo_adabn.csv", "yolo11m+adabn")]:
        for r in _csv(name):
            trk.setdefault(r["level"], []).append({"tracker": r["tracker"], "detector": det,
                                                   "HOTA": round(float(r["HOTA"]) * 100, 1),
                                                   "IDF1": round(float(r["IDF1"]) * 100, 1),
                                                   "AssA": round(float(r["AssA"]) * 100, 1),
                                                   "IDSW": int(r["IDSW"])})
    return {"rows": rows, "cards": cards, "failure": fail, "tracking": trk}


# ------------------------------------------------------------------ 트래킹 프레임
TRACK_SEQ = "MOT17-09-FRCNN"


@lru_cache(maxsize=None)
def _tracks(level: str, tracker: str, seq: str) -> dict:
    out = {}
    p = ROOT / f"outputs/mot_results/{level}/{tracker}/{seq}.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            f, tid, x, y, w, h = line.split(",")[:6]
            out.setdefault(int(f), []).append((int(tid), float(x), float(y), float(w), float(h)))
    return out


def track_frame(level: str, i: int, tracker: str = "botsort-osnet-adabn", seq: str = TRACK_SEQ) -> bytes:
    import cv2
    frames = sorted((ROOT / f"data/vast_mot/{level}/{seq}/img1").glob("*.jpg"))
    i = max(1, min(i, len(frames)))
    img = cv2.imread(str(frames[i - 1]))
    palette = [(71, 181, 245), (240, 180, 141), (120, 200, 120), (200, 140, 220), (90, 160, 250)]
    for tid, x, y, w, h in _tracks(level, tracker, seq).get(i, []):
        c = palette[tid % len(palette)]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 3)
        cv2.rectangle(img, (int(x), int(y) - 26), (int(x) + 78, int(y)), c, -1)
        cv2.putText(img, f"ID {tid:02d}", (int(x) + 6, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


# ------------------------------------------------------------------ 생성형 AI (비전 LLM): 캡션, 오답 설명
# 클릭할 때 한 번만 호출하고 vastreid.llm 디스크 캐시로 재사용 (반 공용 키 → 서버 실행당 호출 상한)
AI_PROVIDER = "openai"
AI_MAX_CALLS = 120
_AI_SYSTEM = ("You describe pedestrians in surveillance images for a person re-identification research demo. "
              "Describe only clothing, colors, carried items and visible context. Never mention or guess gender, age, "
              "race, ethnicity, or identity. Answer in natural Korean.")


@lru_cache(maxsize=1)
def _llm():
    from vastreid.llm import LLM
    return LLM(provider=AI_PROVIDER, max_calls=AI_MAX_CALLS)


def _img_path(level: str, split: str, name: str) -> Path:
    if level not in LEVELS or split not in ("query", "gallery") or not re.match(r"^-?\d{1,4}_c\ds\d_\d{6}_\d{2}$", name):
        raise ValueError("bad image")
    p = DATA / level / split / f"{name}.jpg"
    if not p.is_file():
        raise ValueError("no image")
    return p


def ai_caption(level: str, split: str, name: str) -> dict:
    """인물 사진 한 장을 보고 옷차림·소지품을 자연스러운 한 문장으로 묘사 (시맨틱 캡션)."""
    p = _img_path(level, split, name)
    text = _llm().chat(_AI_SYSTEM,
                       "이 보행자의 옷차림과 소지품을 CCTV 검색에 쓸 수 있도록 자연스러운 한국어 한두 문장으로 묘사해 주세요. "
                       "상의·하의의 색과 종류, 가방·모자 등 소지품을 포함하고, 사진에서 확실하지 않은 것은 추측하지 마세요.",
                       images=[p], tag="demo_caption")
    return {"text": (text or "").strip(), "model": _llm().pcfg["vision_model"], "summary": _llm().summary()}


def ai_explain(q_level: str, q_split: str, q_name: str, g_level: str, g_name: str, causes: list[str]) -> dict:
    """쿼리(현재 조도)와 오답 결과를 함께 보고, 검색 모델이 왜 헷갈렸는지 설명."""
    qp, gp = _img_path(q_level, q_split, q_name), _img_path(g_level, "gallery", g_name)
    hint = "; ".join(causes[:4]) if causes else "없음"
    text = _llm().chat(_AI_SYSTEM,
                       f"첫 번째 사진은 {LEVEL_LABEL[q_level]} 조도의 검색 쿼리, 두 번째 사진은 검색 결과로 나온 **다른 사람**입니다. "
                       f"사람 재식별 모델이 왜 두 사람을 같은 사람으로 헷갈렸을지 2~3문장으로 설명해 주세요. "
                       f"옷차림·소지품의 공통점, 조명·노이즈로 사라진 단서, 그리고 두 사람을 구분할 수 있는 차이점을 포함하세요. "
                       f"분석 시스템이 자동으로 찾은 실패 원인 후보: {hint}",
                       images=[qp, gp], tag="demo_explain")
    return {"text": (text or "").strip(), "model": _llm().pcfg["vision_model"], "summary": _llm().summary()}
