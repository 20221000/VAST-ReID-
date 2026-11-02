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
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ 인덱스
@lru_cache(maxsize=None)
def items(level: str, split: str):
    return list_split(DATA, level, split)


@lru_cache(maxsize=None)
def feats(level: str, split: str, feat: str = "none") -> np.ndarray:
    tag = f"adabn-{level}" if feat == "adabn" and level != "normal" else "none"
    d = torch.load(ROOT / f"outputs/features/osnet_x1_0__{tag}__{level}__{split}.pt", weights_only=True)
    return torch.nn.functional.normalize(d["feats"].float(), dim=1).numpy()


@lru_cache(maxsize=None)
def probs(level: str, split: str) -> dict:
    z = np.load(ROOT / f"outputs/attr_probs/probe__{level}__{split}.npz")
    return {k: z[k].astype(np.float32) for k in ATTR_VALUES}


@lru_cache(maxsize=None)
def hub_score(q_level: str, g_level: str, feat: str, k: int = 50) -> np.ndarray:
    g = torch.from_numpy(feats(g_level, "gallery", feat)).to(DEV)
    r = torch.from_numpy(feats(q_level, "train", feat)).to(DEV)
    return torch.cat([(g[s:s + 4096] @ r.T).topk(k, dim=1).values.mean(1)
                      for s in range(0, len(g), 4096)]).cpu().numpy()


@lru_cache(maxsize=None)
def hub_freq(feat: str, q_level: str) -> np.ndarray | None:
    p = ROOT / f"outputs/demo/hub_freq__{feat}__{q_level}__normal.npy"
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
    terms = []
    for k, v in choices.items():
        if v and v in ATTR_VALUES.get(k, []):
            terms.append(gp[k] @ np.eye(len(ATTR_VALUES[k]), dtype=np.float32)[ATTR_VALUES[k].index(v)])
    return np.mean(terms, axis=0) if terms else None


def _ap(order: np.ndarray, gpids: np.ndarray, valid: np.ndarray, pid: int) -> float | None:
    m = (gpids[order] == pid)[valid[order]]
    if not m.any():
        return None
    hits = np.cumsum(m)
    pos = np.where(m)[0]
    return float((hits[pos] / (pos + 1)).mean())


def _failure_causes(q_pid: int, g_item, j: int, q_level: str, feat: str, use_csls: bool) -> list[str]:
    causes = []
    if g_item.pid <= 0:
        causes.append("사람이 아니거나 일부만 찍힌 이미지")
    freq = hub_freq(feat, q_level) if q_level != "normal" else None  # tools/failure_analysis.py가 저장한 빈도
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


def search(q_level: str, sample: int | None, g_level: str = "normal", adabn: bool = True, csls: bool = True,
           beta: float = 1.0, lam: float = 0.5, attrs: dict | None = None,
           src_level: str | None = None, src_gallery: int | None = None) -> dict:
    feat = "adabn" if adabn else "none"
    gi, gf = items(g_level, "gallery"), feats(g_level, "gallery", feat)
    if src_gallery is not None:  # 텍스트 검색 결과를 query로 재검색
        lv = src_level or g_level
        qi_ = items(lv, "gallery")[src_gallery]
        qf = feats(lv, "gallery", feat)[src_gallery]
        q_level, q_split = lv, "gallery"
    else:
        qi_ = items(q_level, "query")[sample_idx()[sample]]
        qf = feats(q_level, "query", feat)[sample_idx()[sample]]
        q_split = "query"
    score = gf @ qf
    if csls:
        score = score - beta * hub_score(q_level, g_level, feat)
    agr = _agreement(attrs or {}, probs(g_level, "gallery"))
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
        res.append({"rank": r, "url": img_url(g_level, "gallery", g.path.name), "name": g.path.stem,
                    "score": round(float(score[j]), 3), "pid": int(g.pid), "cam": f"cam{g.camid + 1:02d}",
                    "illum": LEVEL_LABEL[g_level], "ok": ok,
                    "causes": [] if ok else _failure_causes(qi_.pid, g, int(j), q_level, feat, csls)})
    ap = _ap(order, gp, valid, qi_.pid)
    ranks = [r for r, j in enumerate(order[np.isfinite(score[order])], 1) if gp[j] == qi_.pid]
    lum = luminance().get((q_level, f"{q_split}/{qi_.path.name}"))
    return {
        "query": {"url": img_url(q_level, q_split, qi_.path.name), "name": qi_.path.stem,
                  "level": LEVEL_LABEL[q_level], "luminance": lum, "pid": int(qi_.pid),
                  "caption": caption(qi_.pid), "original": img_url("normal", q_split, qi_.path.name)},
        "gallery_size": len(gi), "g_level": LEVEL_LABEL[g_level],
        "rank1": bool(res and res[0]["ok"]), "ap": None if ap is None else round(ap, 3),
        "first_correct": ranks[0] if ranks else None, "results": res,
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
    score = _agreement(attrs, probs(g_level, "gallery"))
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
    enh2 = _csv("enhance_v2.csv")

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
    b_n = _pick(base, protocol="same", query_level="normal")
    b_e = _pick(base, protocol="cross", query_level="extreme")
    z_l = _pick(enh2, enhance="zerodce", denoise="none", protocol="cross", query_level="low")
    c_l = _pick(wide, method="cosine", protocol="cross", query_level="low")
    best = _pick(ada, method="csls+witness-probe", protocol="cross", query_level="low")
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
