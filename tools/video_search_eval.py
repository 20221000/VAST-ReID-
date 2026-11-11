"""브라우저 '영상에서 찾기'와 같은 절차를 Python(onnxruntime)으로 재현해 예제 영상의 정확도를 잰다.

절차(app.js와 같음): 영상에서 초당 fps장 추출 → YOLO11n 사람 검출(conf≥0.35, 높이≥min_h) → 증류 OSNet 특징 →
인접 프레임 IoU로 같은 사람 묶기(트랙) → 트랙 점수 = 상위 3개 프레임 유사도 평균 → 점수 순 후보.
쿼리: 주간(normal) 프레임에서 GT 상자로 자른 사람, 영상: 같은 장면의 저조도(low/extreme) 프레임.
쿼리 사진과 같은 순간(기본 ±2초)은 점수에서 빼서, 같은 장면을 어둡게 한 것만 맞히는 쉬운 경우를 막는다.
정답 판정: 후보 트랙 상자의 다수가 쿼리와 같은 GT id(IoU≥0.5)이면 정답.
결과: results/video_search.csv(시퀀스·조도별 Rank-1/5, 쿼리 수), --make-sample 이면 예제 영상·인물을 demo/static/video/에 저장.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "demo/static/models"
RAW = ROOT / "data/raw/mot17/MOT17/train"
IMGSZ = 480


def load_gt(seq: str) -> dict[int, list[tuple[int, np.ndarray]]]:
    gt = defaultdict(list)
    for line in open(RAW / f"{seq}/gt/gt.txt"):
        f, i, x, y, w, h, flag, cls, vis = line.strip().split(",")[:9]
        if int(flag) == 1 and int(cls) == 1:
            gt[int(f)].append((int(i), np.array([float(x), float(y), float(x) + float(w), float(y) + float(h)]), float(vis)))
    return gt


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class Pipe:
    def __init__(self):
        self.det = ort.InferenceSession(str(MOD / "det_yolo11n.onnx"), providers=["CPUExecutionProvider"])
        self.reid = ort.InferenceSession(str(MOD / "reid_osnet.onnx"), providers=["CPUExecutionProvider"])

    def detect(self, img, conf=0.35, min_h=40):
        h, w = img.shape[:2]
        s = IMGSZ / max(h, w)
        nh, nw = round(h * s), round(w * s)
        pad = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = pad[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        o = self.det.run(None, {"images": x})[0][0]
        o = o[(o[:, 4] >= conf) & (o[:, 5] == 0)]
        b = o[:, :4] / s
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w); b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        keep = (b[:, 3] - b[:, 1]) >= min_h
        return b[keep], o[keep, 4]

    def embed(self, img, boxes):
        if len(boxes) == 0:
            return np.zeros((0, 512), np.float32)
        crops = []
        for x1, y1, x2, y2 in boxes.astype(int):
            c = img[y1:y2, x1:x2]
            crops.append(cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR)[..., ::-1].transpose(2, 0, 1))
        x = np.stack(crops).astype(np.float32) / 255.0
        return self.reid.run(None, {"images": x})[0]


def build_tracks(frames):
    """frames: [(t, boxes, feats)] → 트랙 목록 {ids:[(fi, j)]}. 인접 샘플 프레임 IoU≥0.2 탐욕 연결."""
    tracks, prev = [], []  # prev: [(track_idx, box)]
    for fi, (_, boxes, _) in enumerate(frames):
        cur, used = [], set()
        if prev and len(boxes):
            m = iou(boxes, np.stack([b for _, b in prev]))
            for j, k in sorted(((j, k) for j in range(len(boxes)) for k in range(len(prev))), key=lambda p: -m[p]):
                if m[j, k] < 0.2 or j in [c[2] for c in cur] or k in used:
                    continue
                used.add(k)
                cur.append((prev[k][0], boxes[j], j))
        for j in range(len(boxes)):
            if j not in [c[2] for c in cur]:
                tracks.append([])
                cur.append((len(tracks) - 1, boxes[j], j))
        for ti, _, j in cur:
            tracks[ti].append((fi, j))
        prev = [(ti, b) for ti, b, _ in cur]
    return tracks


def rank_tracks(frames, tracks, q, skip=None):
    """skip(프레임 번호) → True면 그 프레임은 점수에서 뺌(쿼리와 같은 순간 제외용)."""
    out = []
    for tr in tracks:
        sims = np.array([float(frames[fi][2][j] @ q) for fi, j in tr if not (skip and skip(frames[fi][0]))])
        out.append((float(np.sort(sims)[-3:].mean()) if len(sims) else -1.0, tr, sims))
    return sorted(out, key=lambda r: -r[0])


def run_seq(pipe, seq, level, fps, n_query, rng, exclude_sec=2.0):
    gt = load_gt(seq)
    n = len(list((RAW / seq / "img1").glob("*.jpg")))
    step = max(1, round(30 / fps))
    fids = list(range(1, n + 1, step))
    frames, det_gt = [], []
    for f in fids:
        img = cv2.imread(str(ROOT / f"data/vast_mot/{level}/{seq}/img1/{f:06d}.jpg"))
        b, _ = pipe.detect(img)
        frames.append((f, b, pipe.embed(img, b)))
        g = gt.get(f, [])
        if len(b) and g:
            m = iou(b, np.stack([x[1] for x in g]))
            det_gt.append([g[k][0] if m[j, k] >= 0.5 else -1 for j, k in enumerate(m.argmax(1))])
        else:
            det_gt.append([-1] * len(b))
    tracks = build_tracks(frames)
    # 쿼리 후보: 주간 영상 전체에서 보이는 비율 높은 상자. 영상(저조도)에도 검출된 사람만.
    seen = defaultdict(int)
    for fi, ids in enumerate(det_gt):
        for i in ids:
            if i > 0:
                seen[i] += 1
    cand = [i for i, c in seen.items() if c >= 4]
    rng.shuffle(cand)
    rows = []
    for pid in cand[:n_query]:
        opts = [(f, b, v) for f, lst in gt.items() for i, b, v in lst if i == pid and v >= 0.8 and b[3] - b[1] >= 120]
        if not opts:
            continue
        f, b, _ = opts[len(opts) // 2]
        day = cv2.imread(str(RAW / seq / "img1" / f"{f:06d}.jpg"))
        h, w = day.shape[:2]
        b = b.copy(); b[[0, 2]] = b[[0, 2]].clip(0, w); b[[1, 3]] = b[[1, 3]].clip(0, h)
        q = pipe.embed(day, b[None])[0]
        ranked = rank_tracks(frames, tracks, q, skip=lambda fr, f=f: abs(fr - f) <= exclude_sec * 30)

        def label(tr):
            ids = [det_gt[fi][j] for fi, j in tr]
            top = max(set(ids), key=ids.count)
            return top if ids.count(top) * 2 > len(ids) else -1

        labels = [label(tr) for _, tr, _ in ranked]
        first = next((k for k, l in enumerate(labels) if l == pid), None)
        rows.append({"seq": seq, "level": level, "pid": pid, "frame": f, "box": b.round(1).tolist(),
                     "first_rank": first, "n_tracks": len(ranked), "top_score": ranked[0][0]})
    return rows, frames, tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN")
    ap.add_argument("--levels", default="low,extreme")
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--n-query", type=int, default=30)
    ap.add_argument("--exclude-sec", type=float, default=2.0, help="쿼리 사진과 같은 순간(±초)은 점수에서 제외")
    args = ap.parse_args()
    pipe, rng = Pipe(), np.random.default_rng(0)
    allrows, summary = [], []
    for seq in args.seqs.split(","):
        for lv in args.levels.split(","):
            rows, frames, tracks = run_seq(pipe, seq, lv, args.fps, args.n_query, rng, args.exclude_sec)
            allrows += rows
            fr = [r["first_rank"] for r in rows]
            s = {"seq": seq, "level": lv, "n_query": len(rows), "n_frames": len(frames), "n_tracks": len(tracks),
                 "rank1": np.mean([x == 0 for x in fr]), "rank5": np.mean([x is not None and x < 5 for x in fr]),
                 "rank10": np.mean([x is not None and x < 10 for x in fr])}
            summary.append(s)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    with open(ROOT / "results/video_search.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    (ROOT / "results/video_search_queries.json").write_text(json.dumps(allrows, default=float, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
