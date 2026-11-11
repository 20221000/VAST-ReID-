"""'영상에서 찾기' 예제 영상·인물 만들기 + 그 영상 파일 그대로 정확도 측정.

- 영상: MOT17-04 저조도(low) 합성 프레임을 15fps·960×540 MP4(H.264)으로 → demo/static/video/sample_low.mp4
- 정확도: 만든 MP4를 다시 읽어 브라우저와 같은 절차(초당 2장, 높이 ≥ 프레임 3.7%)로 검색.
  쿼리 = 주간 원본의 같은 사람(GT 상자, 가림 적고 큰 상자), 쿼리와 같은 순간 ±2초는 점수에서 제외.
- 예제 인물 4명: 위 평가에서 1위로 찾은 사람 중 화면 위치가 겹치지 않게 고름(전체 정답률은 sample.json에 함께 기록해 화면에 표시).
결과: demo/static/video/{sample_low.mp4, q1~4.jpg, sample.json}, results/video_sample.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from video_search_eval import RAW, Pipe, build_tracks, iou, load_gt, rank_tracks  # noqa: E402

SEQ, LEVEL, SRC_FPS, OUT_FPS, W, H = "MOT17-04-FRCNN", "low", 30, 15, 960, 540
OUT = ROOT / "demo/static/video"


def make_video() -> Path:
    """H.264 MP4(브라우저 공통 재생). OpenCV VP8은 품질 대비 용량이 커서(68MB) imageio-ffmpeg의 ffmpeg로 인코딩."""
    import subprocess
    import imageio_ffmpeg
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "sample_low.mp4"
    files = sorted((ROOT / f"data/vast_mot/{LEVEL}/{SEQ}/img1").glob("*.jpg"))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(OUT_FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in files[::SRC_FPS // OUT_FPS]:
        proc.stdin.write(cv2.resize(cv2.imread(str(f)), (W, H), interpolation=cv2.INTER_AREA).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg 인코딩 실패")
    return dst


def main():
    dst = OUT / "sample_low.mp4"
    if "--remake" in sys.argv or not dst.exists():
        dst = make_video()
    print(dst.name, round(dst.stat().st_size / 1e6, 1), "MB")
    pipe, gt = Pipe(), load_gt(SEQ)
    sx = W / 1920
    cap = cv2.VideoCapture(str(dst))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, det_gt = [], []
    for kk in range(int(n / OUT_FPS * 2)):  # 초당 2장, 브라우저와 같이 t = kk/2초
        k = min(n - 1, int(kk / 2 * OUT_FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        if not ok:
            break
        b, _ = pipe.detect(img, min_h=0.037 * H)
        src = k * (SRC_FPS // OUT_FPS) + 1  # 원본 프레임 번호
        frames.append((src, b, pipe.embed(img, b)))
        g = gt.get(src, [])
        if len(b) and g:
            m = iou(b, np.stack([x[1] * sx for x in g]))
            det_gt.append([g[j][0] if m[i, j] >= 0.5 else -1 for i, j in enumerate(m.argmax(1))])
        else:
            det_gt.append([-1] * len(b))
    tracks = build_tracks(frames)
    # 브라우저(video.js)처럼 트랙마다 짝수 번째 등장에서 최대 6장만 특징을 쓴다(점수용). 정답 판정은 트랙 전체로.
    sub = [[e for i, e in enumerate(tr) if i % 2 == 0][:6] for tr in tracks]
    seen = defaultdict(int)
    for ids in det_gt:
        for i in ids:
            if i > 0:
                seen[i] += 1
    rows = []
    for pid in sorted(i for i, c in seen.items() if c >= 4):
        opts = [(f, b, v) for f, lst in gt.items() for i, b, v in lst if i == pid and v >= 0.8 and b[3] - b[1] >= 120]
        if not opts:
            continue
        f, b, _ = opts[len(opts) // 2]
        day = cv2.imread(str(RAW / SEQ / "img1" / f"{f:06d}.jpg"))
        b = b.copy(); b[[0, 2]] = b[[0, 2]].clip(0, 1920); b[[1, 3]] = b[[1, 3]].clip(0, 1080)
        q = pipe.embed(day, b[None])[0]
        scored = rank_tracks(frames, sub, q, skip=lambda fr, f=f: abs(fr - f) <= 2 * SRC_FPS)
        full = {id(s): t for s, t in zip(sub, tracks)}
        ranked = [(sc, full[id(s)], sims) for sc, s, sims in scored]

        def label(tr):
            ids = [det_gt[fi][j] for fi, j in tr]
            top = max(set(ids), key=ids.count)
            return top if ids.count(top) * 2 > len(ids) else -1

        labels = [label(tr) for _, tr, _ in ranked]
        first = next((k for k, l in enumerate(labels) if l == pid), None)
        rows.append({"pid": pid, "frame": f, "box": b, "first": first, "score": ranked[0][0]})
    fr = [r["first"] for r in rows]
    stats = {"n_query": len(rows), "rank1": float(np.mean([x == 0 for x in fr])),
             "rank5": float(np.mean([x is not None and x < 5 for x in fr])), "n_frames": len(frames), "n_tracks": len(tracks)}
    print(stats)
    # 예제 4명: 1위 정답 중 상자 높이 큰 순, 서로 다른 사람
    good = sorted((r for r in rows if r["first"] == 0), key=lambda r: -(r["box"][3] - r["box"][1]))[:4]
    people = []
    for k, r in enumerate(good, 1):
        day = cv2.imread(str(RAW / SEQ / "img1" / f"{r['frame']:06d}.jpg"))
        x1, y1, x2, y2 = r["box"].astype(int)
        cv2.imwrite(str(OUT / f"q{k}.jpg"), day[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        people.append({"file": f"q{k}.jpg", "pid": int(r["pid"]), "time": round((r["frame"] - 1) / SRC_FPS, 1)})
    meta = {"video": "sample_low.mp4", "label": "저조도 거리 (MOT17-04, 합성 저조도)",
            "duration": round(n / OUT_FPS, 1), "people": people,
            "eval": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()},
            "note": "쿼리는 같은 장면 주간 원본에서 자른 사진, 쿼리와 같은 순간 ±2초는 제외하고 측정"}
    (OUT / "sample.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(ROOT / "results/video_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "query_frame", "first_correct_rank", "top_score"])
        for r in rows:
            w.writerow([r["pid"], r["frame"], r["first"], round(r["score"], 4)])
    print(people)


if __name__ == "__main__":
    main()
