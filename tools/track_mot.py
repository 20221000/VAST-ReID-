#!/usr/bin/env python3
"""저조도 합성 MOT17에서 트래커 비교: ByteTrack vs BoT-SORT+OSNet vs BoT-SORT+OSNet(AdaBN).

    python tools/track_mot.py --limit-frames 50        # 샘플로 시간 측정 (평가는 하지 않음)
    python tools/track_mot.py                          # 전체 → outputs/mot_results/, results/tracking.csv

- 검출: YOLO(사람 클래스)로 조도·시퀀스별 1회 검출 후 캐시 → 모든 트래커가 같은 검출을 사용 (트래커 차이만 비교)
- 외형 특징: 우리 OSNet(Market-1501 사전학습)으로 직접 계산해 BoT-SORT에 전달 (boxmot 기본 ReID 대신)
  AdaBN: tools/adabn.py 와 같은 방식으로, 해당 조도의 Market train 이미지로 BN 통계만 재추정한 모델 (라벨 미사용)
- 평가: TrackEval (HOTA, IDF1, MOTA), MOT17 train 정답(gt). 정답은 조도와 무관하게 동일.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from vastreid.reid import MEAN, STD, gpu_temp, load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRACKERS = ["bytetrack", "botsort-osnet", "botsort-osnet-adabn"]


# ------------------------------------------------------------------ 검출
def detect(data: Path, level: str, seq: str, model_name: str, imgsz: int, conf: float, limit: int | None,
           yolo=None, cache_name: str | None = None):
    cp = ROOT / f"outputs/mot_dets/{cache_name or Path(model_name).stem}/{level}/{seq}.npy"
    frames = sorted((data / level / seq / "img1").glob("*.jpg"))[:limit]
    if cp.exists() and limit is None:
        return np.load(cp), frames
    from ultralytics import YOLO
    yolo = yolo or YOLO(model_name)
    rows, i = [], 0
    # 경로 목록을 한 번에 넘기면 전체가 하나의 batch가 되어 VRAM 초과 → 16장씩 나눠 검출
    for s in range(0, len(frames), 16):
        for r in yolo.predict([str(f) for f in frames[s:s + 16]], classes=[0], conf=conf, imgsz=imgsz,
                              device=0 if DEV == "cuda" else "cpu", verbose=False):
            i += 1
            b = r.boxes
            for (x1, y1, x2, y2), c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                rows.append([i, x1, y1, x2, y2, c])
    dets = np.array(rows, np.float32).reshape(-1, 6)
    if limit is None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.save(cp, dets)
    return dets, frames


@torch.no_grad()
def yolo_adabn(model_name: str, frame_dir: Path, imgsz: int, step: int = 2):
    """YOLO의 BatchNorm 통계만 frame_dir의 프레임(라벨 미사용)으로 재추정. 예측 시 Conv-BN이 합쳐지기 전에 수행."""
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    yolo = YOLO(model_name)
    net = yolo.model.to(DEV).float().eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise RuntimeError("BN이 없음 (이미 fuse된 모델)")
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    lb = LetterBox(new_shape=imgsz, auto=True, stride=32)  # ultralytics 예측과 같은 전처리
    frames = sorted(frame_dir.glob("*.jpg"))[::step]
    for s in range(0, len(frames), 8):
        x = np.stack([lb(image=cv2.imread(str(f)))[..., ::-1].transpose(2, 0, 1) for f in frames[s:s + 8]])
        net(torch.from_numpy(np.ascontiguousarray(x)).to(DEV).float() / 255.0)
    net.eval()
    return yolo, len(bns), len(frames)


# ------------------------------------------------------------------ 외형 특징
@torch.no_grad()
def embed(model, img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 512), np.float32)
    h, w = img.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, max(x1 + 1, int(x2))), min(h, max(y1 + 1, int(y2)))
        c = cv2.resize(img[y1:y2, x1:x2], (128, 256), interpolation=cv2.INTER_LINEAR)
        crops.append((c[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD)
    x = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2)).to(DEV)
    f = model(x).float()
    return torch.nn.functional.normalize(f, dim=1).cpu().numpy()


def reid_models(levels):
    from adabn import adapt_bn
    base = load_model("osnet_x1_0", ROOT / "weights/osnet_x1_0_market1501.pth", DEV)
    adapted = {"normal": base}
    for lv in levels:
        if lv != "normal":
            adapted[lv], _, _ = adapt_bn(base, lv)
    return base, adapted


def make_tracker(name: str):
    from boxmot import BotSort, ByteTrack
    if name == "bytetrack":
        return ByteTrack()
    return BotSort()


# ------------------------------------------------------------------ 트래킹
def run_tracker(name, dets, frames, reid_model, out_txt: Path):
    """boxmot 26: 외부 외형 특징은 Detections(embeddings=...)로 전달하고, 결과는 Tracks 구조로 받음."""
    from boxmot.structures import Boxes, Detections
    trk = make_tracker(name)
    lines = []
    for i, f in enumerate(frames, 1):
        img = cv2.imread(str(f))
        d = dets[dets[:, 0] == i]
        d = d[(d[:, 3] > d[:, 1]) & (d[:, 4] > d[:, 2])]
        boxes = torch.from_numpy(np.ascontiguousarray(d[:, 1:5], np.float32))
        emb = None
        if name != "bytetrack":
            emb = torch.from_numpy(embed(reid_model, img, d[:, 1:5])).float()
        det = Detections(geometry=Boxes(boxes), scores=torch.from_numpy(d[:, 5].astype(np.float32)),
                         class_ids=torch.zeros(len(d), dtype=torch.int64), embeddings=emb)
        out = trk.update(det, img)
        xyxy = out.geometry.values.numpy()
        for (x1, y1, x2, y2), tid, conf in zip(xyxy, out.track_ids.numpy(), out.scores.numpy()):
            lines.append(f"{i},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


# ------------------------------------------------------------------ 평가
def trackeval_eval(gt_root: Path, trk_root: Path, trackers: list[str], seqs: list[str]) -> dict:
    for a, t in (("float", float), ("int", int), ("bool", bool)):  # 오래된 numpy 별칭 (TrackEval 호환)
        if not hasattr(np, a):
            setattr(np, a, t)
    import trackeval
    ev = trackeval.Evaluator({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                              "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                              "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True})
    ds = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trk_root), "TRACKERS_TO_EVAL": trackers,
        "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True, "TRACKER_SUB_FOLDER": "",
        "OUTPUT_FOLDER": str(trk_root / "_eval"), "SEQ_INFO": {s: None for s in seqs},
        "CLASSES_TO_EVAL": ["pedestrian"], "PRINT_CONFIG": False,
    })
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = ev.evaluate([ds], metrics)
    out = {}
    for t in trackers:
        r = res["MotChallenge2DBox"][t]["COMBINED_SEQ"]["pedestrian"]
        out[t] = {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "DetA": float(np.mean(r["HOTA"]["DetA"])),
                  "AssA": float(np.mean(r["HOTA"]["AssA"])), "IDF1": float(r["Identity"]["IDF1"]),
                  "MOTA": float(r["CLEAR"]["MOTA"]), "IDSW": int(r["CLEAR"]["IDSW"])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/vast_mot")
    ap.add_argument("--levels", nargs="+", default=["normal", "low", "extreme"])
    ap.add_argument("--seqs", nargs="+", default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"])
    ap.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS)
    ap.add_argument("--det-model", default=str(ROOT / "weights/yolo11m.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--limit-frames", type=int, default=None, help="샘플 테스트 (평가 생략)")
    ap.add_argument("--eval-only", action="store_true", help="저장된 트래킹 결과로 TrackEval 평가만 실행")
    ap.add_argument("--no-eval", action="store_true", help="트래킹만 하고 평가 생략")
    ap.add_argument("--out", type=Path, default=None, help="기본: results/tracking.csv (--det-adabn: tracking_yolo_adabn.csv)")
    ap.add_argument("--det-adabn", action="store_true",
                    help="YOLO BN 통계를 조도별로 재추정 (평가에 쓰지 않는 --adapt-seq 프레임, 라벨 미사용)")
    ap.add_argument("--adapt-data", type=Path, default=ROOT / "data/vast_mot_adapt")
    ap.add_argument("--adapt-seq", default="MOT17-11-FRCNN")
    args = ap.parse_args()

    t0 = time.time()
    args.out = args.out or ROOT / ("results/tracking_yolo_adabn.csv" if args.det_adabn else "results/tracking.csv")
    assert args.adapt_seq not in args.seqs, "적응용 시퀀스는 평가 시퀀스와 달라야 함"
    need_reid = any("botsort" in t for t in args.trackers) and not args.eval_only
    base, adapted = reid_models(args.levels) if need_reid else (None, {})
    rows = []
    for lv in args.levels:
        trk_root = ROOT / f"outputs/mot_results{'_yoloadabn' if args.det_adabn else ''}/{lv}"
        yolo, cache_name = None, None
        if args.det_adabn and not args.eval_only:
            t = time.time()
            yolo, n_bn, n_fr = yolo_adabn(args.det_model, args.adapt_data / lv / args.adapt_seq / "img1", args.imgsz)
            cache_name = f"{Path(args.det_model).stem}-adabn-{lv}"
            print(f"  [yolo-adabn] {lv}: BN {n_bn}개 재추정 ({args.adapt_seq} {n_fr} frames, {time.time() - t:.0f}s)",
                  flush=True)
        for seq in ([] if args.eval_only else args.seqs):
            t = time.time()
            dets, frames = detect(args.data, lv, seq, args.det_model, args.imgsz, args.conf, args.limit_frames,
                                  yolo, cache_name)
            print(f"  [det] {lv}/{seq}: {len(frames)} frames, {len(dets)} dets ({len(dets) / max(len(frames), 1):.1f}/frame) "
                  f"{time.time() - t:.0f}s  GPU {gpu_temp()}°C", flush=True)
            for name in args.trackers:
                t = time.time()
                model = {"botsort-osnet": base, "botsort-osnet-adabn": adapted.get(lv)}.get(name)
                n = run_tracker(name, dets, frames, model, trk_root / name / f"{seq}.txt")
                print(f"    [{name}] {n} boxes {time.time() - t:.0f}s", flush=True)
        if args.limit_frames or args.no_eval:
            continue
        res = trackeval_eval(args.data / lv, trk_root, args.trackers, args.seqs)
        for name, m in res.items():
            rows.append({"level": lv, "detector": "yolo11m+adabn" if args.det_adabn else "yolo11m", "tracker": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                          for k, v in m.items()}})
            print(f"  [{lv:7s}] {name:22s} HOTA {m['HOTA']*100:5.1f}  IDF1 {m['IDF1']*100:5.1f}  "
                  f"MOTA {m['MOTA']*100:5.1f}  AssA {m['AssA']*100:5.1f}  IDSW {m['IDSW']}", flush=True)
    print(f"[done] {time.time() - t0:.0f}s  GPU {gpu_temp()}°C")
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
