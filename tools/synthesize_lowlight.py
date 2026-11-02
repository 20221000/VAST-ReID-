#!/usr/bin/env python3
"""저조도 합성 스크립트.

ReID 모드 (Market-1501):
    python tools/synthesize_lowlight.py --src data/market1501 --out data/vast
    -> data/vast/{normal,low,extreme}/{train,query,gallery}/*.jpg
       data/vast/synthesis_log.csv (이미지별 합성 파라미터 + mean_luminance)

MOT 모드 (MOT17 시퀀스, 시퀀스 단위로 노출 고정):
    python tools/synthesize_lowlight.py --mode mot --src data/mot17/train --out data/vast_mot \
        --sequences MOT17-02-FRCNN MOT17-04-FRCNN
    -> data/vast_mot/{normal,low,extreme}/<seq>/{img1/, gt/, seqinfo.ini ...}

처음에는 --limit 으로 작은 샘플을 돌려 처리 속도와 전체 예상 시간을 확인할 것.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

LEVELS_ORDER = ["normal", "low", "extreme"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- 색공간 변환
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def mean_luma(bgr_u8: np.ndarray) -> float:
    """BT.709 luma(Y', 감마 적용된 sRGB 기준)의 평균, 0~1."""
    b, g, r = (bgr_u8[..., i].astype(np.float32) / 255.0 for i in range(3))
    return float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean())


# ---------------------------------------------------------------- 파라미터
@dataclass
class Params:
    exposure_scale: float
    shot_noise: float
    read_noise: float
    blur_sigma: float
    color_gain_r: float
    color_gain_b: float
    jpeg_quality: int


def make_rng(seed: int, level: str, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(level.encode()), zlib.crc32(key.encode())])


def sample_params(cfg_level: dict, rng: np.random.Generator) -> Params:
    u = lambda k: float(rng.uniform(*cfg_level[k]))
    return Params(
        exposure_scale=u("exposure"),
        shot_noise=u("shot_noise"),
        read_noise=u("read_noise"),
        blur_sigma=u("blur_sigma"),
        color_gain_r=u("color_gain_r"),
        color_gain_b=u("color_gain_b"),
        jpeg_quality=int(round(rng.uniform(*cfg_level["jpeg_quality"]))),
    )


def degrade(bgr_u8: np.ndarray, p: Params, noise_rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """sRGB uint8 BGR -> 저조도 sRGB uint8 BGR. (출력, 유효 노이즈 sigma) 반환."""
    lin = srgb_to_linear(bgr_u8.astype(np.float32) / 255.0)
    if p.blur_sigma > 0.05:
        lin = cv2.GaussianBlur(lin, (0, 0), p.blur_sigma)
    lin = lin * p.exposure_scale
    lin[..., 2] *= p.color_gain_r  # BGR 순서
    lin[..., 0] *= p.color_gain_b
    lin = np.clip(lin, 0.0, 1.0)
    # Poisson-Gaussian 센서 노이즈 (linear 공간)
    noisy = noise_rng.poisson(lin / p.shot_noise).astype(np.float32) * p.shot_noise
    noisy += noise_rng.normal(0.0, p.read_noise, size=lin.shape).astype(np.float32)
    # 평균 신호 수준에서의 유효 노이즈 표준편차 (메타데이터용)
    noise_sigma = float(np.sqrt(p.shot_noise * lin.mean() + p.read_noise ** 2))
    out = np.round(linear_to_srgb(noisy) * 255.0).astype(np.uint8)
    return out, noise_sigma


# ---------------------------------------------------------------- 작업 단위
@dataclass
class Job:
    src: str           # 원본 이미지 절대경로
    dst_rel: str       # 출력 상대경로 (level 폴더 기준)
    split: str
    param_key: str     # 파라미터 샘플링 키 (reid: 이미지 경로, mot: 시퀀스명)
    noise_key: str     # 노이즈 샘플링 키 (항상 이미지 경로)


def process_one(args) -> list[dict]:
    job, out_root, levels, cfg, seed = args
    img = cv2.imread(job.src, cv2.IMREAD_COLOR)
    if img is None:
        return [{"level": "ERROR", "split": job.split, "file": job.dst_rel}]
    rows = []
    for level in levels:
        dst = Path(out_root) / level / job.dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        row = {"level": level, "split": job.split, "file": job.dst_rel}
        if level == "normal":
            shutil.copyfile(job.src, dst)
            row.update(exposure_scale=1.0, noise_sigma=0.0, mean_luminance=round(mean_luma(img), 5))
        else:
            p = sample_params(cfg["levels"][level], make_rng(seed, level, job.param_key))
            out, sigma = degrade(img, p, make_rng(seed, level + "/noise", job.noise_key))
            cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
            row.update({k: round(v, 5) if isinstance(v, float) else v for k, v in asdict(p).items()})
            row.update(noise_sigma=round(sigma, 5), mean_luminance=round(mean_luma(out), 5))
        rows.append(row)
    return rows


def _init_worker():
    cv2.setNumThreads(1)


# ---------------------------------------------------------------- 입력 수집
def collect_reid(src: Path, cfg: dict, limit: int | None) -> list[Job]:
    jobs = []
    for sub, split in cfg["market1501_splits"].items():
        d = src / sub
        if not d.is_dir():
            sys.exit(f"[error] {d} 가 없습니다. --src 는 bounding_box_train/query/bounding_box_test 를 포함한 폴더여야 합니다.")
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
        if limit:
            files = files[:limit]
        for f in files:
            rel = f"{split}/{f.name}"
            jobs.append(Job(str(f), rel, split, rel, rel))
    return jobs


def collect_mot(src: Path, sequences: list[str] | None, limit: int | None) -> tuple[list[Job], list[Path]]:
    seq_dirs = sorted(p.parent for p in src.glob("*/img1"))
    if sequences:
        seq_dirs = [s for s in seq_dirs if s.name in set(sequences)]
        missing = set(sequences) - {s.name for s in seq_dirs}
        if missing:
            sys.exit(f"[error] 시퀀스를 찾을 수 없음: {sorted(missing)}")
    if not seq_dirs:
        sys.exit(f"[error] {src}/*/img1 이 없습니다.")
    jobs = []
    for seq in seq_dirs:
        frames = sorted(f for f in (seq / "img1").iterdir() if f.suffix.lower() in IMG_EXTS)
        if limit:
            frames = frames[:limit]
        for f in frames:
            rel = f"{seq.name}/img1/{f.name}"
            jobs.append(Job(str(f), rel, seq.name, seq.name, rel))
    return jobs, seq_dirs


def copy_mot_aux(seq_dirs: list[Path], out: Path, levels: list[str]):
    """gt/, det/, seqinfo.ini 등 이미지 외 파일을 각 레벨에 복사 (TrackEval 용)."""
    for seq in seq_dirs:
        for item in seq.iterdir():
            if item.name == "img1":
                continue
            for level in levels:
                dst = out / level / seq.name / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(item, dst)


# ---------------------------------------------------------------- 샘플 그리드
def save_grid(out: Path, jobs: list[Job], levels: list[str], path: Path, n: int = 8):
    """행=이미지, 열=조도 등급. 발표 자료·육안 확인용."""
    pick = [j for j in jobs if j.split == "query"] or jobs
    pick = pick[:: max(1, len(pick) // n)][:n]
    rows = []
    for j in pick:
        tiles = [cv2.resize(cv2.imread(str(out / lv / j.dst_rel)), (128, 256)) for lv in levels]
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)
    header = np.full((28, grid.shape[1], 3), 255, np.uint8)
    for i, lv in enumerate(levels):
        cv2.putText(header, lv, (i * 128 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([header, grid]))
    print(f"[grid] {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/lowlight.yaml")
    ap.add_argument("--mode", choices=["reid", "mot"], default="reid")
    ap.add_argument("--seed", type=int, default=None, help="config의 seed 덮어쓰기")
    ap.add_argument("--levels", nargs="+", default=LEVELS_ORDER, choices=LEVELS_ORDER)
    ap.add_argument("--sequences", nargs="+", default=None, help="[mot] 처리할 시퀀스 이름")
    ap.add_argument("--limit", type=int, default=None, help="split/시퀀스당 최대 이미지 수 (샘플 테스트용)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--grid", type=Path, default=None, help="샘플 그리드 저장 경로 (예: docs/figures/lowlight_grid.jpg)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else cfg["seed"]
    levels = [lv for lv in LEVELS_ORDER if lv in args.levels]

    if args.mode == "reid":
        jobs = collect_reid(args.src, cfg, args.limit)
    else:
        jobs, seq_dirs = collect_mot(args.src, args.sequences, args.limit)
        copy_mot_aux(seq_dirs, args.out, levels)
    print(f"[info] mode={args.mode} images={len(jobs)} levels={levels} seed={seed} workers={args.workers}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows, t0, done = [], time.time(), 0
    tasks = ((j, str(args.out), levels, cfg, seed) for j in jobs)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        for r in ex.map(process_one, tasks, chunksize=32):
            rows.extend(r)
            done += 1
            if done % 500 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {done / el:.1f} img/s  elapsed {el:.0f}s", flush=True)
    elapsed = time.time() - t0

    errors = [r for r in rows if r["level"] == "ERROR"]
    rows = [r for r in rows if r["level"] != "ERROR"]
    keys = ["level", "split", "file", "exposure_scale", "shot_noise", "read_noise", "noise_sigma",
            "blur_sigma", "color_gain_r", "color_gain_b", "jpeg_quality", "mean_luminance"]
    log_path = args.out / "synthesis_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"[done] {len(jobs)} images x {len(levels)} levels in {elapsed:.1f}s "
          f"({len(jobs) / max(elapsed, 1e-9):.1f} img/s) -> {args.out}")
    print(f"[log] {log_path}")
    for lv in levels:
        lum = [r["mean_luminance"] for r in rows if r["level"] == lv]
        if lum:
            print(f"  {lv:8s} mean_luminance: mean={np.mean(lum):.3f}  p5={np.percentile(lum, 5):.3f}  p95={np.percentile(lum, 95):.3f}")
    if errors:
        print(f"[warn] 읽기 실패 {len(errors)}개: {[e['file'] for e in errors[:5]]}")
    if args.grid:
        save_grid(args.out, jobs, levels, args.grid)


if __name__ == "__main__":
    main()
