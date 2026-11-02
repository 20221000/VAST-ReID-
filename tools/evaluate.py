#!/usr/bin/env python3
"""조도별 ReID 평가 (사전학습 모델 추론만, 학습 없음).

    python tools/evaluate.py --data data/vast --out results/baseline.csv
    python tools/evaluate.py --limit-query 300            # 샘플로 시간 측정
    python tools/evaluate.py --enhance none gamma clahe zerodce --out results/enhance.csv
    python tools/evaluate.py --enhance gamma zerodce --gate 0.25 --denoise nlm --out results/enhance_v2.csv --append

프로토콜
  same  : query·gallery 모두 같은 조도 L          (L = normal, low, extreme)
  cross : query는 조도 L, gallery는 normal       (야간 query → 주간 gallery, 실제 CCTV 상황)
거리: torchreid Model Zoo와 동일하게 정규화하지 않은 특징의 유클리드 거리.
개선 전처리(--enhance)는 query·gallery 모든 이미지에 동일한 규칙으로 적용:
  --gate τ    : 입력 평균 밝기(BT.709 luma) < τ 인 이미지에만 denoise·enhance 적용 (기본: 전부 적용)
  --denoise   : 밝기 개선 전에 노이즈 제거 (none | nlm)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.data import list_split  # noqa: E402
from vastreid.enhance import ALL as ENHANCERS, CPU_ENHANCERS, DENOISERS, GPU_ENHANCERS, mean_luma  # noqa: E402
from vastreid.metrics import evaluate  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def gpu_temp() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                       text=True).strip()
    except Exception:
        return "?"


def load_model(name: str, weights: Path, device: str):
    import torchreid
    model = torchreid.models.build_model(name=name, num_classes=1, pretrained=False)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k.removeprefix("module."): v for k, v in sd.items() if not k.removeprefix("module.").startswith("classifier.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    if missing or unexpected:
        raise RuntimeError(f"가중치 불일치 missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval().to(device)


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths, size=(128, 256), cpu_enhance=None, denoise=None, gate=None):
        self.paths, self.size, self.cpu_enhance = paths, size, cpu_enhance
        self.denoise, self.gate = denoise, gate

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        apply = self.gate is None or mean_luma(img) < self.gate
        if apply and self.denoise is not None:
            img = self.denoise(img)
        if apply and self.cpu_enhance is not None:
            img = self.cpu_enhance(img)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()), apply  # RGB uint8 CHW, 적용 여부


@torch.no_grad()
def extract(model, paths, device, batch=256, workers=8, cpu_enhance=None, gpu_enhance=None,
            denoise=None, gate=None) -> tuple[torch.Tensor, float]:
    """특징과 개선이 적용된 이미지 비율을 반환."""
    dl = torch.utils.data.DataLoader(ImgDS(paths, cpu_enhance=cpu_enhance, denoise=denoise, gate=gate), batch_size=batch,
                                     num_workers=workers, pin_memory=True)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out, n_applied = [], 0
    for x, applied in dl:
        x = x.to(device, non_blocking=True).float() / 255.0
        applied = applied.to(device)
        n_applied += int(applied.sum())
        if gpu_enhance is not None and applied.any():
            # Zero-DCE는 전 해상도 32ch 활성값을 쓰므로 작은 묶음으로 나눠 VRAM 초과(WSL sysmem fallback) 방지
            sel = x[applied]
            x[applied] = torch.cat([gpu_enhance(c) for c in sel.split(32)])
        out.append(model((x - mean) / std).float())
    return torch.cat(out), n_applied / max(1, len(paths))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/vast")
    ap.add_argument("--model", default="osnet_x1_0")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/osnet_x1_0_market1501.pth")
    ap.add_argument("--levels", nargs="+", default=["normal", "low", "extreme"])
    ap.add_argument("--protocols", nargs="+", default=["same", "cross"], choices=["same", "cross"])
    ap.add_argument("--enhance", nargs="+", default=["none"], choices=ENHANCERS)
    ap.add_argument("--gate", type=float, default=None, help="평균 밝기 < τ 인 이미지에만 개선 적용")
    ap.add_argument("--denoise", default="none", choices=list(DENOISERS))
    ap.add_argument("--append", action="store_true", help="기존 CSV에 행 추가")
    ap.add_argument("--limit-query", type=int, default=None, help="샘플 테스트용 query 수 제한")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "results/baseline.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} model={args.model} GPU {gpu_temp()}°C")
    model = load_model(args.model, args.weights, device)

    # 필요한 (level, split) 특징만 한 번씩 추출해 재사용
    need = set()
    for lv in args.levels:
        if "same" in args.protocols:
            need |= {(lv, "query"), (lv, "gallery")}
        if "cross" in args.protocols and lv != "normal":
            need |= {(lv, "query"), ("normal", "gallery")}
    rows = []
    t0 = time.time()
    for enh in args.enhance:
        cpu_e = CPU_ENHANCERS.get(enh)
        gpu_e = GPU_ENHANCERS[enh](device=device) if enh in GPU_ENHANCERS else None
        items, feats, rate = {}, {}, {}
        for lv, sp in sorted(need):
            it = list_split(args.data, lv, sp)
            if sp == "query" and args.limit_query:
                it = it[:args.limit_query]
            t = time.time()
            feats[(lv, sp)], rate[(lv, sp)] = extract(model, [x.path for x in it], device, args.batch, args.workers,
                                                      cpu_e, gpu_e, DENOISERS[args.denoise], args.gate)
            items[(lv, sp)] = it
            print(f"  [{enh}] feat {lv}/{sp}: {len(it)} imgs {time.time() - t:.1f}s  "
                  f"applied {rate[(lv, sp)]*100:.0f}%  GPU {gpu_temp()}°C", flush=True)

        for lv in args.levels:
            pairs = []
            if "same" in args.protocols:
                pairs.append(("same", lv, lv))
            if "cross" in args.protocols and lv != "normal":
                pairs.append(("cross", lv, "normal"))
            for proto, ql, gl in pairs:
                qf, gf = feats[(ql, "query")], feats[(gl, "gallery")]
                dist = torch.cdist(qf, gf).cpu().numpy()
                qi, gi = items[(ql, "query")], items[(gl, "gallery")]
                r = evaluate(dist, [x.pid for x in qi], [x.pid for x in gi],
                             [x.camid for x in qi], [x.camid for x in gi])
                rows.append({"model": args.model, "enhance": enh, "denoise": args.denoise,
                             "gate": "" if args.gate is None else args.gate, "protocol": proto, "query_level": ql,
                             "gallery_level": gl,
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                             "num_query": len(qi), "num_gallery": len(gi),
                             "q_applied": round(rate[(ql, "query")], 3), "g_applied": round(rate[(gl, "gallery")], 3)})
                print(f"  [{enh}] {proto:5s} q={ql:7s} g={gl:7s}  R1 {r['rank1']*100:5.1f}  "
                      f"R5 {r['rank5']*100:5.1f}  mAP {r['mAP']*100:5.1f}  mINP {r['mINP']*100:5.1f}")
    print(f"[done] {time.time() - t0:.1f}s  GPU {gpu_temp()}°C")

    if args.limit_query:
        args.out = args.out.with_name(args.out.stem + "_sample.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.append and args.out.exists())
    with open(args.out, "a" if args.append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
