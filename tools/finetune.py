#!/usr/bin/env python3
"""저조도 미세조정 (Protocol B): Market-1501 train 이미지를 normal·low·extreme 세 조도로 섞어 OSNet을 추가 학습.

    python tools/finetune.py --epochs 2 --out weights/osnet_ft_sample.pth      # 샘플: 시간·온도 측정
    python tools/finetune.py                                                     # 본 학습 (12 에폭 고정)
    python tools/finetune.py --extract-only                                      # 특징 추출·평가만

- 학습 데이터: train split만 (751명 × 3조도 = 38,808장). query·gallery는 학습에 전혀 사용하지 않음.
- 에폭 수는 미리 고정 (테스트 성능으로 고르지 않음). 마지막 에폭 가중치를 그대로 사용.
- 출발점: torchreid Model Zoo의 Market-1501 학습 가중치 (분류기 포함). 손실: 라벨 스무딩 분류 손실(공식 학습과 동일 계열).
- 학습 후 특징 캐시: outputs/features/osnet_x1_0-ft__{none|adabn-L}__{level}__{split}.pt
  (adabn: 미세조정 모델에 다시 조도별 BN 통계 재추정)
- 평가: results/finetune.csv (same: q·g 같은 조도, cross: 저조도 쿼리 → normal gallery, 코사인)
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vastreid.data import list_split  # noqa: E402
from vastreid.metrics import evaluate_fast  # noqa: E402
from vastreid.reid import MEAN, STD, extract, gpu_temp  # noqa: E402

DATA = ROOT / "data/vast"
LEVELS = ["normal", "low", "extreme"]
DEV = "cuda"
PREFIX = "osnet_x1_0-ft"


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, items, labels):
        self.items, self.labels = items, labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.resize(cv2.imread(str(it.path)), (128, 256), interpolation=cv2.INTER_LINEAR)
        if random.random() < 0.5:
            img = img[:, ::-1]
        x = (img[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        if random.random() < 0.5:  # random erasing (ReID 표준 증강)
            h, w = x.shape[1:]
            eh, ew = int(h * random.uniform(0.2, 0.5)), int(w * random.uniform(0.2, 0.5))
            y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
            x[:, y0:y0 + eh, x0:x0 + ew] = torch.randn(3, eh, ew)
        return x, self.labels[i]


def build(num_classes=751):
    import torchreid
    m = torchreid.models.build_model("osnet_x1_0", num_classes=num_classes, loss="softmax", pretrained=False)
    sd = torch.load(ROOT / "weights/osnet_x1_0_market1501.pth", map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in sd.get("state_dict", sd).items()}
    m.load_state_dict(sd, strict=True)
    return m.to(DEV)


@torch.no_grad()
def classifier_acc(model, dl, n_batches=20):
    model.train()  # 분류 로짓을 얻기 위해 train 모드 (BN은 아래에서 eval로)
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()
    c = n = 0
    for b, (x, y) in enumerate(dl):
        if b >= n_batches:
            break
        out = model(x.to(DEV))
        c += int((out.argmax(1).cpu() == y).sum())
        n += len(y)
    model.eval()
    return c / max(n, 1)


def train(args):
    items, pids = [], []
    for lv in LEVELS:
        it = list_split(DATA, lv, "train")
        items += it
        pids += [x.pid for x in it]
    uniq = sorted(set(pids))
    label = {p: i for i, p in enumerate(uniq)}  # torchreid과 같은 오름차순 relabel
    labels = [label[p] for p in pids]
    ds = TrainDS(items, labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True,
                                     drop_last=True, persistent_workers=True)
    model = build(len(uniq))
    acc0 = classifier_acc(model, dl)
    print(f"[ft] {len(ds)} imgs, {len(uniq)} ids, 시작 시점 분류 정확도(분류기 매핑 확인용) {acc0*100:.1f}%", flush=True)
    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
                            weight_decay=5e-4)
    steps = args.epochs * len(dl)
    warm = len(dl) // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = tc = tn = 0
        te = time.time()
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach()) * len(y)
            tc += int((out.argmax(1) == y).sum())
            tn += len(y)
        print(f"  epoch {ep:2d}/{args.epochs}  loss {tl/tn:.3f}  train acc {tc/tn*100:5.1f}%  "
              f"{time.time()-te:.0f}s  GPU {gpu_temp()}°C", flush=True)
    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out)
    print(f"[ft] saved {args.out}  ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def adapt_bn(base, level):
    m = copy.deepcopy(base)
    bns = [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    m.eval()
    for bn in bns:
        bn.train()
    extract(m, [x.path for x in list_split(DATA, level, "train")], DEV)
    return m.eval()


def extract_all(model):
    cache = ROOT / "outputs/features"
    models = {"none": model}
    for lv in ["low", "extreme"]:
        models[f"adabn-{lv}"] = adapt_bn(model, lv)
    for tag, m in models.items():
        for lv in LEVELS:
            if tag.startswith("adabn") and not tag.endswith(lv):
                continue
            for sp in ["train", "query", "gallery"]:
                it = list_split(DATA, lv, sp)
                f, _ = extract(m, [x.path for x in it], DEV)
                torch.save({"n": len(it), "feats": f.cpu()}, cache / f"{PREFIX}__{tag}__{lv}__{sp}.pt")
        print(f"  [feat] {tag} done  GPU {gpu_temp()}°C", flush=True)


def evaluate_all(out_csv):
    def load(tag, lv, sp):
        d = torch.load(ROOT / f"outputs/features/{PREFIX}__{tag}__{lv}__{sp}.pt", weights_only=True)
        return torch.nn.functional.normalize(d["feats"].float(), dim=1).to(DEV)
    rows = []
    for feat in ["none", "adabn"]:
        for proto, ql, gl in [("same", "normal", "normal"), ("same", "low", "low"), ("cross", "low", "normal"),
                              ("same", "extreme", "extreme"), ("cross", "extreme", "normal")]:
            qt = f"adabn-{ql}" if feat == "adabn" and ql != "normal" else "none"
            gt = f"adabn-{gl}" if feat == "adabn" and gl != "normal" else "none"
            qi, gi = list_split(DATA, ql, "query"), list_split(DATA, gl, "gallery")
            r = evaluate_fast(-(load(qt, ql, "query") @ load(gt, gl, "gallery").T), [x.pid for x in qi],
                              [x.pid for x in gi], [x.camid for x in qi], [x.camid for x in gi])
            rows.append({"model": "osnet_x1_0+lowlight_ft", "feat": feat, "method": "cosine", "protocol": proto,
                         "query_level": ql, "gallery_level": gl,
                         **{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}})
            print(f"  ft {feat:5s} {proto:5s} {ql:7s}→{gl:7s} R1 {r['rank1']*100:5.1f}  mAP {r['mAP']*100:5.1f}  "
                  f"mINP {r['mINP']*100:5.1f}", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "weights/osnet_x1_0_vast_lowlight_ft.pth")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--no-extract", action="store_true", help="학습만 (샘플 실행용)")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.extract_only:
        model = build()
        model.load_state_dict(torch.load(args.out, map_location="cpu", weights_only=True))
        model.eval()
    else:
        model = train(args)
    if not args.no_extract:
        extract_all(model)
        evaluate_all(ROOT / "results/finetune.csv")


if __name__ == "__main__":
    main()
