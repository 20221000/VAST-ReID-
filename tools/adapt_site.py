#!/usr/bin/env python3
"""새 현장 적응: 라벨 없는 새 현장 사진으로 의사 라벨(pseudo label)을 만들어 추가 학습.

    python tools/adapt_site.py   # → weights/clip_vit_b16_vast_lowlight_ft_night_c1234_adapt58.pth, results/adapt_site.csv
    python tools/eval_night_cams.py   # 카메라 그룹별 평가 (적응 모델 포함)

시나리오: 카메라 1~4 장소에서만 실제 야간 라벨로 학습한 모델(`finetune.py --night-cams 1,2,3,4`)을
처음 보는 장소(카메라 5~8)에 설치했다고 가정. 그 장소에서 모은 **라벨 없는** 야간 사진(Night600 train 카메라 5~8, 6,813장)만으로 적응한다.
test(query·gallery) 사람과 겹치지 않으며, 사람 ID 라벨은 학습에 쓰지 않음(의사 라벨 품질 확인용 출력에만 사용).

방법 (클러스터링 기반 비지도 도메인 적응, SpCL·MMT 계열의 기본형):
  매 라운드 ① 현재 모델로 특징 추출 ② k-reciprocal Jaccard 거리 ③ DBSCAN으로 묶어 같은 묶음 = 같은 사람으로 가정(묶이지 않은 사진은 제외)
  ④ 묶음 중심으로 분류기를 새로 만들고 분류 + batch-hard triplet으로 몇 에폭 학습.
영상이 있으면 트래킹으로 같은 사람을 묶을 수 있지만 Night600은 사진만 있어 특징 유사도로 묶음.
하이퍼파라미터는 미리 고정(테스트 성능으로 고르지 않음): 6라운드 × 2에폭, lr 7e-6(분류기 10배), DBSCAN eps 0.6·min_samples 4, k1 30·k2 6.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from finetune import PKSampler, TrainDS, triplet_hard  # noqa: E402
from vastreid.data import list_night600  # noqa: E402
from vastreid.reid import extract, gpu_temp, load_model  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def jaccard(f: torch.Tensor, k1=30, k2=6) -> np.ndarray:
    """k-reciprocal Jaccard 거리 (Zhong et al. CVPR 2017, 같은 집합 안에서). f: L2 정규화 [N, D]."""
    n = len(f)
    sim = f @ f.T
    rank = sim.topk(k1 + 1, dim=1).indices.cpu().numpy()
    simc = sim.cpu().numpy()
    half = int(round(k1 / 2))
    nn1, nnh = rank[:, :k1 + 1], rank[:, :half + 1]
    V = np.zeros((n, n), np.float32)
    for i in range(n):
        fwd = nn1[i]
        rec = fwd[(nn1[fwd] == i).any(1)]
        exp = rec
        for c in rec:
            cf = nnh[c]
            cr = cf[(nnh[cf] == c).any(1)]
            if len(np.intersect1d(cr, rec)) > 2 / 3 * len(cr):
                exp = np.append(exp, cr)
        exp = np.unique(exp)
        w = np.exp(-(2 - 2 * simc[i, exp]))
        V[i, exp] = w / w.sum()
    if k2 > 1:
        V = V[rank[:, :k2]].mean(1)
    Vt = torch.from_numpy(V).to(DEV)
    out = torch.empty(n, n, device=DEV)
    for i in range(n):
        nz = Vt[i].nonzero().squeeze(1)
        mn = torch.minimum(Vt[i, nz][None], Vt[:, nz]).sum(1)  # 각 행 합이 1 → max 합 = 2 - min 합
        out[i] = 1 - mn / (2 - mn)
    return out.clamp_(min=0).cpu().numpy()


def cluster(model, items, args):
    f, _ = extract(model, [it.path for it in items], DEV)
    f = torch.nn.functional.normalize(f.float(), dim=1)
    d = jaccard(f, args.k1, args.k2)
    lab = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="precomputed", n_jobs=-1).fit_predict(d)
    return f, lab


def quality(lab, pids):
    """의사 라벨 품질(학습에는 쓰지 않음): 묶인 비율, 묶음 수, 순도(묶음 안 다수 사람 비율), ARI."""
    m = lab >= 0
    pur = 0
    for c in set(lab[m].tolist()):
        p = pids[lab == c]
        pur += np.bincount(p).max()
    return {"clustered": m.mean(), "n_clusters": len(set(lab[m].tolist())), "purity": pur / max(1, m.sum()),
            "ari": adjusted_rand_score(pids[m], lab[m]) if m.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, default=ROOT / "weights/clip_vit_b16_vast_lowlight_ft_night_c1234.pth")
    ap.add_argument("--out", type=Path, default=ROOT / "weights/clip_vit_b16_vast_lowlight_ft_night_c1234_adapt58.pth")
    ap.add_argument("--cams", default="5,6,7,8", help="적응할 새 현장 카메라 (1부터)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2, help="라운드당 에폭")
    ap.add_argument("--lr", type=float, default=7e-6)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--min-samples", type=int, default=4)
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=6)
    ap.add_argument("--csv", type=Path, default=ROOT / "results/adapt_site.csv")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    cams = {int(c) - 1 for c in args.cams.split(",")}
    items = [x for x in list_night600("train") if x.camid in cams]
    pids = np.array([x.pid for x in items])
    pids = np.unique(pids, return_inverse=True)[1]
    print(f"[adapt] 새 현장 카메라 {args.cams}: 라벨 없는 사진 {len(items)}장 (실제 {pids.max() + 1}명, 품질 확인에만 사용)", flush=True)

    model = load_model(None, args.init, DEV)
    model.loss = "triplet"
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    rows, t0 = [], time.time()
    for r in range(1, args.rounds + 1):
        model.eval()
        tc = time.time()
        f, lab = cluster(model, items, args)
        q = quality(lab, pids)
        print(f"  round {r}/{args.rounds}  묶인 사진 {q['clustered'] * 100:4.1f}%  묶음 {q['n_clusters']}개  "
              f"순도 {q['purity'] * 100:4.1f}%  ARI {q['ari']:.3f}  ({time.time() - tc:.0f}s)", flush=True)
        keep = np.where(lab >= 0)[0]
        sub = [items[i] for i in keep]
        y = np.unique(lab[keep], return_inverse=True)[1]
        n_cls = int(y.max()) + 1
        # 분류기 = 묶음 중심 (BNNeck 뒤 특징 평균) → 새 묶음마다 처음부터 의미 있는 로짓
        with torch.no_grad():
            fb, _ = extract(model, [it.path for it in sub], DEV)
            cen = torch.zeros(n_cls, fb.shape[1], device=DEV).index_add_(0, torch.from_numpy(y).to(DEV), fb.float())
            cen /= torch.bincount(torch.from_numpy(y).to(DEV), minlength=n_cls)[:, None]
        model.classifier = torch.nn.Linear(fb.shape[1], n_cls, bias=False).to(DEV)
        # 자기 묶음 로짓 ≈ 16·cos 가 되게 크기 조정 (특징 노름이 커서 중심을 그대로 쓰면 로짓이 폭주)
        model.classifier.weight.data.copy_(torch.nn.functional.normalize(cen, dim=1) * (16 / fb.float().norm(dim=1).mean()))
        ds = TrainDS(sub, y.tolist())
        nb = len(sub) // 64
        dl = torch.utils.data.DataLoader(ds, batch_sampler=PKSampler(y.tolist(), 16, 4, nb), num_workers=8, pin_memory=True)
        head = [p for n, p in model.named_parameters() if n.startswith(("classifier", "bnneck")) and p.requires_grad]
        body = [p for n, p in model.named_parameters() if not n.startswith(("classifier", "bnneck")) and p.requires_grad]
        opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}], weight_decay=5e-4)
        steps = args.epochs * nb
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
        model.train()
        tl = tn = 0.0
        for ep in range(args.epochs):
            for x, yy in dl:
                x, yy = x.to(DEV, non_blocking=True), yy.to(DEV, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, ff = model(x)
                    loss = crit(out.float(), yy) + triplet_hard(ff.float(), yy)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tl += float(loss.detach()) * len(yy)
                tn += len(yy)
        print(f"           학습 {args.epochs}에폭  loss {tl / tn:.3f}  GPU {gpu_temp()}°C  (누적 {time.time() - t0:.0f}s)", flush=True)
        rows.append({"round": r, **{k: round(float(v), 4) for k, v in q.items()}, "train_imgs": len(sub), "loss": round(tl / tn, 4)})
    model.eval()
    f, lab = cluster(model, items, args)
    q = quality(lab, pids)
    print(f"  최종     묶인 사진 {q['clustered'] * 100:4.1f}%  묶음 {q['n_clusters']}개  순도 {q['purity'] * 100:4.1f}%  ARI {q['ari']:.3f}")
    rows.append({"round": "final", **{k: round(float(v), 4) for k, v in q.items()}, "train_imgs": "", "loss": ""})
    sd = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("classifier.")}
    torch.save(sd, args.out)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[adapt] saved {args.out}, {args.csv}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
