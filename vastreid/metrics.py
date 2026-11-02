"""ReID 평가 지표: CMC(Rank-k), mAP, mINP.

Market-1501 표준 프로토콜: 각 query에 대해 같은 pid이면서 같은 카메라인 gallery는 제외.
mINP는 Ye et al., "Deep Learning for Person Re-identification: A Survey and Outlook"(AGW) 정의를 따름:
  INP = (정답 개수) / (가장 어려운 정답의 순위)
"""
from __future__ import annotations

import numpy as np


def evaluate(distmat: np.ndarray, q_pids, g_pids, q_camids, g_camids, max_rank: int = 50) -> dict:
    q_pids, g_pids = np.asarray(q_pids), np.asarray(g_pids)
    q_camids, g_camids = np.asarray(q_camids), np.asarray(g_camids)
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    order = np.argsort(distmat, axis=1, kind="stable")

    cmc_all, aps, inps = [], [], []
    for i in range(num_q):
        o = order[i]
        keep = ~((g_pids[o] == q_pids[i]) & (g_camids[o] == q_camids[i]))
        match = (g_pids[o][keep] == q_pids[i]).astype(np.int32)
        if not match.any():
            continue  # 다른 카메라에 정답이 없는 query는 제외
        cum = match.cumsum()
        hit = np.ones(max_rank, np.float32)  # 필터 후 gallery가 max_rank보다 짧으면 이후는 정답 포함으로 간주
        first = int(np.argmax(match))
        hit[:min(first, max_rank)] = 0.0
        cmc_all.append(hit)
        pos = np.where(match == 1)[0]
        inps.append(cum[pos[-1]] / (pos[-1] + 1.0))
        precision = cum[pos] / (pos + 1.0)
        aps.append(precision.mean())

    if not aps:
        raise RuntimeError("유효한 query가 없습니다.")
    cmc = np.mean(cmc_all, axis=0)
    return {
        "rank1": float(cmc[0]),
        "rank5": float(cmc[min(4, max_rank - 1)]),
        "rank10": float(cmc[min(9, max_rank - 1)]),
        "mAP": float(np.mean(aps)),
        "mINP": float(np.mean(inps)),
        "num_valid_query": len(aps),
    }


def evaluate_fast(dist, q_pids, g_pids, q_camids, g_camids, max_rank: int = 50, device: str | None = None,
                  chunk: int = 512) -> dict:
    """evaluate()와 같은 결과를 GPU에서 벡터화해 계산 (하이퍼파라미터 탐색용). dist: (Nq, Ng) numpy 또는 tensor."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    D = torch.as_tensor(dist, dtype=torch.float32)
    qp, gp = torch.as_tensor(np.asarray(q_pids), device=device), torch.as_tensor(np.asarray(g_pids), device=device)
    qc, gc = torch.as_tensor(np.asarray(q_camids), device=device), torch.as_tensor(np.asarray(g_camids), device=device)
    max_rank = min(max_rank, D.shape[1])
    hits, aps, inps = [], [], []
    for s in range(0, D.shape[0], chunk):
        d = D[s:s + chunk].to(device)
        order = torch.sort(d, dim=1, stable=True).indices
        gpo, gco = gp[order], gc[order]
        same_pid = gpo == qp[s:s + chunk, None]
        keep = ~(same_pid & (gco == qc[s:s + chunk, None]))
        match = same_pid & keep
        rank = torch.cumsum(keep.int(), dim=1)            # 제외 후 순위 (1부터)
        cm = torch.cumsum(match.int(), dim=1)
        nm = cm[:, -1]
        valid = nm > 0
        first = torch.where(match, rank, torch.full_like(rank, 10**9)).min(dim=1).values
        hit = (first[:, None] <= torch.arange(1, max_rank + 1, device=device)[None, :]).float()
        prec = torch.where(match, cm.float() / rank.float(), torch.zeros_like(rank, dtype=torch.float32))
        ap = prec.sum(1) / nm.clamp(min=1)
        last = torch.where(match, rank, torch.zeros_like(rank)).max(dim=1).values
        inp = nm.float() / last.clamp(min=1).float()
        hits.append(hit[valid]); aps.append(ap[valid]); inps.append(inp[valid])
    cmc = torch.cat(hits).mean(0).cpu().numpy()
    ap, inp = torch.cat(aps), torch.cat(inps)
    if len(ap) == 0:
        raise RuntimeError("유효한 query가 없습니다.")
    return {"rank1": float(cmc[0]), "rank5": float(cmc[min(4, max_rank - 1)]),
            "rank10": float(cmc[min(9, max_rank - 1)]), "mAP": float(ap.mean()), "mINP": float(inp.mean()),
            "num_valid_query": int(len(ap))}
