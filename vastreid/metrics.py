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
