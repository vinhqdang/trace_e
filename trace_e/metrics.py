"""Evaluation metrics for source detection.

All detectors output a score vector ``s`` of shape ``(N,)`` where higher means
more likely source and ``-inf`` marks excluded nodes. Probabilistic detectors
output (unnormalised) log-probabilities so that distributional metrics such as
the credible set size are meaningful.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import rankdata

from .graphs import CSRGraph


def bfs_distances(g: CSRGraph, source: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Hop distances from ``source``; ``mask`` restricts traversal to a node subset."""
    dist = np.full(g.n, -1, dtype=np.int32)
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in g.indices[g.indptr[u]: g.indptr[u + 1]]:
            if dist[v] < 0 and (mask is None or mask[v]):
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def top_k_hit(scores: np.ndarray, true_source: int, k: int, rng: np.random.Generator) -> bool:
    """Whether the true source is in the top-k set, breaking ties uniformly at random."""
    s = scores.astype(np.float64, copy=True)
    chosen: list[int] = []
    while len(chosen) < k:
        finite = np.isfinite(s)
        if not finite.any():
            break
        m = s[finite].max()
        idx = np.flatnonzero(s == m)
        need = k - len(chosen)
        if len(idx) <= need:
            chosen.extend(idx.tolist())
        else:
            chosen.extend(rng.choice(idx, size=need, replace=False).tolist())
        s[idx] = -np.inf
    return true_source in chosen


def map_estimate(scores: np.ndarray, rng: np.random.Generator) -> int:
    finite = np.isfinite(scores)
    if not finite.any():
        return int(rng.integers(len(scores)))
    m = scores[finite].max()
    idx = np.flatnonzero(scores == m)
    return int(idx[0]) if len(idx) == 1 else int(rng.choice(idx))


def error_distance(g: CSRGraph, pred: int, true_source: int) -> int:
    d = bfs_distances(g, true_source)
    return int(d[pred]) if d[pred] >= 0 else -1


def reciprocal_rank(scores: np.ndarray, true_source: int) -> float:
    s = np.where(np.isfinite(scores), scores, -np.inf)
    ranks = rankdata(-s, method="average")
    return float(1.0 / ranks[true_source])


def normalise_log_scores(scores: np.ndarray) -> np.ndarray:
    """Softmax over finite entries (log-sum-exp), zeros elsewhere."""
    p = np.zeros_like(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        return p
    z = scores[finite] - scores[finite].max()
    e = np.exp(z)
    p[finite] = e / e.sum()
    return p


def credible_set_size(scores: np.ndarray, level: float = 0.9) -> int:
    p = normalise_log_scores(scores)
    v = np.sort(p)[::-1]
    c = np.cumsum(v)
    return int(np.searchsorted(c, level - 1e-12) + 1)


def brier_score(scores: np.ndarray, true_source: int) -> float:
    p = normalise_log_scores(scores)
    y = np.zeros_like(p)
    y[true_source] = 1.0
    return float(((p - y) ** 2).sum())


class ResistanceScore:
    """Graph-aware proper scoring rule based on resistance distance.

    score(p, y) = sum_j p_j R[y, j] - 0.5 * p^T R p  (lower is better), where
    R is the resistance-distance matrix of the graph.
    """

    def __init__(self, A: np.ndarray):
        A = np.asarray(A, dtype=np.float64)
        L = np.diag(A.sum(1)) - A
        Li = np.linalg.pinv(L)
        d = np.diag(Li)
        self.R = d[:, None] + d[None, :] - 2 * Li

    def __call__(self, scores: np.ndarray, true_source: int) -> float:
        p = normalise_log_scores(scores)
        return float(self.R[true_source] @ p - 0.5 * p @ self.R @ p)


def evaluate_instance(g: CSRGraph, scores: np.ndarray, true_source: int, rng: np.random.Generator,
                      probabilistic: bool, resistance: ResistanceScore | None = None) -> dict:
    pred = map_estimate(scores, rng)
    out = {
        "map": pred,
        "top1": int(top_k_hit(scores, true_source, 1, rng)),
        "top3": int(top_k_hit(scores, true_source, 3, rng)),
        "top5": int(top_k_hit(scores, true_source, 5, rng)),
        "ed": error_distance(g, pred, true_source),
        "rr": reciprocal_rank(scores, true_source),
    }
    if probabilistic:
        out["css"] = credible_set_size(scores)
        out["brier"] = brier_score(scores, true_source)
        if resistance is not None:
            out["resist"] = resistance(scores, true_source)
    return out


SUMMARY_KEYS = ["top1", "top3", "top5", "ed", "rr", "css", "brier", "resist"]


def summarise(rows: list[dict]) -> dict:
    out = {"n_instances": len(rows)}
    for k in SUMMARY_KEYS:
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        if vals:
            out[k] = float(np.mean(vals))
            if k in ("top1", "top3", "top5"):
                out[k + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return out
