"""Variance-reduced gain estimation for sample-based influence minimisation.

The saving of a blocker v on a live-edge sample is 0 unless v is reached, and
large when it is; with heavy-tailed cascades the sample mean over theta
samples is noisy and the greedy selects on noise (docs/RESULTS_NOTES.md).

Control variates with exactly known means. For every node v let
h_phi(v) = 1[some edge from a seed into v is live]; its expectation
q_v = 1 - prod_{s in S} (1 - p(s, v)) is known exactly. The estimator

    g~(v) = mean_i g_i(v) - beta_v ( mean_i h_i(v) - q_v ),

with beta_v the sample regression slope of g on h (shrunk towards the
pooled slope), is unbiased for E[g(v)] and has variance (1 - rho_v^2) times
that of the plain mean, rho_v = corr(g(v), h(v)). Nodes with no seed
in-edge fall back to the plain mean. A second control variate uses the
number of live seed edges into v (mean sum_s p(s, v)), which captures the
multiplicity of exposure.
"""
from __future__ import annotations

import numpy as np

from ..graphs import CSRGraph
from .imin import SampleSet


class ControlVariates:
    def __init__(self, g: CSRGraph, seeds, forbidden: np.ndarray):
        self.g = g
        self.seeds = [int(s) for s in seeds]
        n = g.n
        # seed out-edges: edge positions and targets
        pos, tgt, prob = [], [], []
        for s in self.seeds:
            lo, hi = g.indptr[s], g.indptr[s + 1]
            pos.append(np.arange(lo, hi))
            tgt.append(g.indices[lo:hi])
            prob.append(g.weights[lo:hi])
        self.pos = np.concatenate(pos) if pos else np.zeros(0, dtype=np.int64)
        self.tgt = np.concatenate(tgt).astype(np.int64) if tgt else np.zeros(0, dtype=np.int64)
        self.prob = np.concatenate(prob) if prob else np.zeros(0)
        keep = ~forbidden[self.tgt]
        self.pos, self.tgt, self.prob = self.pos[keep], self.tgt[keep], self.prob[keep]
        log1m = np.zeros(n)
        np.add.at(log1m, self.tgt, np.log1p(-np.minimum(self.prob, 1 - 1e-12)))
        self.q = -np.expm1(log1m)  # exact P(some seed edge into v is live)
        self.adjacent = np.flatnonzero(self.q > 0)

    def h(self, live: np.ndarray) -> np.ndarray:
        """Indicator, per node, that a live seed edge enters v in this sample."""
        hv = np.zeros(self.g.n)
        hv[self.tgt[live[self.pos]]] = 1.0
        return hv


def cv_gains(S: SampleSet, cv: ControlVariates, shrink: float = 5.0):
    """Control-variate estimate of the single-node savings on the current sample set (after S.refresh())."""
    n = S.g.n
    theta = S.theta
    G = np.zeros((theta, n))
    H = np.zeros((theta, n))
    for i, (order, index, sizes, reach) in enumerate(S.cache):
        if len(order):
            G[i, order] = sizes[index[order]]
        H[i] = cv.h(S.lives[i])
    gm = G.mean(axis=0)
    gm[S.seed_mask] = 0
    gm[S.forbidden] = 0
    adj = cv.adjacent
    adj = adj[~S.forbidden[adj] & ~S.seed_mask[adj]]
    if len(adj) == 0 or theta < 4:
        return gm
    Ga, Ha = G[:, adj], H[:, adj]
    hm = Ha.mean(axis=0)
    cov = ((Ga - Ga.mean(axis=0)) * (Ha - hm)).sum(axis=0) / (theta - 1)
    var = ((Ha - hm) ** 2).sum(axis=0) / (theta - 1)
    # pooled slope as a shrinkage target: E[g | hit] - E[g | miss] ~ conditional mean saving
    pooled = float(cov.sum() / max(var.sum(), 1e-12))
    beta = (theta * np.where(var > 0, cov / np.maximum(var, 1e-12), 0.0) + shrink * pooled) / (theta + shrink)
    out = gm.copy()
    out[adj] = gm[adj] - beta * (hm - cv.q[adj])
    return np.maximum(out, 0.0)


def cv_greedy(S: SampleSet, budget: int) -> list[int]:
    """AdvancedGreedy with control-variate gain estimates."""
    cv = ControlVariates(S.g, S.seeds, S.forbidden)
    for _ in range(budget):
        S.refresh()
        gains = cv_gains(S, cv)
        gains[S.seed_mask] = -np.inf
        gains[S.forbidden] = -np.inf
        v = int(np.argmax(gains))
        if gains[v] <= 0:
            break
        S.block([v])
    return list(S.blocked)
