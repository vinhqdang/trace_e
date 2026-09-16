"""RAG: Racing AdvancedGreedy for influence minimisation by vertex blocking.

Diagnosis (see docs/RAG.md): the marginal saving of a blocker is estimated
on theta live-edge samples whose cascade sizes are heavy-tailed, so
AdvancedGreedy with a fixed theta selects nodes on sampling noise; its
sample objective is far below its true objective and the true objective
keeps improving with theta.

RAG treats every greedy step as a full-information best-arm identification
problem: all candidates are observed on every sample; samples are added in
batches until the empirical-Bernstein confidence interval of the leader
separates from the runner-up (or a cap is reached). The output is, with
probability at least 1 - delta, the greedy sequence that AdvancedGreedy
would produce with infinitely many samples, at a sample size that adapts to
the gaps between candidates.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..graphs import CSRGraph
from .imin import SampleSet


def _bernstein_radius(var: np.ndarray, rng_bound: float, n: int, delta: float) -> np.ndarray:
    """Empirical-Bernstein deviation for the mean of n i.i.d. variables in [0, rng_bound]."""
    l = math.log(3.0 / delta)
    return np.sqrt(2.0 * var * l / n) + 3.0 * rng_bound * l / n


def racing_greedy(g: CSRGraph, seeds, budget: int, rng: np.random.Generator, theta0: int = 50, theta_max: int = 2000,
                  batch: int = 50, delta: float = 0.1, forbidden=None, log=None):
    """Returns (B, info). ``info['theta_used']`` lists the sample size at which each pick was decided."""
    seeds = [int(s) for s in seeds]
    S = SampleSet(g, seeds, theta0, rng, forbidden=forbidden)
    n = g.n
    picks, thetas, stats = [], [], []
    t0 = time.perf_counter()
    delta_step = delta / max(budget, 1)  # union bound over the b greedy steps
    for step in range(budget):
        while True:
            S.refresh()
            theta = S.theta
            acc = np.zeros(n)
            acc2 = np.zeros(n)
            rmax = 1.0
            for order, index, sizes, reach in S.cache:
                if len(order):
                    gv = sizes[index[order]].astype(float)
                    acc[order] += gv
                    acc2[order] += gv * gv
                    rmax = max(rmax, float(gv.max()))
            acc[S.seed_mask] = 0
            acc[S.forbidden] = 0
            mean = acc / theta
            var = np.maximum(acc2 / theta - mean * mean, 0.0)
            if mean.max() <= 0:
                break
            order_idx = np.argsort(-mean)
            lead, second = int(order_idx[0]), int(order_idx[1])
            # per-candidate union bound over the n candidates
            rad = _bernstein_radius(var[[lead, second]], rmax, theta, delta_step / n)
            gap = mean[lead] - mean[second]
            separated = gap > rad[0] + rad[1]
            if separated or theta >= theta_max:
                break
            # add a batch of samples (all cached structures for the new samples are computed on refresh)
            new = [rng.random(len(g.indices)) < g.weights for _ in range(min(batch, theta_max - theta))]
            S.lives.extend(new)
            S.cache.extend([None] * len(new))
            S.dirty.extend([True] * len(new))
            S.theta = len(S.lives)
        if mean.max() <= 0:
            break
        S.block([lead])
        picks.append(lead)
        thetas.append(S.theta)
        stats.append((float(mean[lead]), float(mean[second]), float(rad[0] + rad[1]) if len(order_idx) > 1 else 0.0))
        if log:
            log(f"step {step}: pick {lead} gain {mean[lead]:.2f} vs {mean[second]:.2f} theta={S.theta}")
    return picks, {"theta_used": thetas, "theta_final": S.theta, "time_s": time.perf_counter() - t0, "stats": stats, "sample_set": S}
