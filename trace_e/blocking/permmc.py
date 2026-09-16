"""Permutation Monte Carlo for the expected reach of an independent cascade.

Crude Monte Carlo draws the live/dead state of every edge and measures the
reachable set once. Permutation Monte Carlo (Elperin, Gertsbakh & Lomonosov
1991) instead draws a random *birth order* of the edges, adds edges one at a
time while maintaining the set of nodes reachable from the seeds, and then
takes the exact expectation over the number K of edges alive at time 1:

    E[reach | order] = sum_k P(K = k) reach(first k edges in the order).

The estimator is a conditional expectation of the crude one, so its variance
is never larger (Rao-Blackwell) and it integrates analytically over the
"giant component or not" randomness that dominates the crude estimator on
near-critical graphs. For a constant edge probability p the order is a
uniform permutation and K ~ Binomial(m, p) independently of the order, so
one incremental reachability sweep of O(m) per permutation gives the exact
conditional expectation. Heterogeneous probabilities use exponential birth
times with rate -log(1 - p_e); K given the order is then hypoexponential
and is approximated here by a Poisson-binomial over the *ordered* edges with
weights taken as the marginal probabilities (exact when p is constant).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from ..graphs import CSRGraph


class PermutationMC:
    def __init__(self, g: CSRGraph, seeds, forbidden: np.ndarray | None = None):
        self.g = g
        self.seeds = [int(s) for s in seeds]
        self.forbidden = np.zeros(g.n, dtype=bool) if forbidden is None else forbidden.copy()
        # directed edge list
        self.src = np.repeat(np.arange(g.n), np.diff(g.indptr))
        self.dst = g.indices.astype(np.int64)
        self.p = g.weights
        keep = ~self.forbidden[self.src] & ~self.forbidden[self.dst]
        self.src, self.dst, self.p = self.src[keep], self.dst[keep], self.p[keep]
        self.m = len(self.src)
        self.const = bool(np.allclose(self.p, self.p[0])) if self.m else True
        if self.const:
            self.pk = binom.pmf(np.arange(self.m + 1), self.m, float(self.p[0]) if self.m else 0.0)

    def _order(self, rng):
        if self.const:
            return rng.permutation(self.m)
        t = rng.exponential(1.0, size=self.m) / np.maximum(-np.log1p(-np.minimum(self.p, 1 - 1e-12)), 1e-12)
        return np.argsort(t)

    def _weights(self, order):
        if self.const:
            return self.pk
        # Poisson-binomial over the ordered marginals (exact for constant p)
        w = np.zeros(self.m + 1)
        w[0] = 1.0
        for e in order:
            pe = self.p[e]
            w[1:] = w[1:] * (1 - pe) + w[:-1] * pe
            w[0] *= (1 - pe)
        return w

    def reach_curve(self, order, blocked: np.ndarray) -> np.ndarray:
        """Number of non-seed reachable nodes after adding the first k edges, for k = 0..m."""
        g = self.g
        n = g.n
        reached = np.zeros(n, dtype=bool)
        block = blocked | self.forbidden
        for s in self.seeds:
            reached[s] = True
        # incremental adjacency of added edges
        adj: list[list[int]] = [[] for _ in range(n)]
        curve = np.zeros(self.m + 1)
        cnt = 0
        for k, e in enumerate(order, start=1):
            u, v = int(self.src[e]), int(self.dst[e])
            adj[u].append(v)
            if reached[u] and not reached[v] and not block[v]:
                # BFS from v over already-added edges
                stack = [v]
                reached[v] = True
                cnt += 1
                while stack:
                    x = stack.pop()
                    for y in adj[x]:
                        if not reached[y] and not block[y]:
                            reached[y] = True
                            cnt += 1
                            stack.append(y)
            curve[k] = cnt
        return curve

    def expected_reach(self, blocked_nodes, n_perm: int, rng: np.random.Generator):
        blocked = np.zeros(self.g.n, dtype=bool)
        blocked[list(blocked_nodes)] = True
        vals = []
        for _ in range(n_perm):
            order = self._order(rng)
            w = self._weights(order)
            vals.append(float(w @ self.reach_curve(order, blocked)))
        return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(n_perm)) if n_perm > 1 else 0.0
