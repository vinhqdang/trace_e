"""Dynamic message passing source estimator (Lokhov, Mezard, Ohta & Zdeborova, PRE 2014).

For each candidate source ``i`` the discrete-time DMP equations for SIR are
iterated for ``t_steps`` steps with per-step transmission probability
``lam = 1 - exp(-beta*dt)`` and recovery probability ``mu = 1 - exp(-nu*dt)``,
which discretises the continuous-time model used in the simulator. The
candidate is scored by the mean-field log-likelihood of the observed snapshot
``sum_j log P_{x_j}^j(T | i)``. With ``unknown_T=True`` the score is maximised
over the number of elapsed steps.
"""
from __future__ import annotations

import math

import numpy as np

from .base import SourceDetector, register


@register
class DMP(SourceDetector):
    name = "dmp"
    probabilistic = True

    def __init__(self, ctx, dt: float = 0.05, unknown_T: bool = False, eps: float = 1e-6, max_candidates: int = 400, **params):
        super().__init__(ctx, **params)
        self.dt = dt
        self.unknown_T = unknown_T
        self.eps = eps
        self.max_candidates = max_candidates
        g = ctx.g
        # directed edge list k->i for every undirected edge
        rows, cols, w = [], [], []
        for k in range(g.n):
            lo, hi = g.indptr[k], g.indptr[k + 1]
            rows.append(np.full(hi - lo, k))
            cols.append(g.indices[lo:hi])
            w.append(g.weights[lo:hi])
        self.src = np.concatenate(rows).astype(np.int64)  # k
        self.dst = np.concatenate(cols).astype(np.int64)  # i
        self.w = np.concatenate(w)
        # index of reverse edge (i->k) for each edge (k->i)
        key = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(self.src, self.dst))}
        self.rev = np.array([key[(int(b), int(a))] for a, b in zip(self.src, self.dst)], dtype=np.int64)
        self.t_steps = max(1, int(round(ctx.T / dt)))
        self.lam = 1.0 - np.exp(-ctx.beta * self.w * dt)  # per directed edge
        self.mu = 1.0 - math.exp(-ctx.nu * dt)

    def _marginals(self, source: int) -> np.ndarray:
        """Returns array (t_steps+1, n, 3) of P_S, P_I, P_R for every node."""
        n = self.ctx.n
        E = len(self.src)
        PS0 = np.ones(n)
        PS0[source] = 0.0
        theta = np.ones(E)
        phi = (self.src == source).astype(np.float64)  # phi^{k->i}(0) = P_I^k(0)
        PS_edge_prev = PS0[self.src].copy()  # P_S^{k->i}(0)
        PS = PS0.copy()
        PR = np.zeros(n)
        PI = 1.0 - PS - PR
        out = np.zeros((self.t_steps + 1, n, 3))
        out[0, :, 0], out[0, :, 1], out[0, :, 2] = PS, PI, PR
        log_theta_sum = np.zeros(n)
        for t in range(1, self.t_steps + 1):
            theta = theta - self.lam * phi
            theta = np.clip(theta, 1e-300, 1.0)
            log_theta = np.log(theta)
            log_theta_sum = np.zeros(n)
            np.add.at(log_theta_sum, self.dst, log_theta)  # sum over k in d(i) of log theta^{k->i}
            # P_S^{k->i}(t) = P_S^k(0) prod_{l in d(k) \ i} theta^{l->k} = P_S^k(0) exp(sum_k - log theta^{i->k})
            PS_edge = PS0[self.src] * np.exp(log_theta_sum[self.src] - log_theta[self.rev])
            phi = (1.0 - self.lam) * (1.0 - self.mu) * phi - (PS_edge - PS_edge_prev)
            phi = np.clip(phi, 0.0, 1.0)
            PS_edge_prev = PS_edge
            PR = PR + self.mu * PI
            PS = PS0 * np.exp(log_theta_sum)
            PI = np.clip(1.0 - PS - PR, 0.0, 1.0)
            out[t, :, 0], out[t, :, 1], out[t, :, 2] = PS, PI, PR
        return out

    def score(self, states):
        inf = np.flatnonzero(states != 0)
        s = np.full(self.ctx.n, -np.inf)
        if len(inf) == 1:
            s[inf[0]] = 0.0
            return s
        cands = inf
        if len(cands) > self.max_candidates:
            # prune with distance centrality to keep runtime bounded
            from .centrality import DistanceCenter
            dc = DistanceCenter(self.ctx).score(states)
            cands = np.argsort(-dc)[: self.max_candidates]
        idx = np.arange(self.ctx.n)
        for i in cands:
            m = self._marginals(int(i))
            probs = np.clip(m[:, idx, states.astype(int)], self.eps, 1.0)  # (t, n)
            ll = np.log(probs).sum(axis=1)
            s[i] = ll.max() if self.unknown_T else ll[-1]
        return s
