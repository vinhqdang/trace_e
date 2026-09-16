"""PH-CUT: influence minimisation as stochastic vertex interdiction solved by
scenario-wise minimum cuts with progressive hedging.

Sample-average problem (theta live-edge graphs phi_i, seeds S, budget b):

    min_{B, |B| <= b}  F(B) = (1/theta) sum_i r_i(B),

r_i(B) = number of non-seed nodes reachable from S in phi_i avoiding B.

**Lemma (exact scenario subproblem).** For one live-edge graph and node
costs c(v) >= 0, min_B [ sum_{v in B} c(v) + r(B) ] equals the minimum s*-t*
cut of the network: s* -> seed_in (inf); for every node v an arc v_in -> v_out
of capacity c(v) (inf for seeds); for every live edge (u, v) an arc
u_out -> v_in (inf); for every non-seed v an arc v_out -> t* of capacity 1.
A finite cut pays c(v) for each blocked node (v_in in the source side, v_out
not) and 1 for each unblocked node whose v_out is on the source side, which
is exactly the set of nodes reachable from S avoiding the blocked ones.

**Algorithm.** Lagrangian for the budget (multiplier mu) and scenario
copies B_i tied by progressive hedging (Rockafellar & Wets 1991): each
iteration solves theta min-cuts with costs c_i(v) = mu + w_i(v) + rho (1/2 -
xbar(v)), averages the indicator vectors into xbar, updates w_i, and rounds
xbar to a feasible B (top-b) that is evaluated exactly on the samples; the
best rounded solution over iterations and over a bisection on mu is kept.

**Dual bound.** For any mu >= 0 and multipliers with sum_i w_i = 0,
LB(mu, w) = (1/theta) sum_i min_x [ r_i(x) + sum_v (mu + theta w_i(v)) x_v ] - mu b
is a valid lower bound on min_{|B|<=b} F(B) (weak duality), computed with
theta more min-cuts; the gap F(B) - LB certifies the solution on the samples.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow

from ..graphs import CSRGraph

SCALE = 1000  # capacities are integers for scipy's max-flow
INF = 10**9


def _region(g: CSRGraph, live: np.ndarray, seeds, forbidden: np.ndarray):
    """Nodes reachable from the seeds along live edges avoiding forbidden nodes (seeds included)."""
    seen = np.zeros(g.n, dtype=bool)
    order = []
    q = deque()
    for s in seeds:
        if not seen[s]:
            seen[s] = True
            q.append(int(s))
            order.append(int(s))
    while q:
        u = q.popleft()
        lo, hi = g.indptr[u], g.indptr[u + 1]
        for v, ok in zip(g.indices[lo:hi], live[lo:hi]):
            if ok and not seen[v] and not forbidden[v]:
                seen[v] = True
                q.append(int(v))
                order.append(int(v))
    return np.array(order, dtype=np.int64)


class ScenarioCut:
    """Min-cut oracle for one live-edge sample, restricted to its reachable region."""

    def __init__(self, g: CSRGraph, live: np.ndarray, seeds, forbidden: np.ndarray):
        self.nodes = _region(g, live, seeds, forbidden)
        self.k = len(self.nodes)
        idx = {int(v): i for i, v in enumerate(self.nodes)}
        self.is_seed = np.zeros(self.k, dtype=bool)
        for s in seeds:
            if s in idx:
                self.is_seed[idx[s]] = True
        # network node ids: s* = 0, t* = 1, v_in = 2 + 2i, v_out = 3 + 2i
        rows, cols, caps = [], [], []
        for i, v in enumerate(self.nodes):
            if self.is_seed[i]:
                rows.append(0)
                cols.append(2 + 2 * i)
                caps.append(INF)
            lo, hi = g.indptr[v], g.indptr[v + 1]
            for w, ok in zip(g.indices[lo:hi], live[lo:hi]):
                if ok and int(w) in idx:
                    rows.append(3 + 2 * i)
                    cols.append(2 + 2 * idx[int(w)])
                    caps.append(INF)
            if not self.is_seed[i]:
                rows.append(3 + 2 * i)
                cols.append(1)
                caps.append(SCALE)  # unit penalty for staying reachable
        self.fixed = (np.array(rows), np.array(cols), np.array(caps, dtype=np.int64))
        self.N = 2 + 2 * self.k

    def solve(self, cost: np.ndarray):
        """cost: per-node blocking cost (graph indexing, in units of 'nodes'); returns (blocked set, reach, objective)."""
        if self.k == 0:
            return [], 0, 0.0
        rows, cols, caps = self.fixed
        c = np.maximum(cost[self.nodes], 0.0)
        c_int = np.where(self.is_seed, INF, np.minimum(INF, np.round(c * SCALE).astype(np.int64)))
        r2 = np.concatenate([rows, 2 + 2 * np.arange(self.k)])
        c2 = np.concatenate([cols, 3 + 2 * np.arange(self.k)])
        cap = np.concatenate([caps, c_int])
        A = sp.csr_matrix((cap.astype(np.int32), (r2, c2)), shape=(self.N, self.N))
        A.sum_duplicates()
        res = maximum_flow(A, 0, 1)
        flow = res.flow if hasattr(res, "flow") else res.residual  # antisymmetric: flow[j, i] = -flow[i, j]
        fpos = flow.maximum(0)
        R = ((A - fpos) + fpos.T).tocsr()  # forward residual plus cancellable backward flow
        R.data = (R.data > 0).astype(np.int32)
        R.eliminate_zeros()
        seen = np.zeros(self.N, dtype=bool)
        seen[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for w in R.indices[R.indptr[u]: R.indptr[u + 1]]:
                if not seen[w]:
                    seen[w] = True
                    q.append(int(w))
        vin = seen[2 + 2 * np.arange(self.k)]
        vout = seen[3 + 2 * np.arange(self.k)]
        blocked_pos = np.flatnonzero(vin & ~vout & ~self.is_seed)
        reached_pos = np.flatnonzero(vout & ~self.is_seed)
        blocked = [int(self.nodes[i]) for i in blocked_pos]
        reach = int(len(reached_pos))
        return blocked, reach, float(c[blocked_pos].sum() + reach)


def evaluate(g: CSRGraph, lives, seeds, forbidden: np.ndarray, B) -> float:
    forb = forbidden.copy()
    forb[list(B)] = True
    tot = 0.0
    for lv in lives:
        tot += len(_region(g, lv, seeds, forb)) - len([s for s in seeds if not forbidden[s]])
    return tot / len(lives)


def phcut(g: CSRGraph, lives, seeds, forbidden: np.ndarray, budget: int, rho: float = 0.5, iters: int = 25,
          mu_bisect: int = 6, fallback=None, log=None) -> tuple[list[int], dict]:
    """Returns (B, info) with info holding the sample objective, the dual bound and timing."""
    t0 = time.perf_counter()
    seeds = [int(s) for s in seeds]
    scen = [ScenarioCut(g, lv, seeds, forbidden) for lv in lives]
    theta = len(scen)
    n = g.n
    base = evaluate(g, lives, seeds, forbidden, [])
    best_B, best_F = [], base
    if fallback is not None:
        fb = [v for v in fallback if not forbidden[v]][:budget]
        best_B, best_F = fb, evaluate(g, lives, seeds, forbidden, fb)
    best_lb = 0.0
    # bracket mu: mu = 0 blocks everything reachable; large mu blocks nothing
    lo, hi = 0.0, float(max(1.0, base))
    for _ in range(mu_bisect):
        mu = 0.5 * (lo + hi)
        w = np.zeros((theta, n))
        xbar = np.zeros(n)
        size = 0
        for it in range(iters):
            X = np.zeros((theta, n))
            for i, sc in enumerate(scen):
                cost = mu + w[i] + rho * (0.5 - xbar)
                Bi, _, _ = sc.solve(cost)
                if Bi:
                    X[i, Bi] = 1.0
            xbar = X.mean(axis=0)
            w += rho * (X - xbar)
            size = int((xbar >= 0.5).sum())
            # round: top-b by consensus, ties by how often the node is chosen
            order = np.argsort(-xbar, kind="stable")
            B = [int(v) for v in order[:budget] if xbar[v] > 0]
            F = evaluate(g, lives, seeds, forbidden, B) if B else base
            if F < best_F:
                best_B, best_F = B, F
            if np.allclose(X, X[0][None, :]):
                break  # consensus reached
        # dual bound at this (mu, w): sum_i w_i = 0 holds by construction
        lb = 0.0
        for i, sc in enumerate(scen):
            _, _, val = sc.solve(mu + theta * w[i])
            lb += val
        lb = lb / theta - mu * budget
        best_lb = max(best_lb, lb)
        if log:
            log(f"mu={mu:.3f} consensus size={size} bestF={best_F:.2f} lb={lb:.2f}")
        if size > budget:
            lo = mu  # blocking too cheap
        else:
            hi = mu
    if len(best_B) < budget and fallback is not None:
        extra = [v for v in fallback if v not in best_B and not forbidden[v]][: budget - len(best_B)]
        cand = best_B + extra
        Fc = evaluate(g, lives, seeds, forbidden, cand)
        if Fc <= best_F:
            best_B, best_F = cand, Fc
    return best_B, {"F": best_F, "F_none": base, "lower_bound": best_lb, "gap": best_F - best_lb, "time_s": time.perf_counter() - t0}
