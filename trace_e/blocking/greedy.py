"""Greedy (CELF) intervention selection.

Objective: minimise the expected number of bad adopters, i.e. maximise the
"saved" count relative to no intervention. In ``counter`` mode this is the
Budak et al. (2011) influence-limitation objective, which is monotone and
submodular under the campaign-oblivious IC model, so lazy greedy gives a
(1 - 1/e) approximation of the best expected saving. In ``block`` mode
(vertex removal) the objective is not submodular in general (Xie et al.
2023), so greedy is a heuristic; it is nevertheless the standard reference.

Marginal gains are estimated on a fixed pool of live-edge samples shared by
all evaluations (common random numbers), which makes the lazy evaluations
consistent. For ``block`` mode, reachability on the live-edge sample minus
the blocked set is recomputed by BFS; for ``counter`` mode, a synchronous
two-campaign propagation on the live-edge sample is used.
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from ..graphs import CSRGraph
from .base import Blocker, register


def _reach_block(g: CSRGraph, live: np.ndarray, seeds, blocked: np.ndarray) -> int:
    seen = blocked.copy()
    cnt = 0
    q = deque()
    for s in seeds:
        if not seen[s]:
            seen[s] = True
            cnt += 1
            q.append(int(s))
    while q:
        u = q.popleft()
        lo, hi = g.indptr[u], g.indptr[u + 1]
        for v, ok in zip(g.indices[lo:hi], live[lo:hi]):
            if ok and not seen[v]:
                seen[v] = True
                cnt += 1
                q.append(int(v))
    return cnt


def _bad_count_counter(g: CSRGraph, live: np.ndarray, bad_seeds, good_seeds, good_delay: int) -> int:
    BAD, GOOD = 1, 2
    state = np.zeros(g.n, dtype=np.int8)
    bad_front = [int(s) for s in bad_seeds]
    for s in bad_front:
        state[s] = BAD
    good_front = []
    pending = [int(s) for s in good_seeds]
    t = 0
    while bad_front or good_front or pending:
        if t == good_delay:
            for s in pending:
                if state[s] == 0:
                    state[s] = GOOD
                    good_front.append(s)
            pending = []
        elif t > good_delay:
            pending = []
        ng, nb_ = [], []
        for u in good_front:
            lo, hi = g.indptr[u], g.indptr[u + 1]
            for v, ok in zip(g.indices[lo:hi], live[lo:hi]):
                if ok and state[v] == 0:
                    state[v] = GOOD
                    ng.append(int(v))
        for u in bad_front:
            lo, hi = g.indptr[u], g.indptr[u + 1]
            for v, ok in zip(g.indices[lo:hi], live[lo:hi]):
                if ok and state[v] == 0:
                    state[v] = BAD
                    nb_.append(int(v))
        good_front, bad_front = ng, nb_
        t += 1
    return int((state == BAD).sum())


@register
class GreedyBlocker(Blocker):
    name = "greedy"

    def __init__(self, ctx, n_samples: int = 200, candidate_pool: int = 0, pool_weight: str = "reach", **params):
        super().__init__(ctx, **params)
        self.n_samples = n_samples
        self.candidate_pool = candidate_pool  # 0 = all reachable nodes
        self.pool_weight = pool_weight  # "reach" or "reach_deg"

    def _samples(self, bad_seeds):
        rng = np.random.default_rng(self.ctx.seed + 17)
        g = self.ctx.g
        lives = [rng.random(len(g.indices)) < g.weights for _ in range(self.n_samples)]
        return lives

    def _objective(self, lives, bad_seeds, chosen: list[int]) -> float:
        g = self.ctx.g
        if self.ctx.mode == "block":
            blocked = np.zeros(g.n, dtype=bool)
            blocked[chosen] = True
            return float(np.mean([_reach_block(g, lv, bad_seeds, blocked) for lv in lives]))
        return float(np.mean([_bad_count_counter(g, lv, bad_seeds, chosen, self.ctx.good_delay) for lv in lives]))

    def select(self, bad_seeds, budget):
        g = self.ctx.g
        lives = self._samples(bad_seeds)
        base = self._objective(lives, bad_seeds, [])
        # candidate set: nodes reachable from the seeds in at least one sample (others have zero gain);
        # if a pool size is given, keep the nodes most often reached, weighted by degree
        reach = np.zeros(g.n, dtype=np.int64)
        blocked0 = np.zeros(g.n, dtype=bool)
        for lv in lives:
            seen = blocked0.copy()
            q = deque(int(s) for s in bad_seeds)
            for s in bad_seeds:
                seen[s] = True
            while q:
                u = q.popleft()
                lo, hi = g.indptr[u], g.indptr[u + 1]
                for v, ok in zip(g.indices[lo:hi], lv[lo:hi]):
                    if ok and not seen[v]:
                        seen[v] = True
                        q.append(int(v))
            reach += seen
        reach[list(bad_seeds)] = 0
        cands = np.flatnonzero(reach > 0)
        if self.candidate_pool and len(cands) > self.candidate_pool:
            score = reach[cands] * (1 + g.degree()[cands]) if self.pool_weight == "reach_deg" else reach[cands].astype(float)
            cands = cands[np.argsort(-score, kind="stable")[: self.candidate_pool]]
        chosen: list[int] = []
        cur = base
        # CELF lazy greedy
        heap = [(-(cur - self._objective(lives, bad_seeds, [int(v)])), int(v), 0) for v in cands]
        heapq.heapify(heap)
        while heap and len(chosen) < budget:
            gain, v, it = heapq.heappop(heap)
            if it == len(chosen):
                chosen.append(v)
                cur = cur + gain  # gain is negative of saving
                continue
            new_gain = -(cur - self._objective(lives, bad_seeds, chosen + [v]))
            heapq.heappush(heap, (new_gain, v, len(chosen)))
        return chosen


@register
class DominatorGreedyBlocker(Blocker):
    """Greedy vertex blocking with exact per-sample marginal gains via dominator trees
    (AdvancedGreedy-style, Xie et al. 2023). Block mode only; all reachable nodes are candidates."""

    name = "greedy_dom"

    def __init__(self, ctx, n_samples: int = 200, **params):
        super().__init__(ctx, **params)
        self.n_samples = n_samples

    def select(self, bad_seeds, budget):
        from ..sequential.dominators import dominator_greedy_plan
        if self.ctx.mode != "block":
            raise ValueError("greedy_dom supports block mode only")
        g = self.ctx.g
        rng = np.random.default_rng(self.ctx.seed + 17)
        lives = [rng.random(len(g.indices)) < g.weights for _ in range(self.n_samples)]
        forb = np.zeros(g.n, dtype=bool)
        plan, _ = dominator_greedy_plan(g, lives, [int(s) for s in bad_seeds], forb, budget)
        return plan
