"""Containment strategies applied once a detector fires.

``act(ep, kappa_hat, budget_left)`` returns the nodes to intervene on in the
current round. One-shot containers spend the whole budget at the first call
(the Problem B setting with seeds = current active set). The adaptive
frontier container (AVID) plans with the learned kappa, commits only the
planned nodes that are exposed now, and re-plans every round.
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from ..graphs import CSRGraph
from ..metrics import bfs_distances
from .dominators import dominator_greedy_plan

CONTAINERS: dict[str, type] = {}


def register(cls):
    CONTAINERS[cls.name] = cls
    return cls


def get_container(name: str, **kw):
    if name not in CONTAINERS:
        raise KeyError(f"unknown container '{name}'. Known: {sorted(CONTAINERS)}")
    return CONTAINERS[name](**kw)


def scaled(g: CSRGraph, kappa: float) -> CSRGraph:
    return CSRGraph(n=g.n, indptr=g.indptr, indices=g.indices, weights=np.minimum(1.0, kappa * g.weights))


def reach_count(g: CSRGraph, live: np.ndarray, sources, forbidden: np.ndarray) -> int:
    """Number of nodes reached from ``sources`` along live edges, never entering ``forbidden``."""
    seen = forbidden.copy()
    cnt = 0  # counts only nodes newly reached from the sources
    q = deque(int(s) for s in sources)
    seen[list(sources)] = True
    while q:
        u = q.popleft()
        lo, hi = g.indptr[u], g.indptr[u + 1]
        for v, ok in zip(g.indices[lo:hi], live[lo:hi]):
            if ok and not seen[v]:
                seen[v] = True
                cnt += 1
                q.append(int(v))
    return cnt


def lazy_greedy_plan(g: CSRGraph, lives: list[np.ndarray], sources, forbidden: np.ndarray, budget: int,
                     candidates: np.ndarray) -> tuple[list[int], list[float]]:
    """CELF on the sample-average reach from ``sources``; returns (plan, marginal gains)."""
    def obj(extra):
        f = forbidden.copy()
        f[extra] = True
        return float(np.mean([reach_count(g, lv, sources, f) for lv in lives]))

    cur = obj([])
    heap = [(-(cur - obj([int(v)])), int(v), 0) for v in candidates]
    heapq.heapify(heap)
    plan, gains = [], []
    while heap and len(plan) < budget:
        negg, v, it = heapq.heappop(heap)
        if it == len(plan):
            if -negg <= 1e-12:
                break
            plan.append(v)
            gains.append(-negg)
            cur += negg
            continue
        heapq.heappush(heap, (-(cur - obj(plan + [v])), v, len(plan)))
    return plan, gains


def reachable_candidates(g: CSRGraph, lives, sources, forbidden: np.ndarray, pool: int = 0) -> np.ndarray:
    reach = np.zeros(g.n, dtype=np.int64)
    for lv in lives:
        seen = forbidden.copy()
        q = deque(int(s) for s in sources)
        for s in sources:
            seen[s] = True
        while q:
            u = q.popleft()
            lo, hi = g.indptr[u], g.indptr[u + 1]
            for v, ok in zip(g.indices[lo:hi], lv[lo:hi]):
                if ok and not seen[v]:
                    seen[v] = True
                    q.append(int(v))
        reach += seen
    reach[forbidden] = 0
    reach[list(sources)] = 0
    cands = np.flatnonzero(reach > 0)
    if pool and len(cands) > pool:
        cands = cands[np.argsort(-reach[cands], kind="stable")[:pool]]
    return cands


class Container:
    name = "base"

    def __init__(self, seed: int = 0, n_samples: int = 100, pool: int = 300, use_kappa: bool = True, planner: str = "dominator",
                 horizon: int = 0, **kw):
        self.seed = seed
        self.n_samples = n_samples
        self.pool = pool
        self.use_kappa = use_kappa
        self.planner = planner  # "dominator" (exact gains, all candidates) or "celf" (sampled reach, candidate pool)
        self.horizon = horizon  # 0 = whole reachable region; h > 0 = plan within h live hops of the frontier

    def plan(self, g, lives, sources, forbidden, budget):
        if self.planner == "dominator":
            return dominator_greedy_plan(g, lives, sources, forbidden, budget, horizon=self.horizon)
        cands = reachable_candidates(g, lives, sources, forbidden, self.pool)
        if len(cands) == 0:
            return [], []
        return lazy_greedy_plan(g, lives, sources, forbidden, budget, cands)

    def reset(self, budget: int):
        self.budget = budget
        self.spent = 0
        self.calls = 0

    def act(self, ep, kappa_hat: float) -> list[int]:
        raise NotImplementedError

    def _forbidden(self, ep):
        # nodes the bad cascade can no longer enter: already active/good/blocked
        return ep.active | ep.good | ep.blocked

    def _samples(self, ep, kappa_hat):
        g = scaled(ep.g, kappa_hat if self.use_kappa else 1.0)
        rng = np.random.default_rng(self.seed + 7919 * ep.t + 31)
        return g, [rng.random(len(g.indices)) < g.weights for _ in range(self.n_samples)]


@register
class NoContainer(Container):
    name = "none"

    def act(self, ep, kappa_hat):
        return []


@register
class OneShotGreedy(Container):
    """Spend the whole budget at detection: CELF on live-edge samples from the current frontier."""

    name = "greedy"

    def act(self, ep, kappa_hat):
        self.calls += 1
        if self.spent >= self.budget or self.calls > 1:
            return []
        g, lives = self._samples(ep, kappa_hat)
        plan, _ = self.plan(g, lives, ep.frontier, self._forbidden(ep), self.budget - self.spent)
        self.spent += len(plan)
        return plan


@register
class OneShotProximity(Container):
    """Spend the whole budget on the inactive nodes closest to the frontier, ties by degree."""

    name = "proximity"

    def act(self, ep, kappa_hat):
        self.calls += 1
        if self.calls > 1:
            return []
        g = ep.g
        d = np.full(g.n, np.inf)
        for s in ep.frontier:
            ds = bfs_distances(g, int(s))
            d = np.minimum(d, np.where(ds < 0, np.inf, ds))
        deg = g.degree().astype(float)
        key = d - 1e-3 * deg / (deg.max() + 1)
        key[self._forbidden(ep)] = np.inf
        order = [int(v) for v in np.argsort(key, kind="stable") if np.isfinite(key[v])][: self.budget]
        self.spent += len(order)
        return order


@register
class OneShotDegree(Container):
    name = "degree"

    def act(self, ep, kappa_hat):
        self.calls += 1
        if self.calls > 1:
            return []
        deg = ep.g.degree().astype(float)
        deg[self._forbidden(ep)] = -1
        order = [int(v) for v in np.argsort(-deg, kind="stable")[: self.budget] if deg[v] >= 0]
        self.spent += len(order)
        return order


@register
class AdaptiveFrontier(Container):
    """AVID containment: plan with the learned kappa, commit only exposed planned nodes, re-plan each round.

    A node at distance >= 2 from the frontier cannot activate in the next
    round, so deferring it costs nothing while the next round's observations
    refine the plan; the remaining budget is carried over. When
    ``commit_all=True`` the whole plan is committed (one-shot with re-planning
    of the leftover budget only).
    """

    name = "adaptive"

    def __init__(self, commit_all: bool = False, max_rounds: int = 50, replan_every: int = 1, **kw):
        super().__init__(**kw)
        self.commit_all = commit_all
        self.max_rounds = max_rounds
        self.replan_every = replan_every  # re-plan every k rounds; in between, commit exposed nodes of the standing plan

    def reset(self, budget):
        super().reset(budget)
        self.standing = []

    def act(self, ep, kappa_hat):
        self.calls += 1
        left = self.budget - self.spent
        if left <= 0 or len(ep.frontier) == 0 or self.calls > self.max_rounds:
            return []
        g = ep.g
        forb = self._forbidden(ep)
        self.standing = [v for v in self.standing if not forb[v]]
        need_plan = (self.calls - 1) % self.replan_every == 0 or not self.standing
        if need_plan:
            gs, lives = self._samples(ep, kappa_hat)
            plan, gains = self.plan(gs, lives, ep.frontier, forb, left)
            self.standing = list(plan)
        if not self.standing:
            return []
        if self.commit_all:
            chosen = self.standing[:left]
        else:
            exposed = np.zeros(g.n, dtype=bool)
            for u in ep.frontier:
                exposed[g.indices[g.indptr[u]: g.indptr[u + 1]]] = True
            chosen = [v for v in self.standing if exposed[v]][:left]
        self.standing = [v for v in self.standing if v not in set(chosen)]
        self.spent += len(chosen)
        return chosen


@register
class OneShotGreedyBaselineProbs(OneShotGreedy):
    """One-shot greedy that plans with the benign baseline p0 instead of the learned kappa_hat * p0."""

    name = "greedy_p0"

    def __init__(self, **kw):
        kw["use_kappa"] = False
        super().__init__(**kw)


@register
class AdaptiveFrontierBaselineProbs(AdaptiveFrontier):
    """Adaptive frontier container planning with the baseline p0 (ablation of the learned kappa)."""

    name = "adaptive_p0"

    def __init__(self, **kw):
        kw["use_kappa"] = False
        super().__init__(**kw)


@register
class AdaptiveCommitAll(AdaptiveFrontier):
    """Ablation: re-plans every round but commits the whole plan immediately (no deferral)."""

    name = "adaptive_commit"

    def __init__(self, **kw):
        kw["commit_all"] = True
        super().__init__(**kw)
