"""Non-adaptive heuristics used as references throughout the blocking literature."""
from __future__ import annotations

import networkx as nx
import numpy as np

from ..metrics import bfs_distances
from .base import Blocker, register


def _candidates(ctx, bad_seeds):
    mask = np.ones(ctx.n, dtype=bool)
    mask[list(bad_seeds)] = False
    return mask


@register
class RandomBlocker(Blocker):
    name = "random"

    def select(self, bad_seeds, budget):
        rng = np.random.default_rng(self.ctx.seed + int(np.sum(bad_seeds)))
        cand = np.flatnonzero(_candidates(self.ctx, bad_seeds))
        return rng.choice(cand, size=min(budget, len(cand)), replace=False).tolist()


@register
class DegreeBlocker(Blocker):
    """Highest out-degree nodes (excluding the bad seeds)."""

    name = "degree"

    def select(self, bad_seeds, budget):
        deg = self.ctx.g.degree().astype(float)
        deg[~_candidates(self.ctx, bad_seeds)] = -1
        return np.argsort(-deg, kind="stable")[:budget].tolist()


@register
class PageRankBlocker(Blocker):
    name = "pagerank"

    def prepare(self):
        pr = nx.pagerank(self.ctx.G)
        self.pr = np.array([pr[i] for i in range(self.ctx.n)])

    def select(self, bad_seeds, budget):
        if not hasattr(self, "pr"):
            self.prepare()
        s = self.pr.copy()
        s[~_candidates(self.ctx, bad_seeds)] = -1
        return np.argsort(-s, kind="stable")[:budget].tolist()


@register
class ProximityBlocker(Blocker):
    """Nodes closest to the bad seeds, ties broken by degree (the "neighbourhood" heuristic)."""

    name = "proximity"

    def select(self, bad_seeds, budget):
        g = self.ctx.g
        d = np.full(g.n, np.inf)
        for s in bad_seeds:
            ds = bfs_distances(g, int(s))
            ds = np.where(ds < 0, np.inf, ds)
            d = np.minimum(d, ds)
        deg = g.degree().astype(float)
        key = d - 1e-3 * deg / (deg.max() + 1)
        key[~_candidates(self.ctx, bad_seeds)] = np.inf
        order = np.argsort(key, kind="stable")
        return [int(v) for v in order[:budget] if np.isfinite(key[v])]


@register
class ExpectedInfluenceBlocker(Blocker):
    """Rank nodes by their Monte-Carlo probability of being reached by the bad cascade,
    weighted by out-degree: a cheap proxy for marginal blocking gain."""

    name = "reach"

    def __init__(self, ctx, n_mc: int = 200, **params):
        super().__init__(ctx, **params)
        self.n_mc = n_mc

    def select(self, bad_seeds, budget):
        from .cascade import ic_spread
        rng = np.random.default_rng(self.ctx.seed)
        g = self.ctx.g
        reach = np.zeros(g.n)
        for _ in range(self.n_mc):
            reach += ic_spread(g, bad_seeds, None, rng)
        deg = g.degree().astype(float)
        s = reach / self.n_mc * (1 + deg)
        s[~_candidates(self.ctx, bad_seeds)] = -1
        return np.argsort(-s, kind="stable")[:budget].tolist()
