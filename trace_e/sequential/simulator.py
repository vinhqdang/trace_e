"""Round-based independent cascade with a fixed live-edge sample per episode.

Coupling every policy (and the no-intervention counterfactual) to the same
live-edge draw removes Monte-Carlo noise from policy comparisons: blocking a
node can only remove activations, never add them, and the counterfactual
final harm is the plain reachable set.

Edge activation probability for the bad cascade is ``min(1, kappa * p0)``;
the competing good campaign (counter mode) spreads with the baseline ``p0``
using the *same* uniforms, so its live edges are a subset of the bad ones.
"""
from __future__ import annotations

import numpy as np

from ..graphs import CSRGraph


class Episode:
    def __init__(self, g: CSRGraph, kappa: float, seeds, rng: np.random.Generator, mode: str = "block"):
        self.g = g
        self.n = g.n
        self.kappa = kappa
        self.mode = mode
        u = rng.random(len(g.indices))
        self.u = u
        self.p_bad = np.minimum(1.0, kappa * g.weights)
        self.live_bad = u < self.p_bad
        self.live_good = u < g.weights
        self.seeds = np.asarray(sorted(int(s) for s in seeds))
        self.reset()

    def reset(self):
        self.active = np.zeros(self.n, dtype=bool)
        self.good = np.zeros(self.n, dtype=bool)
        self.blocked = np.zeros(self.n, dtype=bool)
        self.active[self.seeds] = True
        self.frontier = self.seeds.copy()
        self.good_frontier = np.zeros(0, dtype=np.int64)
        self.pending_good: list[int] = []
        self.t = 0
        self.history = [self.seeds.copy()]

    # ----- interventions -------------------------------------------------
    def block(self, nodes):
        nodes = [int(v) for v in nodes if not self.active[v] and not self.good[v] and not self.blocked[v]]
        self.blocked[nodes] = True
        return nodes

    def counter_seed(self, nodes):
        nodes = [int(v) for v in nodes if not self.active[v] and not self.good[v] and not self.blocked[v]]
        self.good[nodes] = True
        self.pending_good.extend(nodes)
        return nodes

    def intervene(self, nodes):
        return self.block(nodes) if self.mode == "block" else self.counter_seed(nodes)

    # ----- dynamics -------------------------------------------------------
    def _spread(self, frontier, live, allowed):
        g = self.g
        new = []
        for u in frontier:
            lo, hi = g.indptr[u], g.indptr[u + 1]
            for v, ok in zip(g.indices[lo:hi], live[lo:hi]):
                if ok and allowed[v]:
                    allowed[v] = False
                    new.append(int(v))
        return np.array(new, dtype=np.int64)

    def step(self, rho: np.ndarray | None = None) -> np.ndarray:
        """Advance one round; returns the newly activated bad nodes.

        ``rho`` (per-node factors in (0, 1]) throttles the bad cascade's edges
        *into* each node for this round only: the edge (u, v) transmits iff
        u_e < rho[v] * min(1, kappa p0(u, v)), using the episode's fixed
        uniforms, so throttling is coupled with the unthrottled trajectory.
        """
        allowed = ~(self.active | self.good | self.blocked)
        if rho is not None:
            live_bad = self.u < rho[self.g.indices] * self.p_bad
        else:
            live_bad = self.live_bad
        # good campaign moves first so it wins ties (Budak et al. semantics)
        if self.mode == "counter":
            gf = np.concatenate([self.good_frontier, np.array(self.pending_good, dtype=np.int64)])
            self.pending_good = []
            new_good = self._spread(gf, self.live_good, allowed)
            self.good[new_good] = True
            self.good_frontier = new_good
        new_bad = self._spread(self.frontier, live_bad, allowed)
        self.active[new_bad] = True
        self.frontier = new_bad
        self.t += 1
        self.history.append(new_bad)
        return new_bad

    @property
    def alive(self) -> bool:
        return len(self.frontier) > 0 or (self.mode == "counter" and (len(self.good_frontier) > 0 or bool(self.pending_good)))

    @property
    def harm(self) -> int:
        return int(self.active.sum())

    def counterfactual_harm(self, max_rounds: int | None = None) -> int:
        """Bad set size with no intervention after ``max_rounds`` rounds (all rounds if None):
        reachability on the bad live edges, truncated at that depth."""
        seen = np.zeros(self.n, dtype=bool)
        seen[self.seeds] = True
        frontier = list(self.seeds)
        g = self.g
        depth = 0
        while frontier and (max_rounds is None or depth < max_rounds):
            depth += 1
            nxt = []
            for u in frontier:
                lo, hi = g.indptr[u], g.indptr[u + 1]
                for v, ok in zip(g.indices[lo:hi], self.live_bad[lo:hi]):
                    if ok and not seen[v]:
                        seen[v] = True
                        nxt.append(int(v))
            frontier = nxt
        return int(seen.sum())


def exposure(g: CSRGraph, frontier: np.ndarray, inactive: np.ndarray, kappas: np.ndarray, rho: np.ndarray | None = None):
    """Exposed nodes and their log(1 - q^kappa) for every kappa in ``kappas``.

    Returns ``(exposed, log1mq)`` with ``log1mq`` of shape ``(len(kappas), len(exposed))``
    (column 0 must correspond to kappa = 1, the baseline). ``rho`` are per-node
    throttle factors applied to the edges into each exposed node (predictable,
    so the likelihood ratios remain exact).
    """
    if len(frontier) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((len(kappas), 0))
    tg, pr = [], []
    for u in frontier:
        lo, hi = g.indptr[u], g.indptr[u + 1]
        tg.append(g.indices[lo:hi])
        pr.append(g.weights[lo:hi])
    tg = np.concatenate(tg)
    pr = np.concatenate(pr)
    keep = inactive[tg]
    tg, pr = tg[keep], pr[keep]
    exposed, inv = np.unique(tg, return_inverse=True)
    out = np.zeros((len(kappas), len(exposed)))
    scale = rho[tg] if rho is not None else 1.0
    for i, k in enumerate(kappas):
        l = np.log1p(-np.minimum(1.0 - 1e-12, scale * np.minimum(1.0, k * pr)))
        np.add.at(out[i], inv, l)
    return exposed, out


def exposed_nodes(g: CSRGraph, frontier: np.ndarray, inactive: np.ndarray) -> np.ndarray:
    if len(frontier) == 0:
        return np.zeros(0, dtype=np.int64)
    tg = np.concatenate([g.indices[g.indptr[u]: g.indptr[u + 1]] for u in frontier])
    tg = tg[inactive[tg]]
    return np.unique(tg)
