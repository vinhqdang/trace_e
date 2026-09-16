"""Independent-cascade models on a CSR graph.

* :func:`ic_spread` - single-campaign independent cascade from ``seeds`` with
  an optional set of blocked (removed) nodes.
* :func:`competitive_ic_spread` - two campaigns (bad starts at ``bad_seeds``,
  good at ``good_seeds``); each node adopts whichever campaign reaches it
  first, ties resolved in favour of the good campaign (Budak et al. 2011,
  "campaign-oblivious" independent cascade with good-campaign tie priority).

Edge activation probabilities are stored per directed edge in the CSR
``weights`` array (the loaders set them with :func:`set_edge_probabilities`).
Both simulators are synchronous (discrete rounds) and return the final set of
bad-adopting nodes as a boolean array.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..graphs import CSRGraph


def set_edge_probabilities(g: CSRGraph, model: str = "wc", p: float = 0.1, seed: int = 0) -> CSRGraph:
    """Return a copy of ``g`` whose weights are IC activation probabilities.

    ``wc``: weighted cascade, p(u->v) = 1/deg(v); ``const``: p(u->v) = p;
    ``tri``: trivalency, p drawn uniformly from {0.1, 0.01, 0.001}.
    """
    deg = g.degree()
    w = np.empty_like(g.weights)
    if model == "wc":
        w[:] = 1.0 / np.maximum(deg[g.indices], 1)
    elif model == "const":
        w[:] = p
    elif model == "tri":
        rng = np.random.default_rng(seed)
        w[:] = rng.choice([0.1, 0.01, 0.001], size=len(w))
    elif model == "bimodal":  # strong ties (reliable relay) and weak ties (uncertain exposure); p = fraction of strong ties
        rng = np.random.default_rng(seed)
        strong = rng.random(len(w)) < p
        w[:] = np.where(strong, 0.9, 0.05)
    else:
        raise ValueError(model)
    return CSRGraph(n=g.n, indptr=g.indptr, indices=g.indices, weights=w)


def ic_spread(g: CSRGraph, seeds, blocked: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    active = np.zeros(g.n, dtype=bool)
    if blocked is None:
        blocked = np.zeros(g.n, dtype=bool)
    frontier = deque()
    for s in seeds:
        if not blocked[s]:
            active[s] = True
            frontier.append(int(s))
    while frontier:
        u = frontier.popleft()
        lo, hi = g.indptr[u], g.indptr[u + 1]
        nb = g.indices[lo:hi]
        pr = g.weights[lo:hi]
        hit = rng.random(len(nb)) < pr
        for v in nb[hit]:
            if not active[v] and not blocked[v]:
                active[v] = True
                frontier.append(int(v))
    return active


def competitive_ic_spread(g: CSRGraph, bad_seeds, good_seeds, rng: np.random.Generator, good_delay: int = 0) -> np.ndarray:
    """Return boolean array of nodes that end up adopting the bad campaign.

    ``good_delay`` rounds may elapse before the good campaign starts (the
    "detection delay" of an intervention). Live-edge semantics: each edge is
    sampled once and shared by both campaigns.
    """
    BAD, GOOD = 1, 2
    state = np.zeros(g.n, dtype=np.int8)
    bad_front = [int(s) for s in bad_seeds]
    good_front = []
    for s in bad_front:
        state[s] = BAD
    pending_good = [int(s) for s in good_seeds]
    t = 0
    while bad_front or good_front or pending_good:
        if t == good_delay:
            for s in pending_good:
                if state[s] == 0:
                    state[s] = GOOD
                    good_front.append(s)
            pending_good = []
        elif t > good_delay:
            pending_good = []
        new_good, new_bad = [], []
        # good campaign first so it wins ties in the same round
        for u in good_front:
            lo, hi = g.indptr[u], g.indptr[u + 1]
            nb = g.indices[lo:hi]
            hit = rng.random(hi - lo) < g.weights[lo:hi]
            for v in nb[hit]:
                if state[v] == 0:
                    state[v] = GOOD
                    new_good.append(int(v))
        for u in bad_front:
            lo, hi = g.indptr[u], g.indptr[u + 1]
            nb = g.indices[lo:hi]
            hit = rng.random(hi - lo) < g.weights[lo:hi]
            for v in nb[hit]:
                if state[v] == 0:
                    state[v] = BAD
                    new_bad.append(int(v))
        good_front, bad_front = new_good, new_bad
        t += 1
        if t > g.n + good_delay + 2:
            break
    return state == BAD


def estimate_bad_spread(g: CSRGraph, bad_seeds, intervention, mode: str, n_mc: int, seed: int, good_delay: int = 0) -> float:
    """Monte-Carlo estimate of the expected number of bad adopters.

    ``mode`` is ``"block"`` (intervention nodes are removed) or ``"counter"``
    (intervention nodes seed a competing good campaign).
    """
    rng = np.random.default_rng(seed)
    tot = 0.0
    if mode == "block":
        blocked = np.zeros(g.n, dtype=bool)
        blocked[list(intervention)] = True
        for _ in range(n_mc):
            tot += ic_spread(g, bad_seeds, blocked, rng).sum()
    else:
        for _ in range(n_mc):
            tot += competitive_ic_spread(g, bad_seeds, list(intervention), rng, good_delay).sum()
    return tot / n_mc


def sample_live_edge_reach(g: CSRGraph, bad_seeds, n_samples: int, seed: int) -> list[np.ndarray]:
    """Sample live-edge graphs and return, for each, the BFS order of bad-reachable nodes.

    Used by the greedy blocker to evaluate marginal gains cheaply: for a
    sample, removing node set B leaves reachable exactly the nodes reachable
    from the seeds in the live-edge graph minus B.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        live = rng.random(len(g.indices)) < g.weights
        out.append(live)
    return out
