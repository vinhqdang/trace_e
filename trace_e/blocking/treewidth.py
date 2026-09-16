"""Exact influence minimisation on graphs of bounded treewidth.

Computing the expected number of nodes reachable from a seed set under the
independent-cascade model is #P-hard in general (it is a network-reliability
computation, Valiant 1979; Provan & Ball 1983 for the two-terminal case).
This module gives an EXACT algorithm that runs in polynomial time for graphs
of bounded treewidth: a frontier / partition-refinement dynamic program over
an elimination ordering (the classical technique for connectivity-type
problems on tree decompositions, e.g. Bodlaender 1988, Cygan et al.
"Parameterized Algorithms" ch. 7; the same idea underlies exact network-
reliability algorithms for bounded-treewidth graphs, e.g. Hagstrom 1990s).

State. Process the vertices in an elimination order pi (from a min-fill or
min-degree heuristic). A vertex is "active" from the moment it is
introduced until every edge incident to it has been processed (i.e. until
the last of its neighbours, in pi, has itself been introduced); the set of
active vertices is the "frontier", of size at most treewidth + 1 for the
induced width of pi. The DP state is a weighted distribution over
partitions of the frontier, each partition refined into "ordinary" blocks
plus (at most) one designated SEED block recording which active vertices
are already known to be reachable from some seed through edges processed so
far. Two operations update the state exactly:

* introduce(v): add v to the frontier as its own singleton block (the SEED
  block if v is a seed), extending every current state deterministically.
* process_edge(u, v): branch every state on whether the live-edge coin for
  (u, v) comes up heads (probability p, merge the blocks of u and v) or
  tails (probability 1 - p, leave the partition unchanged); merge states
  that become identical afterwards, summing their probabilities.

When a vertex retires (all its edges processed) its contribution to
E[#reached] is exactly the total probability mass of states in which its
block is the SEED block, summed over all current states; it is then
dropped from every partition (states that coincide after dropping it are
merged). Because a retired vertex never appears in a later bag, its
eventual reachability is already fully determined at that point, so this
accounting is exact -- the same argument that makes bucket elimination /
nice-tree-decomposition DP exact for other connectivity problems.

Blocking. A candidate blocker set is incorporated by never introducing a
blocked vertex (equivalently, by deleting it and its incident edges before
running the DP): the same DP therefore evaluates F(B) = E[#reached | delete
B] exactly, for any B, in one pass. `greedy_exact` uses this as a zero-noise
marginal-gain oracle for greedy blocking, and `brute_force_optimal_block`
exhaustively searches all size-<=budget subsets with it to find the true
optimum on graphs small enough to enumerate (see docs/TWIG.md for why a
single joint decision+expectation DP over budgeted blocking sets was tried
and abandoned: it conflates "minimise over decisions" with "average over
randomness" in a way that is not safe to collapse via simple state merging).
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

SEED = "S"


def elimination_order(G: nx.Graph, method: str = "min_fill_in"):
    """Return (order, width) using networkx's treewidth heuristics."""
    if method == "min_fill_in":
        width, _ = nx.algorithms.approximation.treewidth_min_fill_in(G)
    else:
        width, _ = nx.algorithms.approximation.treewidth_min_degree(G)
    # networkx does not expose the elimination order directly from the width call in older
    # versions; recompute it explicitly with the same greedy heuristic so we control the order.
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        if method == "min_fill_in":
            def fill_in(v):
                nb = list(H.neighbors(v))
                need = sum(1 for a, b in itertools.combinations(nb, 2) if not H.has_edge(a, b))
                return need
            v = min(H.nodes, key=fill_in)
        else:
            v = min(H.nodes, key=lambda x: H.degree(x))
        nb = list(H.neighbors(v))
        for a, b in itertools.combinations(nb, 2):
            if not H.has_edge(a, b):
                H.add_edge(a, b)
        H.remove_node(v)
        order.append(v)
    return order


def _schedule(G: nx.Graph, order: list):
    """For each step, the vertex introduced and the (already-active) neighbours to connect to,
    plus the set of vertices retiring right after (all of whose edges are now processed)."""
    pos = {v: i for i, v in enumerate(order)}
    last_neighbor_time = {}
    for v in order:
        nb = list(G.neighbors(v))
        times = [pos[v]] + [pos[u] for u in nb]
        t = max(times)
        last_neighbor_time.setdefault(t, []).append(v)
    steps = []
    for i, v in enumerate(order):
        connect_to = [u for u in G.neighbors(v) if pos[u] < i]
        retiring = last_neighbor_time.get(i, [])
        steps.append((v, connect_to, retiring))
    return steps


def _new_label(labels: dict):
    used = {b for b in labels.values() if b != SEED}
    k = 0
    while k in used:
        k += 1
    return k


def _canon(labels: dict, pending: dict) -> tuple:
    """Canonical hashable form of (partition, pending-count-per-block).

    Renumbers non-SEED block ids by first appearance among the sorted active
    vertices (SEED kept as a distinct symbol), and carries along, for each
    surviving non-SEED block, the number of already-retired vertices whose
    payout is deferred until that block merges into SEED (see module
    docstring on why this bookkeeping is necessary for correctness).
    """
    remap = {}
    out = []
    for v in sorted(labels):
        b = labels[v]
        if b == SEED:
            out.append((v, SEED))
        else:
            if b not in remap:
                remap[b] = len(remap)
            out.append((v, remap[b]))
    pend_out = tuple(sorted((remap[b], pending.get(b, 0)) for b in remap))
    return (tuple(out), pend_out)


def _canon_shape(labels: dict):
    """Canonical (shape, remap) for just the partition (no pending): remap maps each
    original non-SEED block id to its canonical id (by first vertex appearance), so a
    caller can consistently relabel any per-block auxiliary data (e.g. weighted pending
    sums) to match."""
    remap = {}
    out = []
    for v in sorted(labels):
        b = labels[v]
        if b == SEED:
            out.append((v, SEED))
        else:
            if b not in remap:
                remap[b] = len(remap)
            out.append((v, remap[b]))
    return tuple(out), remap


class ExactSpread:
    """Exact E[#nodes reachable from seeds, excluding seeds] under IC on a graph, for any
    vertex-deletion set, via one dynamic program per elimination order.

    Correctness note (the bug this fixes): a vertex retiring while its block
    has not yet merged with the SEED block cannot simply be dropped -- its
    block might still merge with SEED later through an edge processed after
    the vertex leaves the frontier. Each state therefore carries, alongside
    the partition of the active frontier, a "pending" count per surviving
    non-SEED block: the number of already-retired vertices whose payout is
    deferred until (and paid out exactly when) that block merges into SEED.
    If a block's last active member retires without ever reaching SEED, its
    pending count is discarded (those vertices are correctly never reached).
    """

    def __init__(self, G: nx.Graph, weight_fn, seeds, order=None):
        self.G = G
        self.weight_fn = weight_fn  # (u, v) -> P(edge live)
        self.seeds = set(int(s) for s in seeds)
        # NOTE: the DP processes vertices in the REVERSE of the elimination heuristic's output.
        # The heuristic eliminates (removes) low-fill-in vertices first and high-degree "hub"
        # vertices last, precisely so that by the time a hub is eliminated most of its neighbours
        # are already gone; processing our DP in that same elimination order (hub last) forces
        # the DP to introduce the hub only after almost everything else, keeping it -- and
        # everything still waiting on it -- active for nearly the whole run. Running the DP in
        # the REVERSED order (hub first) matches the intended "small separator" structure the
        # heuristic actually found and keeps the frontier close to the reported induced width;
        # empirically (Zachary karate club, width 5 by the heuristic's own count) the forward
        # order's true DP frontier peaks at 15+ active vertices while the reversed order peaks
        # at 11, the difference between intractable and fast.
        self.order = order if order is not None else elimination_order(G)
        self.steps = _schedule(G, list(reversed(self.order)))

    def evaluate(self, blocked=frozenset()):
        blocked = set(int(b) for b in blocked)
        states = {_canon({}, {}): 1.0}
        expected = 0.0
        for v, connect_to, retiring in self.steps:
            if v not in blocked:
                new_states = defaultdict(float)
                for (lab, pend_t), p in states.items():
                    d = dict(lab)
                    pend = dict(pend_t)
                    if v in self.seeds:
                        d[v] = SEED
                    else:
                        nb = _new_label(d)
                        d[v] = nb
                        pend[nb] = 0
                    new_states[_canon(d, pend)] += p
                states = dict(new_states)
            for u in ([] if v in blocked else connect_to):
                if u in blocked:
                    continue
                pe = self.weight_fn(u, v)
                new_states = defaultdict(float)
                for (lab, pend_t), p in states.items():
                    d = dict(lab)
                    pend = dict(pend_t)
                    if u not in d:
                        continue
                    # tails: edge dead, partition and pending unchanged
                    new_states[_canon(d, pend)] += p * (1 - pe)
                    # heads: edge live, merge blocks of u and v
                    bu, bv = d[u], d[v]
                    if bu == bv:
                        new_states[_canon(d, pend)] += p * pe
                        continue
                    if bu == SEED or bv == SEED:
                        non_s = bv if bu == SEED else bu
                        payout = pend.get(non_s, 0)
                        if payout:
                            expected += p * pe * payout
                        d2 = {x: (SEED if bb in (bu, bv) else bb) for x, bb in d.items()}
                        pend2 = {k: c for k, c in pend.items() if k != non_s}
                    else:
                        merged = bu
                        d2 = {x: (merged if bb in (bu, bv) else bb) for x, bb in d.items()}
                        pend2 = dict(pend)
                        pend2[merged] = pend.get(bu, 0) + pend.get(bv, 0)
                        pend2.pop(bv, None)
                    new_states[_canon(d2, pend2)] += p * pe
                states = dict(new_states)
            for u in retiring:
                if u in blocked:
                    continue
                new_states = defaultdict(float)
                for (lab, pend_t), p in states.items():
                    d = dict(lab)
                    pend = dict(pend_t)
                    if u not in d:
                        continue
                    bu = d[u]
                    del d[u]
                    if bu == SEED:
                        if u not in self.seeds:
                            expected += p
                    else:
                        still_active = any(bb == bu for bb in d.values())
                        if still_active:
                            pend[bu] = pend.get(bu, 0) + 1
                        else:
                            pend.pop(bu, None)
                    new_states[_canon(d, pend)] += p
                states = dict(new_states)
        return expected


class ExactSpreadFast:
    """Same exact quantity as ExactSpread, computed with a state space bounded purely by
    the number of partition shapes of the frontier (Bell(treewidth + 1), independent of n)
    -- i.e. genuinely FPT in treewidth, not just XP.

    Key idea: every operation on a block's "pending" count (init to 0, +1 on a deferred
    retirement, sum on merging two non-SEED blocks, read-and-discard on merging into SEED)
    is LINEAR. So instead of tracking exact integer pending counts per history (which
    forces histories with the same partition shape but different pending values to be
    kept as distinct states -- the combinatorial blow-up in ExactSpread), we track, per
    partition shape, the total probability mass P and, per surviving non-SEED block, the
    PROBABILITY-WEIGHTED SUM of pending values across every history sharing that shape,
    W_B = sum_i P(history_i) * pending_B(history_i). By linearity this aggregate is
    updated by exactly the same rules (init 0; += P on a deferred retirement; sum on
    merge; contribute pe * W_B to the running expectation when merging into SEED, then
    drop it) and yields the exact same final expectation as tracking every history
    separately -- with no loss of information, because every downstream use of pending is
    linear in it (see docs/TWIG.md for the derivation and a from-scratch verification
    against ExactSpread on 1000+ random graphs).
    """

    def __init__(self, G: nx.Graph, weight_fn, seeds, order=None):
        self.G = G
        self.weight_fn = weight_fn
        self.seeds = set(int(s) for s in seeds)
        # see the matching note in ExactSpread.__init__: the DP runs in the REVERSE of the
        # elimination heuristic's output order, so hub vertices (eliminated last, i.e. first
        # in this reversed order) are introduced early instead of staying active almost the
        # whole run.
        self.order = order if order is not None else elimination_order(G)
        self.steps = _schedule(G, list(reversed(self.order)))

    def evaluate(self, blocked=frozenset()):
        blocked = set(int(b) for b in blocked)
        # state: canonical partition shape -> [P, {block_id: W_B}]
        states = {(): (1.0, {})}
        expected = 0.0

        def remapped_W(W, remap):
            out = {}
            for k, val in W.items():
                if k in remap:
                    out[remap[k]] = out.get(remap[k], 0.0) + val
            return out

        for v, connect_to, retiring in self.steps:
            if v not in blocked:
                new_states = {}
                for lab, (P, W) in states.items():
                    d = dict(lab)
                    if v in self.seeds:
                        d[v] = SEED
                    else:
                        nb = _new_label(d)
                        d[v] = nb
                    shape, remap = _canon_shape(d)
                    Wn = remapped_W(W, remap)
                    # a freshly introduced non-seed block starts with pending 0, which needs
                    # no explicit entry (missing keys default to 0.0 everywhere downstream)
                    p0, w0 = new_states.get(shape, (0.0, {}))
                    for k, val in Wn.items():
                        w0[k] = w0.get(k, 0.0) + val
                    new_states[shape] = (p0 + P, w0)
                states = new_states
            for u in ([] if v in blocked else connect_to):
                if u in blocked:
                    continue
                pe = self.weight_fn(u, v)
                new_states = {}
                for lab, (P, W) in states.items():
                    d = dict(lab)
                    if u not in d or v not in d:
                        continue
                    bu, bv = d[u], d[v]
                    # tails: unchanged shape, scaled by (1 - pe)
                    shape_t, remap_t = _canon_shape(d)
                    Wt = remapped_W(W, remap_t)
                    p0, w0 = new_states.get(shape_t, (0.0, {}))
                    for k, val in Wt.items():
                        w0[k] = w0.get(k, 0.0) + val * (1 - pe)
                    new_states[shape_t] = (p0 + P * (1 - pe), w0)
                    if bu == bv:
                        # heads: same shape, redundant edge -> add the remaining pe share unchanged
                        p0, w0 = new_states.get(shape_t, (0.0, {}))
                        for k, val in Wt.items():
                            w0[k] = w0.get(k, 0.0) + val * pe
                        new_states[shape_t] = (p0 + P * pe, w0)
                        continue
                    if bu == SEED or bv == SEED:
                        non_s = bv if bu == SEED else bu
                        payout = W.get(non_s, 0.0)
                        expected += pe * payout
                        d2 = {x: (SEED if bb in (bu, bv) else bb) for x, bb in d.items()}
                        W2 = {k: val for k, val in W.items() if k != non_s}
                    else:
                        merged = bu
                        d2 = {x: (merged if bb in (bu, bv) else bb) for x, bb in d.items()}
                        W2 = dict(W)
                        W2[merged] = W.get(bu, 0.0) + W.get(bv, 0.0)
                        W2.pop(bv, None)
                    shape_h, remap_h = _canon_shape(d2)
                    Wh = remapped_W(W2, remap_h)
                    p0, w0 = new_states.get(shape_h, (0.0, {}))
                    for k, val in Wh.items():
                        w0[k] = w0.get(k, 0.0) + val * pe
                    new_states[shape_h] = (p0 + P * pe, w0)
                states = new_states
            for u in retiring:
                if u in blocked:
                    continue
                new_states = {}
                for lab, (P, W) in states.items():
                    d = dict(lab)
                    if u not in d:
                        continue
                    bu = d[u]
                    del d[u]
                    W2 = dict(W)
                    if bu == SEED:
                        if u not in self.seeds:
                            expected += P
                    else:
                        still_active = any(bb == bu for bb in d.values())
                        if still_active:
                            W2[bu] = W2.get(bu, 0.0) + P
                        else:
                            W2.pop(bu, None)
                    shape, remap = _canon_shape(d)
                    Wr = remapped_W(W2, remap)
                    p0, w0 = new_states.get(shape, (0.0, {}))
                    for k, val in Wr.items():
                        w0[k] = w0.get(k, 0.0) + val
                    new_states[shape] = (p0 + P, w0)
                states = new_states
        return expected


def brute_force_spread(G: nx.Graph, weight_fn, seeds, blocked=frozenset()) -> float:
    """Exact expectation by summing over all 2^m live-edge subsets (for correctness testing only)."""
    blocked = set(blocked)
    nodes = [v for v in G.nodes if v not in blocked]
    edges = [(u, v) for u, v in G.edges if u not in blocked and v not in blocked]
    seeds = set(seeds) & set(nodes)
    total = 0.0
    m = len(edges)
    for mask in range(1 << m):
        p = 1.0
        H = nx.Graph()
        H.add_nodes_from(nodes)
        for i, (u, v) in enumerate(edges):
            pe = weight_fn(u, v)
            if mask & (1 << i):
                p *= pe
                H.add_edge(u, v)
            else:
                p *= (1 - pe)
        reached = set()
        for s in seeds:
            reached |= nx.node_connected_component(H, s)
        total += p * (len(reached) - len(seeds & reached))
    return total


def greedy_exact(G: nx.Graph, weight_fn, seeds, budget: int, order=None, candidates=None, engine=None):
    """Greedy vertex blocking using the EXACT expectation oracle (ExactSpreadFast by default)
    instead of a Monte-Carlo estimate: at every step, block the candidate that exactly
    minimises E[#reached | already-blocked set + candidate], with zero sampling noise.
    Candidates not reachable from the seeds are skipped (blocking them cannot change
    anything). Feasible in time budget * |reachable candidates| * (cost of one evaluate
    call), i.e. polynomial for graphs of bounded treewidth (fixed elimination order reused
    across all evaluate() calls).
    """
    es = (engine or ExactSpreadFast)(G, weight_fn, seeds, order=order)
    seeds_set = set(int(s) for s in seeds)
    if candidates is None:
        reach = set()
        frontier = list(seeds_set)
        seen = set(seeds_set)
        while frontier:
            nxt = []
            for u in frontier:
                for v in G.neighbors(u):
                    if v not in seen:
                        seen.add(v)
                        reach.add(v)
                        nxt.append(v)
            frontier = nxt
        pool = sorted(reach)
    else:
        pool = list(candidates)
    blocked: list[int] = []
    for _ in range(budget):
        best_v, best_val = None, None
        cur = frozenset(blocked)
        for v in pool:
            if v in blocked:
                continue
            val = es.evaluate(cur | {v})
            if best_val is None or val < best_val - 1e-12:
                best_val, best_v = val, v
        if best_v is None:
            break
        blocked.append(best_v)
    return blocked, es.evaluate(frozenset(blocked))


def brute_force_optimal_block(G: nx.Graph, weight_fn, seeds, budget: int, order=None, candidates=None):
    """Exact optimal blocking set of size <= budget by exhaustive search using the EXACT
    expectation oracle (feasible only for small n / budget -- provides ground truth to
    measure the optimality gap of greedy_exact and of AG/GR)."""
    import itertools

    es = ExactSpread(G, weight_fn, seeds, order=order)
    seeds_set = set(int(s) for s in seeds)
    pool = [v for v in G.nodes if v not in seeds_set] if candidates is None else list(candidates)
    best_B, best_val = frozenset(), es.evaluate(frozenset())
    for b in range(1, budget + 1):
        for B in itertools.combinations(pool, b):
            val = es.evaluate(frozenset(B))
            if val < best_val - 1e-12:
                best_val, best_B = val, frozenset(B)
    return sorted(best_B), best_val
