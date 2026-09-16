"""Influence minimisation by vertex blocking (IMIN) under the independent cascade model.

Baselines re-implemented from the literature on a shared sample set:

* ``ag``     AdvancedGreedy (Xie et al., ICDE 2023 / IJOC 2024): greedy on
             single-node gains, each gain = average dominator-subtree size over
             theta sampled live-edge graphs, recomputed after every pick.
* ``gr``     GreedyReplace (Xie et al. 2024): first greedily block out-neighbours
             of the seeds, then replace blockers in reverse order by the best
             node overall, stopping when a blocker replaces itself.
* ``lsbm``   Lower-bound maximisation of SandIMIN (Wang et al., 2024): greedy
             maximum coverage of the "critical path" (dominator chain) sets
             computed once on the samples; a (1-1/e)-approximation of the
             submodular lower bound of the objective.

Proposed:

* ``isocut`` ISOCUT: cost-sensitive greedy over two kinds of moves priced
             exactly on the samples, single nodes (cost 1, gain = dominator
             subtree) and *isolation* of a seed (cost = its unblocked
             out-neighbours, gain = its dominator subtree minus one), chosen by
             gain per unit budget, with the sample structures recomputed only
             where the chosen nodes were reachable.
* ``isocut+`` best of ISOCUT and AG on the same samples (inherits AG's
             guarantees while keeping ISOCUT's worst-case advantage).

All algorithms share :class:`SampleSet`, so quality differences come from
the selection rule, not from sampling noise.
"""
from __future__ import annotations

import time

import numpy as np

from ..graphs import CSRGraph
from ..sequential.dominators import dominator_subtree_sizes, reachable_dag


class SampleSet:
    """theta live-edge samples restricted to what the seeds can reach, with cached dominator data."""

    def __init__(self, g: CSRGraph, seeds, theta: int, rng: np.random.Generator, forbidden: np.ndarray | None = None, lives=None):
        self.g = g
        self.seeds = [int(s) for s in seeds]
        self.seed_mask = np.zeros(g.n, dtype=bool)
        self.seed_mask[self.seeds] = True
        self.lives = list(lives) if lives is not None else [rng.random(len(g.indices)) < g.weights for _ in range(theta)]
        self.theta = len(self.lives)
        self.forbidden = np.zeros(g.n, dtype=bool) if forbidden is None else forbidden.copy()
        self.forbidden[self.seeds] = False
        self.blocked: list[int] = []
        self.cache = [None] * theta
        self.dirty = [True] * theta
        self.n_dom = 0  # dominator computations performed (cost accounting)
        self.t_dom = 0.0

    # ----- dominator data --------------------------------------------------
    def refresh(self):
        t0 = time.perf_counter()
        for i in range(self.theta):
            if not self.dirty[i]:
                continue
            order, preds, index = reachable_dag(self.g, self.lives[i], self.seeds, self.forbidden)
            sizes = dominator_subtree_sizes(order, preds)
            reach = np.zeros(self.g.n, dtype=bool)
            if order:
                reach[order] = True
            self.cache[i] = (np.array(order, dtype=np.int64), index, sizes, reach)
            self.dirty[i] = False
            self.n_dom += 1
        self.t_dom += time.perf_counter() - t0

    def single_gains(self) -> np.ndarray:
        """Average nodes saved by blocking each node alone (0 for seeds, blocked and unreachable nodes)."""
        acc = np.zeros(self.g.n)
        for order, index, sizes, reach in self.cache:
            if len(order):
                acc[order] += sizes[index[order]]
        acc[self.seed_mask] = 0
        acc[self.forbidden] = 0
        return acc / self.theta

    def isolation_gains(self) -> dict[int, float]:
        """Average nodes saved by cutting all out-edges of each seed."""
        acc = {s: 0.0 for s in self.seeds}
        for order, index, sizes, reach in self.cache:
            for s in self.seeds:
                if reach[s]:
                    acc[s] += sizes[index[s]] - 1
        return {s: v / self.theta for s, v in acc.items()}

    def isolation_cost(self, s: int) -> list[int]:
        g = self.g
        nb = np.unique(g.indices[g.indptr[s]: g.indptr[s + 1]])
        return [int(v) for v in nb if not self.forbidden[v] and not self.seed_mask[v]]

    def expected_reach(self) -> float:
        """Average number of non-seed nodes still reachable (the objective, on the samples)."""
        tot = 0.0
        for order, index, sizes, reach in self.cache:
            tot += len(order) - int(reach[self.seeds].sum())
        return tot / self.theta

    def evaluate(self, nodes) -> float:
        """Average non-seed nodes reachable on the samples when ``nodes`` are blocked in addition to the current set."""
        from ..sequential.containers import reach_count
        forb = self.forbidden.copy()
        forb[[int(v) for v in nodes]] = True
        return float(np.mean([reach_count(self.g, lv, self.seeds, forb) for lv in self.lives]))

    # ----- updates ---------------------------------------------------------
    def block(self, nodes):
        nodes = [int(v) for v in nodes if not self.forbidden[v] and not self.seed_mask[v]]
        if not nodes:
            return []
        self.forbidden[nodes] = True
        self.blocked.extend(nodes)
        for i in range(self.theta):
            if self.cache[i] is None or self.dirty[i]:
                continue
            reach = self.cache[i][3]
            if reach[nodes].any():
                self.dirty[i] = True
        return nodes

    def unblock(self, v: int):
        self.forbidden[v] = False
        self.blocked.remove(v)
        self.dirty = [True] * self.theta

    def snapshot(self):
        return (self.forbidden.copy(), list(self.blocked), [c for c in self.cache], list(self.dirty))

    def restore(self, snap):
        self.forbidden, self.blocked, self.cache, self.dirty = snap[0].copy(), list(snap[1]), list(snap[2]), list(snap[3])


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def advanced_greedy(S: SampleSet, budget: int) -> list[int]:
    for _ in range(budget):
        S.refresh()
        gains = S.single_gains()
        v = int(np.argmax(gains))
        if gains[v] <= 0:
            break
        S.block([v])
    return list(S.blocked)


def greedy_replace(S: SampleSet, budget: int) -> list[int]:
    g = S.g
    cand = set()
    for s in S.seeds:
        cand.update(S.isolation_cost(s))
    cand = np.array(sorted(cand), dtype=np.int64)
    order_ins = []
    for _ in range(min(len(cand), budget)):
        S.refresh()
        gains = S.single_gains()
        c = cand[~S.forbidden[cand]]
        if len(c) == 0:
            break
        v = int(c[np.argmax(gains[c])])
        S.block([v])
        order_ins.append(v)
    # fill any leftover budget greedily (GR assumes |N_out| >= b; this keeps the comparison fair)
    while len(S.blocked) < budget:
        S.refresh()
        gains = S.single_gains()
        v = int(np.argmax(gains))
        if gains[v] <= 0:
            break
        S.block([v])
        order_ins.append(v)
    for u in reversed(order_ins):
        S.unblock(u)
        S.refresh()
        gains = S.single_gains()
        x = int(np.argmax(gains))
        S.block([x])
        if x == u:
            break
    return list(S.blocked)


def lsbm(S: SampleSet, budget: int) -> list[int]:
    """Greedy max coverage of dominator chains (SandIMIN lower bound), samples fixed once."""
    S.refresh()
    g = S.g
    # per sample: idom-based chains are implicit; marginal coverage of v = uncovered nodes in v's subtree.
    covered = [np.zeros(g.n, dtype=bool) for _ in range(S.theta)]
    # children lists in the dominator tree per sample, from idom
    trees = []
    for order, index, sizes, reach in S.cache:
        if len(order) == 0:
            trees.append(None)
            continue
        # recompute idom via a second pass (dominator_subtree_sizes does not expose it); cheap for our sizes
        preds_order, preds, index2 = reachable_dag(g, S.lives[len(trees)], S.seeds, S.forbidden)
        trees.append((np.array(preds_order, dtype=np.int64), index2, _idoms(preds_order, preds)))
    chosen = []
    for _ in range(budget):
        gains = np.zeros(g.n)
        for i, tr in enumerate(trees):
            if tr is None:
                continue
            order, index, idom = tr
            # uncovered subtree sizes: accumulate from leaves (positions in RPO: children after parents)
            unc = (~covered[i][order]).astype(np.int64)
            for pos in range(len(order) - 1, -1, -1):
                d = idom[pos]
                if d is not None and d >= 0:
                    unc[d] += unc[pos]
            gains[order] += unc
        gains[S.seed_mask] = 0
        gains[S.forbidden] = 0
        v = int(np.argmax(gains))
        if gains[v] <= 0:
            break
        S.forbidden[v] = True
        S.blocked.append(v)
        chosen.append(v)
        # mark v's subtree covered in every sample where v is reachable
        for i, tr in enumerate(trees):
            if tr is None:
                continue
            order, index, idom = tr
            pv = index[v]
            if pv < 0:
                continue
            # descendants of pv in dominator tree: walk idom chains
            for pos in range(len(order)):
                q = pos
                while q is not None and q >= 0:
                    if q == pv:
                        covered[i][order[pos]] = True
                        break
                    q = idom[q]
    S.dirty = [True] * S.theta
    return chosen


def _idoms(order, preds):
    """Immediate dominators (positions; -1 = super-source) via the Cooper-Harvey-Kennedy iteration."""
    m = len(order)
    ROOT = -1
    idom = [None] * m

    def intersect(a, b):
        while a != b:
            if a == ROOT or b == ROOT:
                return ROOT
            while a > b:
                a = idom[a]
                if a == ROOT:
                    return ROOT
            while b > a:
                b = idom[b]
                if b == ROOT:
                    return ROOT
        return a

    changed = True
    while changed:
        changed = False
        for i in range(m):
            new = None
            for p in preds[i]:
                if p == ROOT:
                    cand = ROOT
                elif idom[p] is None:
                    continue
                else:
                    cand = p
                new = cand if new is None else intersect(new, cand)
            if new is not None and idom[i] != new:
                idom[i] = new
                changed = True
    return idom


def isocut(S: SampleSet, budget: int, replace: bool = False) -> list[int]:
    left = budget
    while left > 0:
        S.refresh()
        gains = S.single_gains()
        v = int(np.argmax(gains))
        best_ratio, move = float(gains[v]), [v] if gains[v] > 0 else []
        iso = S.isolation_gains()
        for s, G in iso.items():
            if G <= 0:
                continue
            grp = S.isolation_cost(s)
            if 0 < len(grp) <= left and G / len(grp) > best_ratio:
                best_ratio, move = G / len(grp), grp
        if not move:
            break
        S.block(move)
        left -= len(move)
    if replace:
        _replace_stage(S)
    return list(S.blocked)


def _replace_stage(S: SampleSet):
    for u in list(reversed(S.blocked)):
        S.unblock(u)
        S.refresh()
        gains = S.single_gains()
        x = int(np.argmax(gains))
        S.block([x])
        if x == u:
            break


def isocut_plus(S: SampleSet, budget: int) -> list[int]:
    snap = S.snapshot()
    a = isocut(S, budget)
    S.refresh()
    fa = S.expected_reach()
    S.restore(snap)
    b = advanced_greedy(S, budget)
    S.refresh()
    fb = S.expected_reach()
    return a if fa <= fb else b


ALGORITHMS = {"ag": advanced_greedy, "gr": greedy_replace, "lsbm": lsbm, "isocut": isocut,
              "isocut_r": lambda S, b: isocut(S, b, replace=True), "isocut+": isocut_plus}


def run_algorithm(name: str, g: CSRGraph, seeds, budget: int, theta: int = 100, seed: int = 0, forbidden=None, fill: bool = False):
    """Run an IMIN algorithm on fresh samples. ``fill=True`` tops up an under-spent plan with single-node greedy picks."""
    S = SampleSet(g, seeds, theta, np.random.default_rng(seed), forbidden=forbidden)
    t0 = time.perf_counter()
    B0 = set(S.blocked)
    B = [v for v in ALGORITHMS[name](S, budget) if v not in B0]
    if fill and len(B) < budget:
        advanced_greedy(S, budget - len(B))
        B = [v for v in S.blocked if v not in B0]
    return B, {"time_s": time.perf_counter() - t0, "dominator_calls": S.n_dom, "t_dominators": S.t_dom}


# ---------------------------------------------------------------------------
# CUTGREEDY: group moves priced by minimum vertex cuts
# ---------------------------------------------------------------------------

def _union_adjacency(S: SampleSet):
    """Adjacency (as CSR-like dict of arrays) of edges live in at least one sample, restricted to the region."""
    g = S.g
    anylive = np.zeros(len(g.indices), dtype=bool)
    for lv in S.lives:
        anylive |= lv
    return anylive


def min_vertex_cut(S: SampleSet, anylive: np.ndarray, target: int, limit: int) -> list[int] | None:
    """Minimum set of non-seed, unblocked vertices whose removal disconnects every seed from ``target``
    in the union-of-samples graph; unit vertex capacities, at most ``limit`` augmenting paths.

    Node splitting: v_in -> v_out with capacity 1 (infinite for seeds and target); edges u_out -> v_in.
    Returns None when more than ``limit`` vertex-disjoint paths exist (cut too expensive).
    """
    g = S.g
    n = g.n
    INF = 10**9
    # residual capacities: cap_node[v] for v_in->v_out (1), plus flow bookkeeping via path lists
    node_cap = np.ones(n, dtype=np.int64)
    node_cap[S.seed_mask] = INF
    node_cap[target] = INF
    node_flow = np.zeros(n, dtype=np.int64)  # flow through v_in -> v_out
    edge_flow: dict[tuple[int, int], int] = {}
    forbidden = S.forbidden
    src = S.seeds

    def bfs():
        # states: (v, side) side 0 = v_in, 1 = v_out ; super-source connects to seeds' in-nodes
        parent = {}
        q = []
        for s in src:
            if forbidden[s]:
                continue
            parent[(s, 0)] = None
            q.append((s, 0))
        head = 0
        while head < len(q):
            v, side = q[head]
            head += 1
            if side == 0:
                if node_flow[v] < node_cap[v] and (v, 1) not in parent:
                    parent[(v, 1)] = (v, 0)
                    q.append((v, 1))
                    if v == target:
                        return parent
                # residual backwards along incoming edges with positive flow: (u_out -> v_in) reverse
                # (handled from u_out side below through edge_flow lookups) - skipped for simplicity of
                # unit-capacity augmenting paths (we allow cancelling via node reverse edges only)
            else:
                lo, hi = g.indptr[v], g.indptr[v + 1]
                for e in range(lo, hi):
                    if not anylive[e]:
                        continue
                    w = int(g.indices[e])
                    if forbidden[w]:
                        continue
                    if edge_flow.get((v, w), 0) < 1 and (w, 0) not in parent:
                        parent[(w, 0)] = (v, 1)
                        q.append((w, 0))
                # reverse node edge: v_out -> v_in cancelling flow
                if node_flow[v] > 0 and (v, 0) not in parent:
                    parent[(v, 0)] = (v, 1)
                    q.append((v, 0))
        return None

    flow = 0
    while flow <= limit:
        parent = bfs()
        if parent is None:
            break
        # augment along path to (target, 1)
        cur = (target, 1)
        while parent[cur] is not None:
            prev = parent[cur]
            if prev[0] == cur[0]:
                if prev[1] == 0 and cur[1] == 1:
                    node_flow[cur[0]] += 1
                else:
                    node_flow[cur[0]] -= 1
            else:
                edge_flow[(prev[0], cur[0])] = edge_flow.get((prev[0], cur[0]), 0) + 1
            cur = prev
        flow += 1
        if flow > limit:
            return None
    # min cut: nodes v with v_in reachable in residual graph but v_out not
    parent = bfs_reach = None
    reach = set()
    q = []
    for s in src:
        if not forbidden[s]:
            reach.add((s, 0))
            q.append((s, 0))
    head = 0
    while head < len(q):
        v, side = q[head]
        head += 1
        if side == 0:
            if node_flow[v] < node_cap[v] and (v, 1) not in reach:
                reach.add((v, 1))
                q.append((v, 1))
        else:
            lo, hi = g.indptr[v], g.indptr[v + 1]
            for e in range(lo, hi):
                if not anylive[e]:
                    continue
                w = int(g.indices[e])
                if forbidden[w]:
                    continue
                if edge_flow.get((v, w), 0) < 1 and (w, 0) not in reach:
                    reach.add((w, 0))
                    q.append((w, 0))
            if node_flow[v] > 0 and (v, 0) not in reach:
                reach.add((v, 0))
                q.append((v, 0))
    cut = [v for (v, side) in reach if side == 0 and (v, 1) not in reach and not S.seed_mask[v] and v != target]
    return cut


def cutgreedy(S: SampleSet, budget: int, n_targets: int = 20, replace: bool = False) -> list[int]:
    """Cost-sensitive greedy over single-node moves and minimum-vertex-cut group moves.

    Targets for cuts are the nodes with the largest expected downstream mass
    among those that no single node can protect (single gain far below their
    value), plus the seeds' isolation cuts. Each cut is priced exactly on the
    samples (nodes disconnected from the seeds once the cut is removed).
    """
    g = S.g
    anylive = _union_adjacency(S)
    left = budget
    while left > 0:
        S.refresh()
        gains = S.single_gains()
        v = int(np.argmax(gains))
        best_ratio, move = float(gains[v]), ([v] if gains[v] > 0 else [])
        # candidate targets: reach frequency x out-mass, excluding nodes whose own blocking already captures their value
        freq = np.zeros(g.n)
        for order, index, sizes, reach in S.cache:
            freq += reach
        freq /= S.theta
        out_mass = np.array([g.weights[g.indptr[u]: g.indptr[u + 1]].sum() for u in range(g.n)]) if not hasattr(S, "_out_mass") else S._out_mass
        S._out_mass = out_mass
        value = freq * (1.0 + out_mass)
        value[S.seed_mask] = 0
        value[S.forbidden] = 0
        value[gains >= 0.5 * value.max()] = 0  # a good single move already covers these
        targets = [int(t) for t in np.argsort(-value)[:n_targets] if value[t] > 0]
        for t in targets:
            cut = min_vertex_cut(S, anylive, t, left)
            if not cut or len(cut) > left:
                continue
            # exact gain of the cut on the samples
            forb = S.forbidden.copy()
            forb[cut] = True
            saved = 0.0
            for i, (order, index, sizes, reach) in enumerate(S.cache):
                if not reach[cut].any():
                    continue
                o2, _, _ = reachable_dag(g, S.lives[i], S.seeds, forb)
                saved += len(order) - len(o2)
            ratio = saved / S.theta / len(cut)
            if ratio > best_ratio:
                best_ratio, move = ratio, cut
        # seed isolation as a cut move
        for s, G in S.isolation_gains().items():
            grp = S.isolation_cost(s)
            if G > 0 and 0 < len(grp) <= left and G / len(grp) > best_ratio:
                best_ratio, move = G / len(grp), grp
        if not move:
            break
        S.block(move)
        left -= len(move)
    if replace:
        _replace_stage(S)
    return list(S.blocked)


ALGORITHMS.update({"cutgreedy": cutgreedy, "cutgreedy_r": lambda S, b: cutgreedy(S, b, replace=True)})


# ---------------------------------------------------------------------------
# SWAP: best-improvement 1-swap local search with exact dominator pricing
# ---------------------------------------------------------------------------

def swap_local_search(S: SampleSet, budget: int, init: str = "ag", max_passes: int = 20, min_rel_improve: float = 0.002,
                      first_improvement: bool = False) -> list[int]:
    """SWAP: start from a one-shot solution, then repeatedly apply the best (remove u, add v) exchange.

    For each blocker u the samples are re-priced with u released (one dominator
    computation per sample), which gives the exact loss of releasing u and the
    exact gain of every candidate v given B \\ {u}; the exchange with the largest
    net improvement on the sample-average reach is applied. Stops at a 1-swap
    local optimum on the samples (or after ``max_passes`` passes / when the
    best improvement is below ``min_rel_improve`` of the current reach).
    """
    ALGORITHMS[init](S, budget)
    S.refresh()
    cur = S.expected_reach()
    for _ in range(max_passes):
        best = (0.0, None, None)
        B = list(S.blocked)
        for u in B:
            S.unblock(u)
            S.refresh()
            reach_wo = S.expected_reach()
            loss = reach_wo - cur  # >= 0: nodes lost by releasing u
            gains = S.single_gains()
            gains[u] = 0.0
            v = int(np.argmax(gains))
            improve = gains[v] - loss
            S.block([u])
            if improve > best[0]:
                best = (improve, u, v)
                if first_improvement:
                    break
        improve, u, v = best
        if u is None or improve <= min_rel_improve * max(cur, 1.0):
            break
        S.unblock(u)
        S.block([v])
        S.refresh()
        cur = S.expected_reach()
    return list(S.blocked)


ALGORITHMS.update({
    "swap": lambda S, b: swap_local_search(S, b, init="ag"),
    "swap_gr": lambda S, b: swap_local_search(S, b, init="gr"),
    "swap_first": lambda S, b: swap_local_search(S, b, init="ag", first_improvement=True),
})


# ---------------------------------------------------------------------------
# Sample-noise-aware greedy: penalise gains that are large only on few samples
# ---------------------------------------------------------------------------

def _gain_stats(S: SampleSet):
    """Per-node mean and standard error of the single-node saving across samples."""
    n = S.g.n
    acc = np.zeros(n)
    acc2 = np.zeros(n)
    for order, index, sizes, reach in S.cache:
        if len(order):
            gv = sizes[index[order]].astype(float)
            acc[order] += gv
            acc2[order] += gv * gv
    acc[S.seed_mask] = 0
    acc[S.forbidden] = 0
    mean = acc / S.theta
    var = np.maximum(acc2 / S.theta - mean * mean, 0.0)
    se = np.sqrt(var / S.theta)
    return mean, se


def lcb_greedy(S: SampleSet, budget: int, z: float = 1.0) -> list[int]:
    """AdvancedGreedy that picks argmax of (mean saving - z * standard error): a lower-confidence-bound
    rule that discounts nodes whose large average saving rests on a few samples (selection bias)."""
    for _ in range(budget):
        S.refresh()
        mean, se = _gain_stats(S)
        score = mean - z * se
        score[S.seed_mask] = -np.inf
        score[S.forbidden] = -np.inf
        v = int(np.argmax(score))
        if mean[v] <= 0:
            break
        S.block([v])
    return list(S.blocked)


def cv_greedy(S: SampleSet, budget: int, top_k: int = 10) -> list[int]:
    """Cross-validated greedy: shortlist the top-k nodes on the odd samples, pick the best of them on the
    even samples (honest estimate), so the selection is not made on the samples that inflate it."""
    half = S.theta // 2
    for _ in range(budget):
        S.refresh()
        n = S.g.n
        a = np.zeros(n)
        b = np.zeros(n)
        for i, (order, index, sizes, reach) in enumerate(S.cache):
            if len(order):
                (a if i < half else b)[order] += sizes[index[order]]
        for arr in (a, b):
            arr[S.seed_mask] = -np.inf
            arr[S.forbidden] = -np.inf
        short = np.argsort(-a)[:top_k]
        v = int(short[np.argmax(b[short])])
        if a[v] <= 0 and b[v] <= 0:
            break
        S.block([v])
    return list(S.blocked)


ALGORITHMS.update({"ag_lcb": lcb_greedy, "ag_lcb2": lambda S, b: lcb_greedy(S, b, z=2.0), "ag_cv": cv_greedy})


def phcut_algorithm(S: SampleSet, budget: int, **kw) -> list[int]:
    """PH-CUT on the sample set: scenario min-cuts with progressive hedging, AG solution as fallback/fill."""
    from .phcut import phcut
    snap = S.snapshot()
    fb = advanced_greedy(S, budget)
    S.restore(snap)
    B, info = phcut(S.g, S.lives, S.seeds, S.forbidden, budget, fallback=fb, **kw)
    S.last_info = info
    S.forbidden[B] = True
    S.blocked = list(B)
    return list(B)


ALGORITHMS.update({"phcut": phcut_algorithm})
