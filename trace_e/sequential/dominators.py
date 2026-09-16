"""Exact marginal blocking gains for all nodes via dominator trees.

On a fixed live-edge graph, the nodes that become unreachable from the
sources when a single node ``v`` is removed are exactly the nodes that ``v``
dominates in the flow graph rooted at a virtual super-source connected to
all sources. Hence the marginal saving of blocking ``v`` equals the size of
``v``'s subtree in the dominator tree, for *every* candidate at once, in
near-linear time per sample (Cooper, Harvey & Kennedy 2001 iterative
algorithm on the reachable subgraph). Re-running after each accepted greedy
pick (with the picked nodes removed) gives exact conditional gains, which is
the AdvancedGreedy idea of Xie et al. (ICDE 2023) for influence minimisation.
"""
from __future__ import annotations

import numpy as np

from ..graphs import CSRGraph


def reachable_dag(g: CSRGraph, live: np.ndarray, sources, forbidden: np.ndarray, horizon: int = 0):
    """Reverse-postorder DFS from a super-source over live edges avoiding ``forbidden``.

    ``horizon > 0`` truncates the flow graph to nodes within that many hops of
    the sources (BFS distance on live edges), which bounds the cost of the
    dominator computation on large cascades at the price of ignoring savings
    beyond the horizon.

    Returns (order, preds, index) where ``order`` lists reachable nodes in
    reverse postorder (super-source excluded, sources first), ``preds[i]`` is the
    list of predecessor positions of the i-th node in ``order`` (position -1
    denotes the super-source), and ``index`` maps node id -> position (or -1).
    """
    n = g.n
    index = np.full(n, -1, dtype=np.int64)
    src = [int(s) for s in sources]
    if horizon > 0:
        # restrict to nodes within `horizon` live hops of the sources
        dist = np.full(n, -1, dtype=np.int64)
        frontier = []
        for s0 in src:
            dist[s0] = 0
            frontier.append(s0)
        for d in range(1, horizon + 1):
            nxt = []
            for u in frontier:
                lo, hi = g.indptr[u], g.indptr[u + 1]
                for v, ok in zip(g.indices[lo:hi], live[lo:hi]):
                    if ok and not forbidden[v] and dist[v] < 0:
                        dist[v] = d
                        nxt.append(int(v))
            frontier = nxt
            if not frontier:
                break
        forbidden = forbidden | (dist < 0)
        for s0 in src:
            forbidden[s0] = False
    # iterative DFS to get postorder
    visited = np.zeros(n, dtype=bool)
    post = []
    preds_map: dict[int, list[int]] = {}
    stack = []
    for s in src:
        if visited[s]:
            continue
        visited[s] = True
        preds_map.setdefault(s, []).append(-1)
        stack.append((s, iter(range(g.indptr[s], g.indptr[s + 1]))))
        while stack:
            u, it = stack[-1]
            advanced = False
            for e in it:
                if not live[e]:
                    continue
                v = int(g.indices[e])
                if forbidden[v]:
                    continue
                preds_map.setdefault(v, []).append(u)
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, iter(range(g.indptr[v], g.indptr[v + 1]))))
                    advanced = True
                    break
            if not advanced:
                post.append(u)
                stack.pop()
    order = post[::-1]
    for i, u in enumerate(order):
        index[u] = i
    # super-source edges to *all* sources (a source reachable from another source still has the root as a predecessor)
    for s in src:
        if s in preds_map and -1 not in preds_map[s]:
            preds_map[s].append(-1)
    preds = [[(-1 if p == -1 else int(index[p])) for p in preds_map[u]] for u in order]
    return order, preds, index


def dominator_subtree_sizes(order, preds) -> np.ndarray:
    """Cooper-Harvey-Kennedy immediate dominators; returns subtree sizes in dominator tree (positions)."""
    m = len(order)
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    ROOT = -1
    idom = [None] * m  # None = undefined
    # positions are already reverse postorder: smaller position = earlier in RPO
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
    sizes = np.ones(m, dtype=np.int64)
    for i in range(m - 1, -1, -1):  # children have larger RPO positions than their dominators
        d = idom[i]
        if d is not None and d != ROOT:
            sizes[d] += sizes[i]
    return sizes


def marginal_gains(g: CSRGraph, live: np.ndarray, sources, forbidden: np.ndarray, with_sources: bool = False, horizon: int = 0):
    """(nodes, gains): for every reachable non-source node, the number of nodes saved by blocking it.

    With ``with_sources=True`` also returns ``(src_nodes, src_gains)``: for each
    source, the number of nodes saved by cutting all of its out-edges
    (isolating it), which equals its dominator-subtree size minus one.
    """
    order, preds, index = reachable_dag(g, live, sources, forbidden, horizon)
    sizes = dominator_subtree_sizes(order, preds)
    src = set(int(s) for s in sources)
    nodes = np.array([u for u in order if u not in src], dtype=np.int64)
    gains = np.array([sizes[index[u]] for u in nodes], dtype=np.int64)
    if not with_sources:
        return nodes, gains
    src_nodes = np.array([u for u in order if u in src], dtype=np.int64)
    src_gains = np.array([sizes[index[u]] - 1 for u in src_nodes], dtype=np.int64)
    return nodes, gains, src_nodes, src_gains


def dominator_greedy_plan(g: CSRGraph, lives: list[np.ndarray], sources, forbidden: np.ndarray, budget: int,
                          cut_moves: bool = True, horizon: int = 0):
    """Cost-sensitive greedy blocking with exact per-sample marginal gains from dominator trees.

    Two kinds of moves are compared by expected nodes saved per budget unit:
    blocking one node (cost 1; gain = its dominator-subtree size), and, when
    ``cut_moves`` is on, isolating a source by blocking all of its inactive,
    unblocked out-neighbours (cost = their number; gain = the source's
    dominator-subtree size minus one). Isolation moves capture coordinated
    cuts whose single-node marginal gains are individually small, the
    non-submodular structure that makes one-node greedy myopic on vertex
    blocking. Returns (plan, gains) where gains are per-node average savings
    (an isolation move contributes its gain split evenly over its nodes).
    """
    forb = forbidden.copy()
    plan, gains = [], []
    L = len(lives)
    src_list = [int(s) for s in sources]
    left = budget
    while left > 0:
        acc = np.zeros(g.n)
        acc_src = np.zeros(g.n)
        for lv in lives:
            if cut_moves:
                nodes, gv, sn, sg = marginal_gains(g, lv, src_list, forb, with_sources=True, horizon=horizon)
                if len(sn):
                    acc_src[sn] += sg
            else:
                nodes, gv = marginal_gains(g, lv, src_list, forb, horizon=horizon)
            if len(nodes):
                acc[nodes] += gv
        acc[forb] = 0
        acc[src_list] = 0
        v = int(np.argmax(acc))
        best_ratio, best_move = acc[v], ("node", v, [v])
        if cut_moves:
            for u in src_list:
                if acc_src[u] <= 0:
                    continue
                nb = g.indices[g.indptr[u]: g.indptr[u + 1]]
                grp = [int(x) for x in np.unique(nb) if not forb[x] and x not in src_list]
                if 0 < len(grp) <= left and acc_src[u] / len(grp) > best_ratio:
                    best_ratio, best_move = acc_src[u] / len(grp), ("cut", u, grp)
        if best_ratio <= 0:
            break
        kind, u, grp = best_move
        for x in grp:
            plan.append(x)
            gains.append(float(best_ratio / L))
            forb[x] = True
        left -= len(grp)
    return plan, gains
