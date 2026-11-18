"""Vertex orders with a small frontier (vertex separation) for the TWIG dynamic program.

The frontier width of a linear order sigma is  max_i |F_i| - 1  with F_i = {sigma_j : j <= i, some neighbour of sigma_j (or sigma_j itself)
sits at position >= i}. Minimising it over orders gives the pathwidth. ``ExactSpread`` / ``ExactSpreadFast`` take an *elimination* order and
process it reversed, so a layout ``sigma`` is passed as ``order=layout_to_order(sigma)``.
"""
from __future__ import annotations

import networkx as nx


def layout_to_order(layout):
    return list(reversed(layout))


def frontier_width(G: nx.Graph, layout) -> int:
    pos = {v: i for i, v in enumerate(layout)}
    last = {v: max([pos[v]] + [pos[u] for u in G.neighbors(v)]) for v in layout}
    ends = [0] * (len(layout) + 1)
    for v in layout:
        ends[last[v]] += 1
    active, best = 0, 0
    for i in range(len(layout)):
        active += 1                 # sigma_i enters
        best = max(best, active)
        active -= ends[i]           # vertices whose last position is i leave after step i
    return best - 1


def greedy_frontier_layout(G: nx.Graph, start) -> list:
    """Greedy layout from ``start`` (components are laid out one after the other): place next the vertex that leaves the smallest
    frontier, preferring vertices adjacent to the frontier."""
    unplaced_nb = {v: G.degree(v) for v in G}   # number of unplaced neighbours (self-loops ignored by nx.Graph.degree counting twice: none expected)
    placed, layout, active = set(), [], set()
    remaining = set(G.nodes)

    def score(v):
        retire = sum(1 for u in G.neighbors(v) if u in active and unplaced_nb[u] == 1)
        stays = 1 if (unplaced_nb[v] > 0) else 0
        return (len(active) - retire + stays, -sum(1 for u in G.neighbors(v) if u in placed), G.degree(v))

    first = start
    while remaining:
        if not active:
            v = first if first in remaining else min(remaining, key=lambda x: (G.degree(x), x))
            first = None
        else:
            cand = {u for a in active for u in G.neighbors(a) if u in remaining}
            if not cand:
                cand = remaining
            v = min(cand, key=lambda x: (score(x), x))
        layout.append(v); remaining.discard(v); placed.add(v)
        for u in G.neighbors(v):
            unplaced_nb[u] -= 1
        # update frontier membership
        active = {u for u in (active | {v}) if unplaced_nb[u] > 0}
    return layout


def best_layout(G: nx.Graph, tries: int = 8, seed: int = 0):
    """Best of a few greedy layouts (start vertices: peripheral vertices from a double sweep and low-degree vertices) and reverse Cuthill-McKee."""
    import random
    rng = random.Random(seed)
    starts = []
    comp = max(nx.connected_components(G), key=len)
    a = next(iter(comp))
    for _ in range(2):  # double sweep to find a peripheral vertex
        dist = nx.single_source_shortest_path_length(G, a); a = max(dist, key=lambda v: (dist[v], -G.degree(v)))
    starts.append(a)
    low = sorted(G.nodes, key=lambda v: (G.degree(v), v))[: max(1, tries // 2)]
    starts += low + rng.sample(list(G.nodes), min(tries, len(G)))
    best, best_w = None, None
    seen = set()
    for s in starts:
        if s in seen or len(seen) >= tries: continue
        seen.add(s)
        lay = greedy_frontier_layout(G, s); w = frontier_width(G, lay)
        if best_w is None or w < best_w: best, best_w = lay, w
    try:
        rcm = list(nx.utils.reverse_cuthill_mckee_ordering(G)); w = frontier_width(G, rcm)
        if w < best_w: best, best_w = rcm, w
    except Exception:
        pass
    return best, best_w
