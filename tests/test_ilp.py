import itertools

import numpy as np

from trace_e.blocking.ilp import _sample_scenarios, solve_plan
from trace_e.graphs import to_csr
from trace_e.sequential.simulator import Episode
import networkx as nx


def _saa_value(scen, blocked):
    """Average number of (non-frontier) nodes reached when `blocked` is blocked at time 0."""
    tot = 0.0
    for arr, edges in scen:
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
        front = {u for u, _ in edges if u not in arr}  # frontier nodes (never arrive)
        seen, stack = set(), [u for u in front]
        while stack:
            u = stack.pop()
            for v in adj.get(u, []):
                if v not in blocked and v not in seen:
                    seen.add(v)
                    stack.append(v)
        tot += len(seen)
    return tot / len(scen)


def test_one_shot_ilp_matches_brute_force():
    G = nx.gnp_random_graph(14, 0.25, seed=3, directed=True)
    for u, v in G.edges:
        G[u][v]["weight"] = 0.5
    g = to_csr(G)
    g.weights[:] = 0.5
    ep = Episode(g, 1.0, [0, 1], np.random.default_rng(0))
    for b in (1, 2, 3):
        scen = _sample_scenarios(g, ep, 25, np.random.default_rng(7))
        cand = sorted({v for arr, _ in scen for v in arr})
        best = min(_saa_value(scen, set(S)) for S in itertools.combinations(cand, min(b, len(cand))))
        nodes, info = solve_plan(g, ep, b, 25, 3, False, np.random.default_rng(7), 30.0, slack=100)
        assert abs(info["obj"] - best) < 1e-6, (b, info["obj"], best)
        assert abs(_saa_value(scen, set(nodes)) - best) < 1e-6
