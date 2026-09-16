"""Correctness tests for trace_e.blocking.treewidth.ExactSpread.

The dynamic program computes E[#nodes reachable from seeds, excluding
seeds] under the independent-cascade model exactly, for any elimination
order and any set of deleted (blocked) nodes. It is checked here against
brute-force enumeration over all 2^m live-edge subsets on small random
graphs (both without and with a blocked set), which is the ground truth
for this #P-hard quantity on graphs small enough to enumerate exhaustively.
"""
from __future__ import annotations

import random

import networkx as nx
import pytest

from trace_e.blocking.treewidth import ExactSpread, brute_force_spread


def _random_instance(rng, n_max=9, m_max=16):
    n = rng.randint(2, n_max)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    m = rng.randint(1, min(m_max, len(all_edges)))
    edges = all_edges[:m]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    weights = {frozenset(e): rng.uniform(0.02, 0.98) for e in G.edges}
    return G, weights


def test_exact_spread_matches_brute_force():
    rng = random.Random(0)
    checked = 0
    for _ in range(120):
        G, weights = _random_instance(rng)
        if G.number_of_edges() == 0:
            continue

        def wfn(u, v, weights=weights):
            return weights[frozenset((u, v))]

        seeds = rng.sample(list(G.nodes), k=rng.randint(1, min(3, G.number_of_nodes())))
        es = ExactSpread(G, wfn, seeds)
        dp = es.evaluate()
        bf = brute_force_spread(G, wfn, seeds)
        assert dp == pytest.approx(bf, abs=1e-6)
        checked += 1
    assert checked > 50


def test_exact_spread_matches_brute_force_with_blocking():
    rng = random.Random(1)
    checked = 0
    for _ in range(120):
        G, weights = _random_instance(rng)
        if G.number_of_edges() == 0:
            continue

        def wfn(u, v, weights=weights):
            return weights[frozenset((u, v))]

        n = G.number_of_nodes()
        seeds = rng.sample(list(G.nodes), k=rng.randint(1, min(3, n)))
        candidates = [v for v in G.nodes if v not in seeds]
        if not candidates:
            continue
        blocked = set(rng.sample(candidates, k=min(rng.randint(1, 3), len(candidates))))
        es = ExactSpread(G, wfn, seeds)
        dp = es.evaluate(blocked)
        bf = brute_force_spread(G, wfn, seeds, blocked)
        assert dp == pytest.approx(bf, abs=1e-6)
        checked += 1
    assert checked > 50


def test_seed_only_graph_contributes_nothing():
    """A seed with no edges (or fully blocked neighbours) contributes zero spread."""
    G = nx.Graph()
    G.add_nodes_from(range(3))
    es = ExactSpread(G, lambda u, v: 0.5, [0])
    assert es.evaluate() == 0.0


def test_deterministic_path():
    """A deterministic (p=1) path from a seed reaches every downstream node with certainty."""
    G = nx.path_graph(5)
    es = ExactSpread(G, lambda u, v: 1.0, [0])
    assert es.evaluate() == pytest.approx(4.0)
    # blocking the middle node severs the path: only node 1 stays connected to the seed
    assert es.evaluate({2}) == pytest.approx(1.0)
