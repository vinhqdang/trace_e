"""Correctness tests for trace_e.source.exact_sir.ExactSIR: verify the
variable-elimination DP's log-likelihood matches an explicit brute-force
simulation of the full 3^n-state discrete-time SIR Markov chain, on random
graphs small enough to enumerate exactly.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from trace_e.source.exact_sir import ExactSIR, brute_force_posterior


def _random_instance(rng, n_max=6, m_max=10):
    n = rng.randint(3, n_max)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    m = rng.randint(n - 1, min(m_max, len(all_edges)))
    edges = all_edges[:m]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    # ensure connected (retry-free: just union find fallback -> star backbone)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for a, b in zip(comps, comps[1:]):
            G.add_edge(next(iter(a)), next(iter(b)))
    return G


def _random_snapshot(rng, n):
    """A plausible-ish observed snapshot: not used as ground truth, just a
    fixed query point compared between the two implementations."""
    return {v: rng.choice([0, 1, 2]) for v in range(n)}


def test_exact_sir_matches_brute_force_one_step():
    rng = random.Random(0)
    checked = 0
    for _ in range(150):
        G = _random_instance(rng, n_max=5, m_max=8)
        n = G.number_of_nodes()
        beta = rng.uniform(0.3, 1.5)
        nu = 1.0
        dt = 0.2
        T = dt  # exactly one step
        source = rng.randrange(n)
        observed = _random_snapshot(rng, n)
        model = ExactSIR(G, lambda u, v: 1.0, beta, nu, T, dt=dt)
        ll = model.log_likelihood(source, observed)
        p_dp = math.exp(ll) if ll > float("-inf") else 0.0
        p_bf = brute_force_posterior(G, lambda u, v: 1.0, source, observed, beta, nu, T, dt=dt)
        assert p_dp == pytest.approx(p_bf, abs=1e-9)
        checked += 1
    assert checked > 20


def test_exact_sir_matches_brute_force_multi_step():
    rng = random.Random(1)
    checked = 0
    for _ in range(120):
        G = _random_instance(rng, n_max=6, m_max=9)
        n = G.number_of_nodes()
        beta = rng.uniform(0.3, 1.2)
        nu = 1.0
        dt = 0.2
        K = rng.randint(2, 4)
        T = dt * K
        source = rng.randrange(n)
        observed = _random_snapshot(rng, n)
        model = ExactSIR(G, lambda u, v: 1.0, beta, nu, T, dt=dt)
        ll = model.log_likelihood(source, observed)
        p_dp = math.exp(ll) if ll > float("-inf") else 0.0
        p_bf = brute_force_posterior(G, lambda u, v: 1.0, source, observed, beta, nu, T, dt=dt)
        assert p_dp == pytest.approx(p_bf, abs=1e-8)
        checked += 1
    assert checked > 15


def test_exact_sir_posterior_sums_to_one():
    """A structural check independent of the brute-force comparison: summing
    exp(log_likelihood) over every one of the 3^n possible observed
    snapshots must give exactly 1 (it's a probability distribution over the
    full joint state at time K)."""
    import itertools as it

    rng = random.Random(2)
    checked = 0
    for _ in range(15):
        G = _random_instance(rng, n_max=5, m_max=7)
        n = G.number_of_nodes()
        beta = rng.uniform(0.3, 1.2)
        nu = 1.0
        dt = 0.25
        K = rng.randint(1, 3)
        T = dt * K
        source = rng.randrange(n)
        model = ExactSIR(G, lambda u, v: 1.0, beta, nu, T, dt=dt)
        total = 0.0
        for combo in it.product(range(3), repeat=n):
            observed = {v: combo[v] for v in range(n)}
            ll = model.log_likelihood(source, observed)
            if ll > float("-inf"):
                total += math.exp(ll)
        assert total == pytest.approx(1.0, abs=1e-6)
        checked += 1
    assert checked > 8


def test_exact_sir_deterministic_no_recovery_reaches_certainty():
    """beta huge, dt/T such that with p~=1 transmission the source's
    immediate neighbours become infected with near-certainty after one step."""
    G = nx.path_graph(4)
    model = ExactSIR(G, lambda u, v: 1.0, beta=50.0, nu=0.0001, T=0.1, dt=0.1)
    observed = {0: 1, 1: 1, 2: 0, 3: 0}  # source + immediate neighbour infected, rest S
    ll = model.log_likelihood(0, observed)
    assert math.exp(ll) > 0.99
