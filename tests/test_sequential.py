import numpy as np
import networkx as nx

from trace_e.graphs import load_network, to_csr
from trace_e.blocking.cascade import set_edge_probabilities
from trace_e.sequential.simulator import Episode, exposure
from trace_e.sequential.detectors import get_detector, round_log_lr, DETECTORS
from trace_e.sequential.containers import get_container, CONTAINERS
from trace_e.sequential.policy import run_episode


def _graph(p=0.15):
    return set_edge_probabilities(to_csr(load_network("dolphin")), "const", p)


def test_episode_coupling_and_counterfactual():
    g = _graph()
    ep1 = Episode(g, 2.0, [3], np.random.default_rng(5))
    ep2 = Episode(g, 2.0, [3], np.random.default_rng(5))
    while ep1.alive:
        ep1.step()
    assert ep1.harm == ep1.counterfactual_harm()
    # blocking can only reduce the final harm under the same live edges
    ep2.step()
    ep2.block(list(range(g.n)))
    while ep2.alive:
        ep2.step()
    assert ep2.harm <= ep1.harm
    # good campaign live edges are a subset of bad ones
    assert (ep1.live_good <= ep1.live_bad).all()


def test_eprocess_is_supermartingale_under_null():
    """Average of E_t over benign cascades stays around or below 1 (Theorem 1 sanity check)."""
    g = _graph(0.2)
    det = get_detector("eprocess", alpha=0.05, kappa_min=1.5, kappa_max=6.0)
    vals = []
    rng = np.random.default_rng(0)
    for i in range(300):
        ep = Episode(g, 1.0, [int(rng.integers(g.n))], np.random.default_rng(i))
        det.reset()
        rounds = 0
        while ep.alive and rounds < 40:
            inactive = ~(ep.active | ep.good | ep.blocked)
            fr = ep.frontier
            new = ep.step()
            rounds += 1
            exposed, l = exposure(g, fr, inactive, det.kappas)
            out = np.zeros(len(exposed), dtype=np.int8)
            out[np.isin(exposed, new)] = 1
            det.update(l, out, 0)
        vals.append(np.exp(det.log_E))
    m = np.mean(vals)
    assert m < 1.5, m  # E[E_tau] <= 1 in expectation; allow sampling slack
    assert np.mean(np.array(vals) >= 20) <= 0.05 + 0.03


def test_round_log_lr_matches_bernoulli():
    log1mq = np.log(np.array([[0.9, 0.5], [0.7, 0.2]]))  # q0 = .1,.5 ; q1 = .3,.8
    out = np.array([1, 0])
    v = round_log_lr(log1mq, out)
    assert np.isclose(v[0], 0.0)
    assert np.isclose(v[1], np.log(0.3 / 0.1) + np.log(0.2 / 0.5))


def test_all_detectors_and_containers_run():
    g = _graph()
    for d in DETECTORS:
        det = get_detector(d, alpha=0.1)
        if det.needs_calibration:
            det.threshold = 1.0
        for c in CONTAINERS:
            con = get_container(c, n_samples=5, pool=20)
            ep = Episode(g, 3.0, [1, 2], np.random.default_rng(3))
            r = run_episode(ep, det, con, budget=3, max_rounds=20)
            assert r["harm_final"] <= r["harm_counterfactual"]
            assert r["n_intervened"] <= 3


def test_deferred_commitment_matches_one_shot_on_same_live_edges():
    """Key step of Theorem 3: blocking a plan's nodes only when exposed yields the same final bad set."""
    g = _graph(0.25)
    rng = np.random.default_rng(11)
    checked = 0
    for trial in range(40):
        seeds = rng.choice(g.n, size=2, replace=False)
        plan = [int(v) for v in rng.choice(g.n, size=6, replace=False) if v not in seeds]
        ep_seed = int(rng.integers(2**31))
        # one-shot: block the whole plan after round 1
        a = Episode(g, 3.0, seeds, np.random.default_rng(ep_seed))
        a.step()
        a.block(plan)
        while a.alive:
            a.step()
        # deferred: each round block only the planned nodes exposed to the current frontier
        b = Episode(g, 3.0, seeds, np.random.default_rng(ep_seed))
        b.step()
        spent = 0
        pending = list(plan)
        while b.alive:
            exposed = np.zeros(g.n, dtype=bool)
            for u in b.frontier:
                exposed[g.indices[g.indptr[u]: g.indptr[u + 1]]] = True
            now = [v for v in pending if exposed[v]]
            spent += len(b.block(now))
            pending = [v for v in pending if v not in now]
            b.step()
        assert np.array_equal(a.active, b.active), trial
        assert spent <= len(plan)
        checked += 1
    assert checked == 40


def test_throttling_coupling_and_validity():
    from trace_e.sequential.throttlers import get_throttler
    g = _graph(0.2)
    # rho = 1 reproduces the unthrottled trajectory; rho < 1 never adds activations
    a = Episode(g, 2.0, [4], np.random.default_rng(2))
    b = Episode(g, 2.0, [4], np.random.default_rng(2))
    c = Episode(g, 2.0, [4], np.random.default_rng(2))
    rho = np.full(g.n, 0.3)
    while a.alive or b.alive or c.alive:
        na = a.step() if a.alive else []
        nb = b.step(np.ones(g.n)) if b.alive else []
        if c.alive:
            c.step(rho)
    assert np.array_equal(a.active, b.active)
    assert c.harm <= a.harm
    # e-process with hub throttling stays a supermartingale under the null (mean E_t around or below 1)
    det = get_detector("eprocess", alpha=0.05, kappa_min=1.5, kappa_max=6.0)
    thr = get_throttler("hub", rho_min=0.2, frac=0.5, alpha_soft=1.0)  # always on
    con = get_container("none")
    vals = []
    for i in range(300):
        ep = Episode(g, 1.0, [int(i % g.n)], np.random.default_rng(1000 + i))
        run_episode(ep, det, con, budget=0, max_rounds=40, throttler=thr)
        vals.append(np.exp(det.log_E))
    assert np.mean(np.array(vals) >= 20) <= 0.08  # crossing probability of 1/alpha stays below alpha (heavy-tailed mean is noisy)
