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
