import numpy as np
import networkx as nx

from trace_e.graphs import load_network, to_csr
from trace_e.simulate import simulate_dataset, simulate_sir
from trace_e.metrics import top_k_hit, credible_set_size, reciprocal_rank, error_distance
from trace_e.source import Context, get_detector, REGISTRY


def test_sir_states_and_determinism():
    G = load_network("karate")
    g = to_csr(G)
    d1 = simulate_dataset(g, 1.3, 0.5, 0.85, 3, seed=7, n_jobs=1)
    d2 = simulate_dataset(g, 1.3, 0.5, 0.85, 3, seed=7, n_jobs=2)
    assert np.array_equal(d1.states, d2.states) and np.array_equal(d1.sources, d2.sources)
    assert set(np.unique(d1.states)) <= {0, 1, 2}
    # the source is never susceptible
    assert all(d1.states[i, s] != 0 for i, s in enumerate(d1.sources))
    # infected set is connected
    for st in d1.states[:20]:
        sub = G.subgraph(np.flatnonzero(st != 0).tolist())
        assert nx.is_connected(sub)


def test_sir_T_zero_and_beta_zero():
    g = to_csr(load_network("karate"))
    rng = np.random.default_rng(0)
    st = simulate_sir(g, 0, beta=1.0, T=0.0, rng=rng)
    assert (st != 0).sum() == 1
    st = simulate_sir(g, 0, beta=1e-9, T=5.0, rng=rng)
    assert (st != 0).sum() == 1 and st[0] == 2


def test_metrics():
    rng = np.random.default_rng(0)
    s = np.array([-np.inf, 0.0, 1.0, 1.0])
    assert top_k_hit(s, 2, 1, rng) or top_k_hit(s, 3, 1, rng)
    assert top_k_hit(s, 1, 3, rng)
    assert not top_k_hit(s, 0, 4, rng)
    assert reciprocal_rank(s, 1) == 1 / 3
    assert credible_set_size(np.log(np.array([0.5, 0.3, 0.15, 0.05]))) == 3
    g = to_csr(nx.path_graph(4))
    assert error_distance(g, 0, 3) == 3


def test_all_detectors_run():
    G = load_network("karate")
    ctx = Context.from_graph(G, beta=1.3, T=0.85, seed=0)
    train = simulate_dataset(ctx.g, 1.3, 0.5, 0.85, 2, seed=1)
    test = simulate_dataset(ctx.g, 1.3, 0.0, 0.85, 1, seed=2)
    for name in REGISTRY:
        kw = {"epochs": 2} if name in ("gcn", "gcn_skip", "mlp", "igcn") else {}
        det = get_detector(name, ctx, **kw)
        det.fit(train if det.needs_training_sims else None)
        sc = det.score(test.states[0])
        assert sc.shape == (ctx.n,)
        assert np.isinf(sc[test.states[0] == 0]).all() or name in ("degree",)
        assert np.isfinite(sc[test.sources[0]])


def test_trajectories_match_snapshots():
    from trace_e.simulate import simulate_trajectories, simulate_sir_times, snapshot_at
    g = to_csr(load_network("dolphin"))
    # same RNG consumption as simulate_sir -> identical snapshot at T_max
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    st = simulate_sir(g, 5, 1.3, 0.85, rng1)
    a, r = simulate_sir_times(g, 5, 1.3, 0.85, rng2)
    assert np.array_equal(st, snapshot_at(a, r, 0.85))
    traj = simulate_trajectories(g, 1.3, 0.5, 2.0, 2, seed=9, n_jobs=2)
    s1, s2 = traj.snapshots(0.5), traj.snapshots(2.0)
    # infected sets are monotone in time
    assert ((s1 != 0) <= (s2 != 0)).all()
    assert (traj.inf_times[np.arange(len(traj)), traj.sources] == 0).all()
