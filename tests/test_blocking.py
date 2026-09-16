import numpy as np
import networkx as nx

from trace_e.graphs import load_network, to_csr
from trace_e.blocking import BlockingContext, get_blocker, REGISTRY
from trace_e.blocking.cascade import ic_spread, competitive_ic_spread, estimate_bad_spread, set_edge_probabilities


def test_ic_spread_blocking_and_probabilities():
    g = set_edge_probabilities(to_csr(nx.path_graph(5)), "const", 1.0)
    rng = np.random.default_rng(0)
    assert ic_spread(g, [0], None, rng).sum() == 5
    blocked = np.zeros(5, dtype=bool)
    blocked[2] = True
    assert ic_spread(g, [0], blocked, rng).sum() == 2
    g0 = set_edge_probabilities(to_csr(nx.path_graph(5)), "const", 0.0)
    assert ic_spread(g0, [0], None, rng).sum() == 1


def test_competitive_ic_first_arrival():
    g = set_edge_probabilities(to_csr(nx.path_graph(7)), "const", 1.0)
    rng = np.random.default_rng(0)
    bad = competitive_ic_spread(g, [0], [6], rng)
    # bad reaches 0,1,2 and good reaches 6,5,4; node 3 is a tie -> good wins
    assert bad.sum() == 3 and bad[:3].all()
    bad = competitive_ic_spread(g, [0], [6], rng, good_delay=2)
    assert bad.sum() == 4


def test_all_blockers_reduce_spread():
    G = load_network("dolphin")
    ctx = BlockingContext.from_graph(G, prob_model="const", p=0.3, mode="block", seed=0)
    seeds = np.array([0])
    base = estimate_bad_spread(ctx.g, seeds, [], "block", 200, seed=5)
    for name in REGISTRY:
        kw = {"n_samples": 30} if name in ("greedy", "proposed") else {}
        b = get_blocker(name, ctx, **kw)
        b.prepare()
        chosen = b.select(seeds, 3)
        assert len(chosen) == 3 and 0 not in chosen and len(set(chosen)) == 3
        after = estimate_bad_spread(ctx.g, seeds, chosen, "block", 200, seed=5)
        assert after <= base
    greedy = get_blocker("greedy", ctx, n_samples=100)
    after_g = estimate_bad_spread(ctx.g, seeds, greedy.select(seeds, 3), "block", 400, seed=5)
    after_r = estimate_bad_spread(ctx.g, seeds, get_blocker("random", ctx).select(seeds, 3), "block", 400, seed=5)
    assert after_g <= after_r


def test_dominator_greedy_matches_or_beats_celf_on_samples():
    G = load_network("dolphin")
    ctx = BlockingContext.from_graph(G, prob_model="const", p=0.3, mode="block", seed=0)
    seeds = np.array([0, 5])
    a = get_blocker("greedy_dom", ctx, n_samples=100).select(seeds, 4)
    b = get_blocker("greedy", ctx, n_samples=100).select(seeds, 4)
    assert len(a) == 4 and len(set(a)) == 4 and not set(a) & set(seeds.tolist())
    sa = estimate_bad_spread(ctx.g, seeds, a, "block", 500, seed=3)
    sb = estimate_bad_spread(ctx.g, seeds, b, "block", 500, seed=3)
    assert sa <= sb * 1.1 + 1


def test_adaptivity_gap_family():
    """Theorem 2 of docs/DEFER.md: one-shot budget-1 blocking cannot beat DEFER on the star-of-paths instance."""
    import networkx as nx
    from trace_e.graphs import CSRGraph
    from trace_e.sequential.simulator import Episode
    from trace_e.sequential.containers import get_container
    Delta, L, p = 8, 30, 0.15
    G = nx.DiGraph()
    s = 0
    nid = 1
    for i in range(Delta):
        a = nid
        nid += 1
        G.add_edge(s, a, w=p)
        prev = a
        for _ in range(L):
            G.add_edge(prev, nid, w=1.0)
            prev = nid
            nid += 1
    n = nid
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight="w", format="csr", dtype=float)
    g = CSRGraph(n=n, indptr=A.indptr.astype(np.int64), indices=A.indices.astype(np.int64), weights=A.data.astype(float))
    one_shot, defer, none = [], [], []
    for r in range(300):
        ep = Episode(g, 1.0, [s], np.random.default_rng(r))
        ep.block([1])  # best one-shot node: the first branch head
        while ep.alive:
            ep.step()
        one_shot.append(ep.harm - 1)
        ep = Episode(g, 1.0, [s], np.random.default_rng(r))
        con = get_container("adaptive", n_samples=20, horizon=0, replan_every=1)
        con.reset(1)
        while ep.alive:
            ep.intervene(con.act(ep, 1.0))
            ep.step()
        defer.append(ep.harm - 1)
        ep = Episode(g, 1.0, [s], np.random.default_rng(r))
        while ep.alive:
            ep.step()
        none.append(ep.harm - 1)
    saved_one = np.mean(none) - np.mean(one_shot)
    saved_defer = np.mean(none) - np.mean(defer)
    # theory: one-shot saves ~ p(L+1), DEFER saves ~ (1-(1-p)^Delta)(L+1)
    assert saved_defer > 3 * saved_one, (saved_one, saved_defer)
