"""(1) How accurate is Monte Carlo evaluation of a blocking set, against the exact value from TWIG, on graphs of 400-900 vertices?
   (2) How good are the sampling-based AdvancedGreedy plans, evaluated exactly, against exact greedy (and the optimum for budget 2)?

Model: every undirected edge is live independently with its probability (one coin shared by both directions), as in TWIG.
Results: results/theory/twig_accuracy.json and results/theory/twig_planners.csv
    python scripts/twig_accuracy_planners.py
"""
from __future__ import annotations

import csv, itertools, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from trace_e.blocking.imin import ALGORITHMS, SampleSet
from trace_e.blocking.layout import best_layout, layout_to_order
from trace_e.blocking.treewidth import ExactSpreadFast, greedy_exact
from trace_e.graphs import CSRGraph, load_empirical


def make_weights(G, lo, hi, seed):
    rng = random.Random(seed); return {frozenset(e): rng.uniform(lo, hi) for e in G.edges}


class MC:
    """Monte Carlo estimator for the undirected live-edge model, with common samples across blocking sets."""
    def __init__(self, G, W, seeds):
        self.G, self.n = G, G.number_of_nodes(); self.edges = np.array(list(G.edges)); self.p = np.array([W[frozenset(e)] for e in G.edges]); self.seeds = list(seeds)

    def samples(self, theta, rng):
        return rng.random((theta, len(self.p))) < self.p

    def spread(self, live, blocked):
        blk = np.zeros(self.n, dtype=bool); blk[list(blocked)] = True; tot = 0.0; e = self.edges
        keep_e = ~(blk[e[:, 0]] | blk[e[:, 1]])
        for row in live:
            k = row & keep_e
            A = sp.coo_matrix((np.ones(k.sum()), (e[k, 0], e[k, 1])), shape=(self.n, self.n))
            _, lab = connected_components(A, directed=False)
            tot += np.isin(lab, np.unique(lab[self.seeds])).sum() - len(self.seeds) - (blk & np.isin(lab, np.unique(lab[self.seeds]))).sum()
        return tot / len(live)


def exact(G, W, seeds, blocked, order=None):
    wfn = lambda u, v: W[frozenset((u, v))]
    lay, _ = best_layout(G)
    return ExactSpreadFast(G, wfn, seeds, order=order or layout_to_order(lay)).evaluate(frozenset(blocked))


def instances():
    out = {}
    out["grid 3 x 300 (n=900)"] = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 300))
    out["random tree (n=800)"] = nx.random_labeled_tree(800, seed=2) if hasattr(nx, "random_labeled_tree") else nx.random_tree(800, seed=2)
    out["ring lattice k=4 (n=500)"] = nx.watts_strogatz_graph(500, 4, 0.0, seed=1)
    P = nx.convert_node_labels_to_integers(load_empirical("powergrid")); rng = random.Random(3); best = None
    for root in rng.sample(list(P.nodes), 40):
        seen = [root]; S = {root}; i = 0
        while i < len(seen) and len(seen) < 400:
            for u in P.neighbors(seen[i]):
                if u not in S and len(seen) < 400: S.add(u); seen.append(u)
            i += 1
        B = nx.convert_node_labels_to_integers(P.subgraph(seen).copy())
        if B.number_of_nodes() >= 380:
            _, w = best_layout(B)
            if w <= 8: best = B; break
    if best is not None: out["power-grid ball (n=400)"] = best
    return out


def part_accuracy():
    res = []
    for name, G in instances().items():
        W = make_weights(G, 0.1, 0.6, 5); seeds = [0]; rng = np.random.default_rng(0); mc = MC(G, W, seeds)
        near = [v for v, d in nx.single_source_shortest_path_length(G, 0).items() if 1 <= d <= 8]
        r = random.Random(9); B1 = r.sample(near, 5); B2 = B1[:-1] + [r.choice([v for v in near if v not in B1])]
        t0 = time.time(); F1, F2 = exact(G, W, seeds, B1), exact(G, W, seeds, B2); t_ex = (time.time() - t0) / 2
        row = dict(instance=name, n=G.number_of_nodes(), exact_seconds=t_ex, F_exact=F1, diff_exact=F1 - F2, theta={})
        for theta, reps in ((100, 200), (1000, 100), (10000, 15)):
            est, sgn = [], 0
            for _ in range(reps):
                live = mc.samples(theta, rng); a, b = mc.spread(live, B1), mc.spread(live, B2); est.append(a); sgn += int((a - b) * (F1 - F2) < 0 or (a - b) == 0)
            est = np.array(est); row["theta"][theta] = dict(reps=reps, rel_rmse=float(np.sqrt(np.mean((est - F1) ** 2)) / F1), rel_bias=float((est.mean() - F1) / F1),
                                                            coverage95=float(np.mean(np.abs(est - F1) <= 1.96 * est.std(ddof=1))) if reps > 5 else None, sign_error=sgn / reps)
        res.append(row); print(name, {k: (round(v["rel_rmse"], 4), v["sign_error"]) for k, v in row["theta"].items()}, "exact", round(F1, 3), "diff", round(F1 - F2, 4), flush=True)
    json.dump(res, open("results/theory/twig_accuracy.json", "w"), indent=1)


def csr_symmetric(G, W):
    n = G.number_of_nodes(); eid = {}
    for i, e in enumerate(G.edges): eid[frozenset(e)] = i
    H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(G.edges, weight=1.0)
    A = nx.to_scipy_sparse_array(H, nodelist=range(n), weight="weight", format="csr"); A.sort_indices()
    ind = A.indices.astype(np.int64); ptr = A.indptr.astype(np.int64)
    arc_e = np.array([eid[frozenset((u, int(v)))] for u in range(n) for v in ind[ptr[u]:ptr[u + 1]]])
    p = np.array([W[frozenset(e)] for e in G.edges]); return CSRGraph(n=n, indptr=ptr, indices=ind, weights=p[arc_e]), arc_e


def part_planners():
    rows = []
    cases = {"karate": nx.convert_node_labels_to_integers(load_empirical("karate")), "Iceland": nx.convert_node_labels_to_integers(load_empirical("iceland")),
             "grid 3 x 25": nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 25)), "random tree (80)": nx.random_labeled_tree(80, seed=4) if hasattr(nx, "random_labeled_tree") else nx.random_tree(80, seed=4),
             "ring lattice k=4 (80)": nx.watts_strogatz_graph(80, 4, 0.0, seed=2)}
    for name, G in cases.items():
        for inst in range(3):
            W = make_weights(G, 0.1, 0.6, 20 + inst); rg = random.Random(100 + inst); seeds = rg.sample(list(G.nodes), 2); lay, w = best_layout(G); order = layout_to_order(lay)
            wfn = lambda u, v, W=W: W[frozenset((u, v))]
            g, arc_e = csr_symmetric(G, W); es = ExactSpreadFast(G, wfn, seeds, order=order)
            for b in (2, 4):
                Bg, Fg = greedy_exact(G, wfn, seeds, b, order=order)
                Fopt = None
                if b == 2 and G.number_of_nodes() <= 80:
                    pool = [v for v in G.nodes if v not in seeds]; Fopt = min(es.evaluate(frozenset(c)) for c in itertools.combinations(pool, 2))
                for theta in (100, 1000):
                    vals = []
                    for rep in range(10):
                        rng = np.random.default_rng(1000 * rep + inst); coins = rng.random((theta, G.number_of_edges())); lives = [coins[i][arc_e] < g.weights for i in range(theta)]
                        S = SampleSet(g, seeds, theta, rng, lives=lives); S.refresh(); plan = ALGORITHMS["ag"](S, b); vals.append(es.evaluate(frozenset(plan)))
                    rows.append(dict(graph=name, inst=inst, n=G.number_of_nodes(), width=w, budget=b, theta=theta, exact_greedy=Fg, optimum=Fopt, ag_mean=float(np.mean(vals)), ag_worst=float(np.max(vals)),
                                     ag_equal_to_greedy_value=float(np.mean([abs(v - Fg) < 1e-9 for v in vals]))))
                print(rows[-1], flush=True)
    with open("results/theory/twig_planners.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["accuracy", "planners"]
    if "accuracy" in which: part_accuracy()
    if "planners" in which: part_planners()
