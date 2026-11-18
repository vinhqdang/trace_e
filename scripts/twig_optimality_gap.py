"""Exact optimality gap of greedy influence minimisation on small random instances (3 families x 100).

For each instance the expected spread (beyond the seeds) is computed EXACTLY (TWIG), greedy blocking uses the
exact oracle as its gain function, and the optimum is found by exhaustive search over all blocking sets of size
<= budget. Instances are drawn with a fixed seed; per-instance results go to results/theory/twig_optimality_gap.csv.

    python scripts/twig_optimality_gap.py
"""
from __future__ import annotations

import itertools, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
import numpy as np

from trace_e.blocking.treewidth import ExactSpreadFast, ExactSpread, brute_force_spread, greedy_exact, elimination_order

N_PER_FAMILY, SEED, MAX_WIDTH = 100, 20261, 8
FAMILIES = {"sparse": (2.0, 3.5, True), "medium": (0.25, 0.5, False), "dense": (0.5, 0.8, False)}  # (lo, hi, lo/hi are mean degrees?)


def realized_width(G, order):
    """max_i |F_i| - 1 for the schedule actually run (the reverse of the elimination order)."""
    o = list(reversed(order)); pos = {v: i for i, v in enumerate(o)}
    last = {v: max([pos[v]] + [pos[u] for u in G.neighbors(v)]) for v in o}
    return max(sum(1 for v in o[: i + 1] if last[v] >= i) for i in range(len(o))) - 1


def instance(rng, fam):
    lo, hi, by_degree = FAMILIES[fam]
    while True:
        n = rng.randint(6, 14)
        p = rng.uniform(lo, hi) / (n - 1) if by_degree else rng.uniform(lo, hi)
        G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
        if G.number_of_edges() == 0:
            continue
        order = elimination_order(G)
        w = realized_width(G, order)
        if w > MAX_WIDTH:
            continue
        W = {frozenset(e): rng.uniform(0.1, 0.9) for e in G.edges}
        wfn = lambda u, v, W=W: W[frozenset((u, v))]
        seeds = rng.sample(list(G.nodes), k=rng.randint(1, min(4, n - 2)))
        budget = rng.randint(1, min(4, n - len(seeds)))
        return G, wfn, seeds, budget, order, w


def main():
    rows = []
    for fam in FAMILIES:
        rng = random.Random(SEED + len(fam))
        for i in range(N_PER_FAMILY):
            G, wfn, seeds, b, order, w = instance(rng, fam)
            t0 = time.time()
            es = ExactSpreadFast(G, wfn, seeds, order=order)
            none = es.evaluate(frozenset())
            Bg, Fg = greedy_exact(G, wfn, seeds, b, order=order)
            pool = [v for v in G.nodes if v not in seeds]
            best, Bo = none, ()
            for k in range(1, b + 1):
                for B in itertools.combinations(pool, k):
                    val = es.evaluate(frozenset(B))
                    if val < best - 1e-12:
                        best, Bo = val, B
            ok = ""  # independent check by enumeration of all live-edge subsets, where feasible
            if G.number_of_edges() <= 15:
                chk = brute_force_spread(G, wfn, seeds, frozenset(Bo))
                ok = "ok" if abs(chk - best) < 1e-8 else f"MISMATCH({chk:.6f})"
            exc = (Fg - best) / best if best > 1e-12 else 0.0
            rows.append(dict(family=fam, i=i, n=G.number_of_nodes(), m=G.number_of_edges(), width=w, n_seeds=len(seeds), budget=b, spread_none=none,
                             spread_greedy=Fg, spread_opt=best, greedy_set=list(Bg), opt_set=list(Bo), excess=exc, check=ok, sec=time.time() - t0))
        print(fam, "done", flush=True)
    import csv
    os.makedirs("results/theory", exist_ok=True)
    with open("results/theory/twig_optimality_gap.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    for fam in list(FAMILIES) + ["all"]:
        ex = np.array([r["excess"] for r in rows if fam in ("all", r["family"])])
        print(f"{fam:7s} n={len(ex)} optimal={int((ex < 1e-9).sum())} mean_excess={ex.mean():.4f} max_excess={ex.max():.4f} "
              f"suboptimal_excesses={np.round(np.sort(ex[ex >= 1e-9]), 3).tolist()}")
    print("mismatches:", sum(1 for r in rows if r["check"].startswith("MISMATCH")), "of checked:", sum(1 for r in rows if r["check"]))


if __name__ == "__main__":
    main()
