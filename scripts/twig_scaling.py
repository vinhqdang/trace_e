"""Scaling experiments for the exact evaluator TWIG with the frontier-layout orders.

  orders   : realised width of the reversed min-fill-in order versus the best greedy layout, on real networks and structured families
  feasible : one exact evaluation on karate / Iceland / dolphin with the best order
  n-scaling: time of one evaluation versus n at (almost) fixed width (grid strips 3 x m, random trees, power-grid balls)
  w-scaling: time versus width on grid strips k x m with n ~ 240

Each evaluation runs in a child process with a wall-clock limit and an address-space limit. Results: results/theory/twig_scaling.json
    python scripts/twig_scaling.py
"""
from __future__ import annotations

import json, multiprocessing as mp, os, random, resource, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx

from trace_e.blocking.layout import best_layout, frontier_width, layout_to_order
from trace_e.blocking.treewidth import ExactSpreadFast, elimination_order
from trace_e.graphs import load_empirical

LIMIT, MEM_GB = 300, 8


def weights(G, seed=1):
    rng = random.Random(seed); W = {frozenset(e): rng.uniform(0.2, 0.8) for e in G.edges}
    return lambda u, v: W[frozenset((u, v))]


def _eval(G, order, q):
    resource.setrlimit(resource.RLIMIT_AS, (MEM_GB * 2**30, MEM_GB * 2**30))
    t0 = time.time()
    try:
        v = ExactSpreadFast(G, weights(G), [next(iter(G.nodes))], order=order).evaluate()
        q.put(dict(outcome="ok", seconds=time.time() - t0, spread=v))
    except MemoryError:
        q.put(dict(outcome="memory limit", seconds=time.time() - t0))


def timed(G, order):
    ctx = mp.get_context("fork"); q = ctx.Queue(); p = ctx.Process(target=_eval, args=(G, order, q)); p.start(); p.join(LIMIT)
    if p.is_alive():
        p.terminate(); return dict(outcome="time limit", seconds=LIMIT)
    try: return q.get(timeout=5)
    except Exception: return dict(outcome="process died (memory limit or crash)")


def best_order(G):
    lay, w = best_layout(G); mf = elimination_order(G); wm = frontier_width(G, list(reversed(mf)))
    return (layout_to_order(lay), w, wm) if w < wm else (mf, wm, wm)


def named():
    L = lambda nm: nx.convert_node_labels_to_integers(load_empirical(nm))
    g = {nm: L(nm) for nm in ("karate", "iceland", "dolphin", "workplace", "fraternity", "airtraffic", "powergrid")}
    g.update({"grid 3x200": nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 200)), "grid 4x200": nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 200)),
              "grid 6x200": nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 200)), "random tree (500)": nx.random_labeled_tree(500, seed=1) if hasattr(nx, "random_labeled_tree") else nx.random_tree(500, seed=1),
              "complete binary tree (511)": nx.balanced_tree(2, 8), "Watts-Strogatz k=4, p=0.02 (500)": nx.connected_watts_strogatz_graph(500, 4, 0.02, seed=1)})
    return g


def ball(G, root, size):
    seen = [root]; S = {root}; i = 0
    while i < len(seen) and len(seen) < size:
        for u in G.neighbors(seen[i]):
            if u not in S and len(seen) < size: S.add(u); seen.append(u)
        i += 1
    return nx.convert_node_labels_to_integers(G.subgraph(seen).copy())


def main():
    out = dict(limit_s=LIMIT, memory_gb=MEM_GB, orders=[], feasible={}, n_scaling=[], w_scaling=[], powergrid_balls=[])
    for nm, G in named().items():
        mf = frontier_width(G, list(reversed(elimination_order(G)))); t0 = time.time(); lay, w = best_layout(G)
        out["orders"].append(dict(graph=nm, n=G.number_of_nodes(), m=G.number_of_edges(), width_min_fill_reversed=mf, width_best_layout=w, layout_seconds=round(time.time() - t0, 2))); print(out["orders"][-1], flush=True)
    for nm in ("karate", "iceland", "dolphin"):
        G = nx.convert_node_labels_to_integers(load_empirical(nm)); order, w, _ = best_order(G)
        out["feasible"][nm] = dict(n=G.number_of_nodes(), width=w, **timed(G, order)); print(nm, out["feasible"][nm], flush=True)
    for m in (50, 100, 200, 400, 800, 1600):
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, m)); order, w, _ = best_order(G)
        out["n_scaling"].append(dict(family="grid 3 x m", n=G.number_of_nodes(), width=w, **timed(G, order))); print(out["n_scaling"][-1], flush=True)
    for n in (100, 200, 400, 800, 1600):
        G = nx.random_labeled_tree(n, seed=2) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=2); order, w, _ = best_order(G)
        out["n_scaling"].append(dict(family="random tree", n=n, width=w, **timed(G, order))); print(out["n_scaling"][-1], flush=True)
    for k in (2, 3, 4, 5, 6, 7, 8):
        m = 240 // k; G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(k, m)); order, w, _ = best_order(G)
        out["w_scaling"].append(dict(family="grid k x m", k=k, n=G.number_of_nodes(), width=w, **timed(G, order))); print(out["w_scaling"][-1], flush=True)
    P = nx.convert_node_labels_to_integers(load_empirical("powergrid")); rng = random.Random(3)
    for size in (100, 200, 400, 800):
        done = 0
        for root in rng.sample(list(P.nodes), 60):
            B = ball(P, root, size)
            if B.number_of_nodes() < size * 0.95: continue
            order, w, _ = best_order(B)
            rec = dict(n=B.number_of_nodes(), width=w)
            if w <= 12: rec.update(timed(B, order))
            else: rec.update(outcome="not run (width > 12)")
            out["powergrid_balls"].append(rec); print(rec, flush=True); done += 1
            if done >= 5: break
    os.makedirs("results/theory", exist_ok=True); json.dump(out, open("results/theory/twig_scaling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
