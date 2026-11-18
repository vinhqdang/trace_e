"""Feasibility with the best layout order, in exactly the configuration of scripts/twig_feasibility.py
(one seed = node 0, edge probabilities uniform on [0.05, 0.4], fixed RNG, 600 s and 8 GB limits), for a like-for-like comparison.
Appends results to results/theory/twig_feasibility.json under 'best_order'.
"""
from __future__ import annotations

import json, multiprocessing as mp, os, random, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
from trace_e.blocking.layout import layout_to_order
from trace_e.blocking.treewidth import ExactSpreadFast
from trace_e.blocking.layout import best_layout
from trace_e.graphs import load_empirical

TIME_LIMIT, MEM_GB = 600, 8


def _run(name, q):
    resource.setrlimit(resource.RLIMIT_AS, (MEM_GB * 2**30, MEM_GB * 2**30))
    G = nx.convert_node_labels_to_integers(load_empirical(name)); rng = random.Random(1)
    W = {frozenset(e): rng.uniform(0.05, 0.4) for e in G.edges}; wfn = lambda u, v: W[frozenset((u, v))]
    lay, w = best_layout(G); t0 = time.time()
    try:
        v = ExactSpreadFast(G, wfn, [0], order=layout_to_order(lay)).evaluate(); q.put(dict(width=w, outcome="ok", seconds=time.time() - t0, expected_spread=v))
    except MemoryError:
        q.put(dict(width=w, outcome="memory limit", seconds=time.time() - t0))


def main():
    path = "results/theory/twig_feasibility.json"; out = json.load(open(path)); out["best_order"] = {}
    for name in ("karate", "iceland", "dolphin"):
        ctx = mp.get_context("fork"); q = ctx.Queue(); p = ctx.Process(target=_run, args=(name, q)); p.start(); p.join(TIME_LIMIT)
        if p.is_alive(): p.terminate(); rec = dict(outcome="time limit", seconds=TIME_LIMIT)
        else:
            try: rec = q.get(timeout=5)
            except Exception: rec = dict(outcome="process died (memory limit or crash)")
        out["best_order"][name] = rec; print(name, rec, flush=True)
    json.dump(out, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
