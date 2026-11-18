"""Feasibility of exact evaluation on real small networks, and realised width on complete binary trees.

Runs ExactSpreadFast.evaluate() on karate / iceland / dolphin (one seed, edge probabilities U(0.05, 0.4), fixed RNG)
with a wall-clock limit and a memory limit, and records machine, widths and outcomes in results/theory/twig_feasibility.json.
    python scripts/twig_feasibility.py
"""
from __future__ import annotations

import json, math, multiprocessing as mp, os, platform, random, resource, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx

from scripts.twig_optimality_gap import realized_width
from trace_e.blocking.treewidth import ExactSpread, ExactSpreadFast, elimination_order
from trace_e.graphs import load_empirical

TIME_LIMIT, MEM_GB, VARIANT_LIMIT = 600, 8, 300


def bell(n):
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row: nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def _run(name, q, variant="fast"):
    resource.setrlimit(resource.RLIMIT_AS, (MEM_GB * 2**30, MEM_GB * 2**30))
    G = nx.convert_node_labels_to_integers(load_empirical(name))
    rng = random.Random(1)
    W = {frozenset(e): rng.uniform(0.05, 0.4) for e in G.edges}
    wfn = lambda u, v: W[frozenset((u, v))]
    t0 = time.time()
    try:
        order = elimination_order(G)
        if variant == "fast":
            es = ExactSpreadFast(G, wfn, [0])
        elif variant == "xp":  # the XP variant, same (reversed) order
            es = ExactSpread(G, wfn, [0], order=order)
        else:  # fast variant run in the forward elimination order (the constructor reverses, so pass the reverse)
            es = ExactSpreadFast(G, wfn, [0], order=list(reversed(order)))
        val = es.evaluate()
        q.put(dict(outcome="ok", seconds=time.time() - t0, expected_spread=val))
    except MemoryError:
        q.put(dict(outcome="memory limit", seconds=time.time() - t0))


def main(variants_only=False):
    out = dict(machine=platform.platform(), python=platform.python_version(), cpu_count=os.cpu_count(), time_limit_s=TIME_LIMIT, memory_limit_gb=MEM_GB, networks={}, binary_trees=[])
    if variants_only:
        out = json.load(open("results/theory/twig_feasibility.json"))
        out["karate_variants"] = {}
        for variant in ("xp", "fast_forward_order"):
            q = mp.get_context("fork").Queue(); p = mp.get_context("fork").Process(target=_run, args=("karate", q, variant)); p.start(); p.join(VARIANT_LIMIT)
            if p.is_alive():
                p.terminate(); rec = dict(outcome="time limit", seconds=VARIANT_LIMIT)
            else:
                try: rec = q.get(timeout=5)
                except Exception: rec = dict(outcome="process died (memory limit or crash)")
            out["karate_variants"][variant] = rec; print(variant, rec, flush=True)
        json.dump(out, open("results/theory/twig_feasibility.json", "w"), indent=2)
        return
    for name in ("karate", "iceland", "dolphin"):
        G = nx.convert_node_labels_to_integers(load_empirical(name))
        order = elimination_order(G)
        w = realized_width(G, order)
        rec = dict(n=G.number_of_nodes(), m=G.number_of_edges(), width=w, bell_bound=bell(w + 2))
        q = mp.get_context("fork").Queue(); p = mp.get_context("fork").Process(target=_run, args=(name, q)); p.start(); p.join(TIME_LIMIT)
        if p.is_alive():
            p.terminate(); rec.update(outcome="time limit", seconds=TIME_LIMIT)
        else:
            try: rec.update(q.get(timeout=5))
            except Exception: rec.update(outcome="process died (memory limit or crash)")
        out["networks"][name] = rec; print(name, rec, flush=True)
    for h in range(2, 8):
        T = nx.balanced_tree(2, h)
        w = realized_width(T, elimination_order(T))
        out["binary_trees"].append(dict(height=h, n=T.number_of_nodes(), realised_width=w, pathwidth_lower_bound=math.ceil(h / 2)))
    print(out["binary_trees"])
    os.makedirs("results/theory", exist_ok=True)
    json.dump(out, open("results/theory/twig_feasibility.json", "w"), indent=2)


if __name__ == "__main__":
    main(variants_only="--variants" in sys.argv)
