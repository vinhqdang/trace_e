"""Measured adaptivity gap on the star-of-paths family of Theorem 2 (docs/DEFER.md).

For each (Delta, p): seed s with Delta children (edge probability p), each
child heading a private deterministic path of length L. Budget 1. Compares
the best one-shot blocker (a branch head), DEFER, and the theoretical
ratio (1-(1-p)^Delta)/p of savings.

    python scripts/adaptivity_gap.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np

from trace_e.graphs import CSRGraph
from trace_e.logging_utils import RESULTS_DIR
from trace_e.sequential.containers import get_container
from trace_e.sequential.simulator import Episode


def instance(Delta, L, p):
    G = nx.DiGraph()
    nid = 1
    for _ in range(Delta):
        a = nid
        nid += 1
        G.add_edge(0, a, w=p)
        prev = a
        for _ in range(L):
            G.add_edge(prev, nid, w=1.0)
            prev = nid
            nid += 1
    A = nx.to_scipy_sparse_array(G, nodelist=range(nid), weight="w", format="csr", dtype=float)
    return CSRGraph(n=nid, indptr=A.indptr.astype(np.int64), indices=A.indices.astype(np.int64), weights=A.data.astype(float))


def main():
    L, reps = 30, 400
    rows = []
    for Delta in [4, 8, 16, 32]:
        for p in [0.02, 0.05, 0.1, 0.2]:
            g = instance(Delta, L, p)
            one, defer, none = [], [], []
            for r in range(reps):
                ep = Episode(g, 1.0, [0], np.random.default_rng(r))
                ep.block([1])
                while ep.alive:
                    ep.step()
                one.append(ep.harm - 1)
                ep = Episode(g, 1.0, [0], np.random.default_rng(r))
                con = get_container("adaptive", n_samples=30, replan_every=1)
                con.reset(1)
                while ep.alive:
                    ep.intervene(con.act(ep, 1.0))
                    ep.step()
                defer.append(ep.harm - 1)
                ep = Episode(g, 1.0, [0], np.random.default_rng(r))
                while ep.alive:
                    ep.step()
                none.append(ep.harm - 1)
            s1 = np.mean(none) - np.mean(one)
            s2 = np.mean(none) - np.mean(defer)
            row = {"Delta": Delta, "p": p, "saved_one_shot": float(s1), "saved_defer": float(s2), "measured_ratio": float(s2 / max(s1, 1e-9)),
                   "theory_ratio": float((1 - (1 - p) ** Delta) / p), "spread_none": float(np.mean(none))}
            rows.append(row)
            print(json.dumps(row))
    os.makedirs(os.path.join(RESULTS_DIR, "theory"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "theory", "adaptivity_gap.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
