"""Star-of-paths benchmark (Theorem 2 family): all policies on identical realisations, paired comparison.

Seed 0 -> Delta gateways (edge prob p), each heading a private deterministic path of L vertices.
    python scripts/star_of_paths_bench.py
"""
from __future__ import annotations

import json, multiprocessing as mp, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scripts.adaptivity_gap import instance
from trace_e.blocking.ilp import ILPController
from trace_e.blocking.imin import run_algorithm
from trace_e.logging_utils import RESULTS_DIR
from trace_e.sequential.containers import get_container
from trace_e.sequential.simulator import Episode

POLICIES = ["none", "ag", "gr", "commit", "defer_nopush", "defer", "ilp1", "ilp_roll", "defer_ilp"]
THETA, K = 100, 48


def run(task):
    Delta, L, p, b, r = task
    g = instance(Delta, L, p)
    seeds = [0]
    out = {}
    plans = {pol: run_algorithm(pol, g, seeds, b, theta=THETA, seed=r)[0] for pol in ("ag", "gr")}
    for pol in POLICIES:
        ep = Episode(g, 1.0, seeds, np.random.default_rng(10_000 + r))
        kw = dict(n_samples=THETA, horizon=0, replan_every=1, seed=r)
        con = None
        if pol in plans:
            ep.block(plans[pol])
        elif pol == "commit":
            con = get_container("adaptive_commit", planner="imin:ag", **kw)
        elif pol == "defer":
            con = get_container("adaptive", planner="imin:ag", initial_plan=plans["ag"], **kw)
        elif pol == "defer_nopush":
            con = get_container("adaptive", planner="imin:ag", initial_plan=plans["ag"], pushdown=False, **kw)
        elif pol == "ilp1":
            con = ILPController("ilp1", K=K, time_limit=20, seed=r)
        elif pol == "ilp_roll":
            con = ILPController("ilp_roll", K=K, H=3, M=12, time_limit=20, seed=r)
        elif pol == "defer_ilp":
            f = ILPController("ilp1", K=K, time_limit=20, seed=r); f.reset(b)
            con = get_container("adaptive", planner="ilp", initial_plan=f.act(ep), ilp_k=K, ilp_tl=10.0, **kw)
        if con is not None:
            con.reset(b)
            while ep.alive:
                ep.intervene(con.act(ep, 1.0))
                ep.step()
        while ep.alive:
            ep.step()
        out[pol] = int(ep.harm - 1)
    return dict(Delta=Delta, L=L, p=p, b=b, r=r, **out)


if __name__ == "__main__":
    # Delta=16 needs far more ILP time than the smaller stars, so it gets fewer repetitions
    reps = {4: int(os.environ.get("REPS", 100)), 8: int(os.environ.get("REPS8", 50)), 16: int(os.environ.get("REPS16", 20))}
    tasks = [(D, 30, p, b, r) for D in (4, 8, 16) for p in (0.05, 0.1, 0.2) for b in (1, 2) for r in range(reps[D])]
    out = os.path.join(RESULTS_DIR, "theory", "star_of_paths_bench.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):  # resumable
        done = {(d["Delta"], d["L"], d["p"], d["b"], d["r"]) for d in map(json.loads, open(out))}
    tasks = [t for t in tasks if t not in done]
    t0 = time.time()
    with mp.get_context("fork").Pool(os.cpu_count()) as pool, open(out, "a") as fh:
        for i, row in enumerate(pool.imap_unordered(run, tasks, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(tasks)} {round(time.time() - t0)}s", flush=True)
    print("done", round(time.time() - t0), "s", len(tasks))
