"""Evaluation runner for misinformation blocking / influence limitation (Problem B).

Example::

    python -m trace_e.eval.run_blocking --network dolphin --mode block --budgets 1,2,5,10

Protocol: the graph is turned into an independent-cascade model with edge
activation probabilities from ``--prob-model`` (weighted cascade by default).
For each of ``--n-instances`` bad seed sets (``--n-seeds`` nodes each, drawn at
random with a fixed seed) and each budget, every method picks an
intervention set; the expected number of bad adopters with and without the
intervention is then estimated with ``--n-mc`` fresh Monte-Carlo cascades
(common random numbers across methods). Metrics: bad fraction after
intervention, saved fraction, selection time. ``--mode block`` removes the
chosen nodes; ``--mode counter`` seeds a competing good campaign (Budak et
al. 2011) that starts ``--good-delay`` rounds after the bad one.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..graphs import load_network, graph_stats
from ..logging_utils import RunLogger, RESULTS_DIR
from ..blocking import BlockingContext, get_blocker, REGISTRY
from ..blocking.cascade import estimate_bad_spread

DEFAULT_METHODS = "random,degree,pagerank,proximity,reach,greedy,greedy_dom"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True)
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--mode", choices=["block", "counter"], default="block")
    p.add_argument("--prob-model", choices=["wc", "const", "tri"], default="wc")
    p.add_argument("--p", type=float, default=0.1, help="activation probability for --prob-model const")
    p.add_argument("--budgets", default="1,2,5,10")
    p.add_argument("--n-seeds", type=int, default=1, help="size of the bad seed set")
    p.add_argument("--n-instances", type=int, default=20)
    p.add_argument("--n-mc", type=int, default=1000, help="Monte-Carlo cascades for the final evaluation")
    p.add_argument("--greedy-samples", type=int, default=200)
    p.add_argument("--greedy-pool", type=int, default=0, help="restrict greedy candidates to the top-k reachable nodes by degree (0 = all)")
    p.add_argument("--good-delay", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.mode == "counter":
        methods = [m for m in methods if m != "greedy_dom"]
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; available: {sorted(REGISTRY)}")
    budgets = [int(b) for b in args.budgets.split(",")]
    tag = args.tag or f"{args.network.replace(':', '-')}_{args.mode}"
    rl = RunLogger("blocking", tag, results_dir=args.results_dir or RESULTS_DIR, config=vars(args))
    log = rl.log

    G = load_network(args.network, seed=args.seed)
    st = graph_stats(G)
    log.info("network %s: %s", args.network, st)
    ctx = BlockingContext.from_graph(G, prob_model=args.prob_model, p=args.p, mode=args.mode, good_delay=args.good_delay, seed=args.seed, n_jobs=args.n_jobs)

    rng = np.random.default_rng(args.seed)
    deg = ctx.g.degree()
    eligible = np.flatnonzero(deg > 0)
    instances = [np.sort(rng.choice(eligible, size=args.n_seeds, replace=False)) for _ in range(args.n_instances)]

    # baseline spread without intervention (same MC seed for every method -> common random numbers)
    t0 = time.time()
    base = np.array([estimate_bad_spread(ctx.g, s, [], ctx.mode, args.n_mc, seed=args.seed + 1000 + i, good_delay=args.good_delay) for i, s in enumerate(instances)])
    log.info("no-intervention spread: mean %.2f nodes (%.3f of graph) over %d instances (%.1fs)", base.mean(), base.mean() / ctx.n, len(instances), time.time() - t0)

    blockers = {}
    for name in methods:
        kw = {"n_samples": args.greedy_samples, "candidate_pool": args.greedy_pool} if name in ("greedy", "proposed") else ({"n_samples": args.greedy_samples} if name == "greedy_dom" else {})
        b = get_blocker(name, ctx, **kw)
        tp = time.time()
        b.prepare()
        blockers[name] = (b, time.time() - tp)

    for name in methods:
        blk, prep_time = blockers[name]
        log.info("=== method %s (%s mode) ===", name, args.mode)
        rows_by_budget = {k: [] for k in budgets}
        with rl.instance_writer(name) as fh:
            for i, seeds in enumerate(instances):
                # every blocker returns a ranking whose prefixes are its solutions for smaller budgets
                ts = time.time()
                chosen_max = blk.select(seeds, max(budgets))
                sel_time = time.time() - ts
                for k in budgets:
                    chosen = chosen_max[:k]
                    after = estimate_bad_spread(ctx.g, seeds, chosen, ctx.mode, args.n_mc, seed=args.seed + 1000 + i, good_delay=args.good_delay)
                    r = {"i": i, "budget": k, "seeds": [int(x) for x in seeds], "chosen": [int(x) for x in chosen],
                         "base": float(base[i]), "after": float(after), "saved_frac": float(1 - after / base[i]) if base[i] > 0 else 0.0,
                         "bad_frac": float(after / ctx.n), "select_time": sel_time}
                    rows_by_budget[k].append(r)
                    fh.write(json.dumps(r) + "\n")
                if (i + 1) % 5 == 0:
                    log.info("[%s] %d/%d instances, saved@%d=%.3f", name, i + 1, len(instances), budgets[-1], np.mean([r["saved_frac"] for r in rows_by_budget[budgets[-1]]]))
        for k in budgets:
            rows = rows_by_budget[k]
            saved = np.array([r["saved_frac"] for r in rows])
            bad = np.array([r["bad_frac"] for r in rows])
            sel = np.array([r["select_time"] for r in rows])
            row = dict(network=args.network, n_nodes=st["n_nodes"], n_edges=st["n_edges"], method=name,
                       beta=args.p if args.prob_model == "const" else "", sigma="", T=k, nu="",
                       train_sims=args.greedy_samples if name in ("greedy", "proposed") else 0, test_sims=args.n_mc, test_sigma="", seed=args.seed,
                       n_instances=len(rows), avg_outbreak=float(base.mean()),
                       top1=round(float(saved.mean()), 5), top1_se=round(float(saved.std(ddof=1) / np.sqrt(len(saved))) if len(saved) > 1 else 0.0, 5),
                       top3=round(float(bad.mean()), 5), ed=round(float(np.mean([r["after"] for r in rows])), 4),
                       fit_time_s=round(prep_time, 3), infer_time_per_instance_s=round(float(sel.mean()), 4), sim_time_s="",
                       notes=(f"mode={args.mode} prob={args.prob_model} n_seeds={args.n_seeds} budget={k}; top1=saved_frac top3=bad_frac ed=bad_count " + args.notes).strip())
            rl.write_summary(row)
            log.info("RESULT %s %s %s budget=%d: saved=%.4f (se %.4f) bad_frac=%.4f bad_count=%.2f select=%.3fs", args.network, args.mode, name, k, saved.mean(), row["top1_se"], bad.mean(), row["ed"], sel.mean())
    rl.finish()
    return rl.run_dir


if __name__ == "__main__":
    main()
