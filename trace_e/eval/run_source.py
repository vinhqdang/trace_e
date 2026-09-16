"""Evaluation runner for source detection (Problem A).

Example::

    python -m trace_e.eval.run_source --network karate --methods random,jordan,rumor,netsleuth,dmp,sme,mcs,gcn

Protocol (following the Sterchi et al. benchmark): continuous-time SIR with
recovery rate ``nu``, infection rate ``beta`` perturbed per outbreak by a
log-normal with std ``sigma`` (training set), snapshot at time ``T``.
Training set: ``train_sims`` outbreaks per source node; test set:
``test_sims`` outbreaks per node with ``test_sigma`` (0 by default) and a
fixed test seed. Results go to ``results/summary_source.csv`` (one row per
method) and ``results/runs/<run_id>/`` (config, log, per-instance JSONL).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..graphs import load_network, graph_stats, adjacency
from ..logging_utils import RunLogger, RESULTS_DIR
from ..metrics import evaluate_instance, summarise, ResistanceScore
from ..simulate import simulate_dataset
from ..source import Context, get_detector, REGISTRY

DEFAULT_METHODS = "random,degree,jordan,distance,rumor,netsleuth,dmp,sme,mcs,mlp,gcn"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True, help="e.g. karate, dolphin, powergrid, er:200, ba:500:m=2")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--beta", type=float, default=1.3)
    p.add_argument("--sigma", type=float, default=0.5, help="log-normal std of beta in training simulations")
    p.add_argument("--T", type=float, default=0.85)
    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--train-sims", type=int, default=10, help="training outbreaks per node")
    p.add_argument("--test-sims", type=int, default=100, help="test outbreaks per node")
    p.add_argument("--test-sigma", type=float, default=0.0)
    p.add_argument("--max-test", type=int, default=0, help="cap on number of test instances (0 = all)")
    p.add_argument("--min-outbreak", type=int, default=1, help="drop test outbreaks with fewer infected nodes (1 keeps singletons, as in the benchmark)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--test-seed", type=int, default=4253219)
    p.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--dmp-dt", type=float, default=0.05)
    p.add_argument("--no-resistance", action="store_true", help="skip resistance score (needs dense pinv)")
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args(argv)


def build_detector(name, ctx, args, log):
    kw = {}
    if name in ("gcn", "gcn_skip", "mlp", "igcn"):
        kw = dict(epochs=args.epochs, hidden=args.hidden, layers=args.layers, log=log)
    if name == "dmp":
        kw = dict(dt=args.dmp_dt)
    return get_detector(name, ctx, **kw)


def main(argv=None):
    args = parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; available: {sorted(REGISTRY)}")

    tag = args.tag or args.network.replace(":", "-")
    rl = RunLogger("source", tag, results_dir=args.results_dir or RESULTS_DIR, config=vars(args))
    log = rl.log

    G = load_network(args.network, seed=args.seed)
    st = graph_stats(G)
    log.info("network %s: %s", args.network, st)
    ctx = Context.from_graph(G, beta=args.beta, T=args.T, nu=args.nu, sigma=args.sigma, n_jobs=args.n_jobs, seed=args.seed)

    # ---- simulations ----
    t0 = time.time()
    need_train = any(REGISTRY[m].needs_training_sims for m in methods)
    train = None
    if need_train:
        train = simulate_dataset(ctx.g, args.beta / args.nu, args.sigma, args.T / args.nu, args.train_sims, seed=args.seed, n_jobs=args.n_jobs)
        log.info("train: %d outbreaks, avg size %.2f, singletons %d (%.1fs)", len(train), train.outbreak_sizes.mean(),
                 (train.outbreak_sizes == 1).sum(), time.time() - t0)
    t1 = time.time()
    test = simulate_dataset(ctx.g, args.beta / args.nu, args.test_sigma, args.T / args.nu, args.test_sims, seed=args.test_seed, n_jobs=args.n_jobs)
    if args.min_outbreak > 1:
        keep = test.outbreak_sizes >= args.min_outbreak
        test.states, test.sources, test.betas = test.states[keep], test.sources[keep], test.betas[keep]
    if args.max_test and len(test) > args.max_test:
        rng = np.random.default_rng(args.test_seed)
        keep = np.sort(rng.choice(len(test), size=args.max_test, replace=False))
        test.states, test.sources, test.betas = test.states[keep], test.sources[keep], test.betas[keep]
    sim_time = time.time() - t0
    log.info("test: %d outbreaks, avg size %.2f, singletons %d (%.1fs)", len(test), test.outbreak_sizes.mean(),
             (test.outbreak_sizes == 1).sum(), time.time() - t1)

    resistance = None
    if not args.no_resistance and ctx.n <= 1500:
        resistance = ResistanceScore(adjacency(G).toarray())

    # ---- methods ----
    for name in methods:
        log.info("=== method %s ===", name)
        det = build_detector(name, ctx, args, log)
        tf = time.time()
        det.fit(train if det.needs_training_sims else None)
        fit_time = time.time() - tf
        rng = np.random.default_rng(args.seed)
        rows = []
        ti = time.time()
        with rl.instance_writer(name) as fh:
            # batch scoring for methods that support it efficiently
            if name in ("gcn", "gcn_skip", "mlp", "igcn"):
                scores_all = det.score_batch(test.states)
                get = lambda i: scores_all[i]  # noqa: E731
            else:
                get = lambda i: det.score(test.states[i])  # noqa: E731
            for i in range(len(test)):
                sc = get(i)
                r = evaluate_instance(ctx.g, sc, int(test.sources[i]), rng, det.probabilistic, resistance)
                r.update({"i": i, "source": int(test.sources[i]), "n_inf": int((test.states[i] != 0).sum())})
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                if (i + 1) % args.log_every == 0:
                    part = summarise(rows)
                    log.info("[%s] %d/%d top1=%.3f top5=%.3f ed=%.3f rr=%.3f (%.0fs)", name, i + 1, len(test), part["top1"], part["top5"], part["ed"], part["rr"], time.time() - ti)
        infer_time = (time.time() - ti) / len(test)
        summ = summarise(rows)
        row = dict(network=args.network, n_nodes=st["n_nodes"], n_edges=st["n_edges"], method=name, beta=args.beta, sigma=args.sigma, T=args.T, nu=args.nu,
                   train_sims=args.train_sims if det.needs_training_sims else 0, test_sims=args.test_sims, test_sigma=args.test_sigma, seed=args.seed,
                   avg_outbreak=float(test.outbreak_sizes.mean()), fit_time_s=round(fit_time, 3), infer_time_per_instance_s=round(infer_time, 6),
                   sim_time_s=round(sim_time, 2), notes=(args.notes + (f" min_outbreak={args.min_outbreak}" if args.min_outbreak > 1 else "")).strip())
        row.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in summ.items()})
        rl.write_summary(row)
        log.info("RESULT %s %s: top1=%.4f top3=%.4f top5=%.4f ed=%.4f rr=%.4f%s fit=%.1fs infer=%.5fs/inst", args.network, name,
                 summ["top1"], summ["top3"], summ["top5"], summ["ed"], summ["rr"],
                 f" css={summ['css']:.2f}" if "css" in summ else "", fit_time, infer_time)
    rl.finish()
    return rl.run_dir


if __name__ == "__main__":
    main()
