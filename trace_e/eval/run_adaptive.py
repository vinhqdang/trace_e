"""Adaptive influence minimisation benchmark: deferred frontier blocking vs one-shot blocking.

Example::

    python -m trace_e.eval.run_adaptive --network cahepth --prob-model const --p 0.1 --n-seeds 20 --budgets 20,50,100

For each seed-set instance and each of ``--draws`` live-edge realisations,
every policy runs on the same realisation with the same total budget:

* one-shot plans computed at time 0 with the seeds known (``ag``, ``gr``,
  ``lsbm``, ``isocut``, ``proximity``, ``degree``, ``random``);
* adaptive policies that observe activations round by round:
  ``defer`` (DEFER: plan with AdvancedGreedy on the current frontier, commit
  only planned nodes exposed to the frontier that pass the push-down rule,
  re-plan each round), ``defer_nopush`` (no push-down), ``defer_gr`` (GreedyReplace
  as the planner), ``defer_cut`` (dominator planner with isolation moves),
  ``commit`` (re-plan and commit the whole plan each round; ablation).

Metrics: final spread (bad nodes beyond the seeds), saved fraction vs no
intervention, budget actually used, wall-clock per episode. Results go to
results/summary_adaptive.csv and results/runs/<run_id>/.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from ..blocking import BlockingContext, get_blocker
from ..blocking.imin import run_algorithm
from ..graphs import load_network, graph_stats
from ..logging_utils import RunLogger, RESULTS_DIR, append_csv, git_commit
from ..sequential.containers import get_container
from ..sequential.simulator import Episode

COLUMNS = ["timestamp", "run_id", "network", "n_nodes", "n_edges", "prob_model", "p", "n_seeds", "seed_rule", "budget", "policy",
           "n_instances", "draws", "spread_none", "spread", "spread_se", "saved_frac", "budget_used", "time_s_per_episode", "theta", "seed", "git_commit", "notes"]

ONE_SHOT = {"ag", "gr", "lsbm", "isocut", "isocut_plus", "cutgreedy", "swap", "swap_gr", "swap_first"}
HEURISTIC = {"proximity", "degree", "random", "pagerank"}
_G = {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True)
    p.add_argument("--prob-model", choices=["wc", "const", "tri", "bimodal"], default="const")
    p.add_argument("--p", type=float, default=0.1)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed-rule", choices=["random", "degree"], default="random", help="random seeds or the highest-degree nodes")
    p.add_argument("--budgets", default="20,50,100")
    p.add_argument("--n-instances", type=int, default=10)
    p.add_argument("--draws", type=int, default=5, help="live-edge realisations per instance")
    p.add_argument("--theta", type=int, default=100)
    p.add_argument("--policies", default="none,ag,gr,lsbm,proximity,degree,defer,defer_gr,commit")
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def _episode(task):
    inst, draw, policy, budget, seeds, ep_seed = task
    g = _G["g"]
    ep = Episode(g, 1.0, seeds, np.random.default_rng(ep_seed))
    t0 = time.perf_counter()
    used = 0
    if policy == "none":
        pass
    elif policy in ONE_SHOT:
        B = _G["plans"][(inst, policy, budget)]
        used = len(ep.block(B))
    elif policy in HEURISTIC:
        B = _G["plans"][(inst, policy, budget)]
        used = len(ep.block(B))
    else:
        kw = dict(n_samples=_G["theta"], horizon=0, replan_every=1, seed=ep_seed)
        plans = _G["plans"]
        ag_plan = plans.get((inst, "ag", budget))
        gr_plan = plans.get((inst, "gr", budget))
        if policy in ("defer", "defer_ag"):  # DEFER wrapping AdvancedGreedy's own solution
            con = get_container("adaptive", planner="imin:ag", initial_plan=ag_plan, **kw)
        elif policy == "defer_nopush":
            con = get_container("adaptive", planner="imin:ag", initial_plan=ag_plan, pushdown=False, **kw)
        elif policy == "defer_fresh":  # own round-0 plan from the container's samples
            con = get_container("adaptive", planner="imin:ag", **kw)
        elif policy == "defer_cut":
            con = get_container("adaptive", planner="dominator", initial_plan=plans.get((inst, "isocut", budget)), **kw)
        elif policy == "defer_gr":
            con = get_container("adaptive", planner="imin:gr", initial_plan=gr_plan, **kw)
        elif policy == "defer_swap":  # DEFER wrapping SWAP's solution
            con = get_container("adaptive", planner="imin:ag", initial_plan=plans.get((inst, "swap", budget)), **kw)
        elif policy == "defer_gr_nopush":
            con = get_container("adaptive", planner="imin:gr", initial_plan=gr_plan, pushdown=False, **kw)
        elif policy == "commit":  # ablation: fresh plan every round, committed in full (no deferral, no protection)
            con = get_container("adaptive_commit", planner="imin:ag", **kw)
        elif policy == "defer_h4":
            kw["horizon"] = 4
            con = get_container("adaptive", **kw)
        else:
            raise ValueError(policy)
        con.reset(budget)
        rounds = 0
        while ep.alive and rounds < _G["max_rounds"]:
            used += len(ep.intervene(con.act(ep, 1.0)))
            ep.step()
            rounds += 1
    while ep.alive:
        ep.step()
    return {"inst": inst, "draw": draw, "policy": policy, "budget": budget, "spread": int(ep.harm - len(seeds)), "used": used,
            "time": time.perf_counter() - t0}


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or f"{args.network.replace(':', '-')}_{args.prob_model}"
    rl = RunLogger("adaptive", tag, results_dir=args.results_dir or RESULTS_DIR, config=vars(args))
    log = rl.log
    G = load_network(args.network, seed=args.seed)
    st = graph_stats(G)
    ctx = BlockingContext.from_graph(G, prob_model=args.prob_model, p=args.p, mode="block", seed=args.seed)
    g = ctx.g
    log.info("network %s: %s", args.network, st)
    rng = np.random.default_rng(args.seed)
    deg = g.degree()
    elig = np.flatnonzero(deg > 0)
    instances = []
    for i in range(args.n_instances):
        if args.seed_rule == "degree":
            top = np.argsort(-deg, kind="stable")[: args.n_seeds * 3]
            instances.append(np.sort(rng.choice(top, size=args.n_seeds, replace=False)))
        else:
            instances.append(np.sort(rng.choice(elig, size=args.n_seeds, replace=False)))
    budgets = [int(b) for b in args.budgets.split(",")]
    policies = [p for p in args.policies.split(",") if p]
    # adaptive wrappers need the one-shot plans they start from
    need = set()
    if any(p in ("defer", "defer_ag", "defer_nopush", "commit") for p in policies):
        need.add("ag")
    if any(p.startswith("defer_gr") for p in policies):
        need.add("gr")
    if "defer_cut" in policies:
        need.add("isocut")
    if "defer_swap" in policies:
        need.add("swap")
    plan_only = sorted(need - set(policies))
    # one-shot plans (computed once per instance and budget)
    plans = {}
    plan_time = {}
    for i, seeds in enumerate(instances):
        for pol in policies + plan_only:
            if pol in ONE_SHOT:
                for b in budgets:
                    B, info = run_algorithm(pol, g, seeds, b, theta=args.theta, seed=args.seed + i)
                    plans[(i, pol, b)] = B
                    plan_time[(pol, b)] = plan_time.get((pol, b), 0.0) + info["time_s"]
            elif pol in HEURISTIC:
                blk = get_blocker(pol, ctx)
                blk.prepare()
                for b in budgets:
                    t0 = time.perf_counter()
                    plans[(i, pol, b)] = blk.select(seeds, b)
                    plan_time[(pol, b)] = plan_time.get((pol, b), 0.0) + time.perf_counter() - t0
        log.info("plans ready for instance %d/%d", i + 1, len(instances))
    _G.update(g=g, plans=plans, theta=args.theta, max_rounds=args.max_rounds)
    tasks = []
    for i, seeds in enumerate(instances):
        for d in range(args.draws):
            ep_seed = int(rng.integers(2**31))
            for b in budgets:
                for pol in policies:
                    tasks.append((i, d, pol, b, seeds, ep_seed))
    log.info("running %d episodes", len(tasks))
    t0 = time.time()
    if args.n_jobs > 1:
        with mp.get_context("fork").Pool(args.n_jobs) as pool:
            rows = pool.map(_episode, tasks, chunksize=4)
    else:
        rows = [_episode(t) for t in tasks]
    log.info("episodes done in %.0fs", time.time() - t0)
    with rl.instance_writer("episodes") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    none = {}
    for r in rows:
        if r["policy"] == "none":
            none[(r["inst"], r["draw"])] = r["spread"]
    for b in budgets:
        base = np.mean([v for v in none.values()]) if none else float("nan")
        for pol in policies:
            rs = [r for r in rows if r["policy"] == pol and r["budget"] == b]
            if not rs:
                continue
            sp = np.array([r["spread"] for r in rs], dtype=float)
            t_ep = np.mean([r["time"] for r in rs]) + (plan_time.get((pol, b), 0.0) / (len(instances) * args.draws) if (pol, b) in plan_time else 0.0)
            row = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run_id=rl.run_id, network=args.network, n_nodes=st["n_nodes"], n_edges=st["n_edges"],
                       prob_model=args.prob_model, p=args.p, n_seeds=args.n_seeds, seed_rule=args.seed_rule, budget=b, policy=pol, n_instances=len(instances), draws=args.draws,
                       spread_none=round(float(base), 3), spread=round(float(sp.mean()), 3), spread_se=round(float(sp.std(ddof=1) / np.sqrt(len(sp))), 3) if len(sp) > 1 else 0.0,
                       saved_frac=round(float(1 - sp.sum() / max(1.0, sum(none[(r["inst"], r["draw"])] for r in rs))), 4),
                       budget_used=round(float(np.mean([r["used"] for r in rs])), 2), time_s_per_episode=round(float(t_ep), 3), theta=args.theta, seed=args.seed,
                       git_commit=git_commit(), notes=args.notes)
            append_csv(os.path.join(args.results_dir or RESULTS_DIR, "summary_adaptive.csv"), row, COLUMNS)
            log.info("RESULT %s budget=%d %-10s spread=%.2f (se %.2f) saved=%.4f used=%.1f time=%.2fs", args.network, b, pol, row["spread"], row["spread_se"], row["saved_frac"], row["budget_used"], row["time_s_per_episode"])
    rl.finish()


if __name__ == "__main__":
    main()
