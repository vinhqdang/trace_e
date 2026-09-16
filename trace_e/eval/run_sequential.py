"""Benchmark for sequential detect-and-contain (Problem C, AVID and baselines).

Example::

    python -m trace_e.eval.run_sequential --network dolphin --prob-model const --p 0.15 --budget 5

A stream of cascades is simulated on the network under the independent
cascade model with baseline probabilities p0. Benign cascades use
kappa ~ U[benign-kappa-min, benign-kappa-max] (default exactly 1); harmful
cascades use kappa log-uniform in [kappa-min, kappa-max]. Each (detector,
container) pair is run interactively on the same live-edge draws. Threshold
detectors are calibrated to false-alarm level alpha on a separate set of
benign cascades (optionally with a different seed-set size, to measure
robustness to shift). Results: results/summary_sequential.csv and
results/runs/<run_id>/.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import time

import numpy as np

from ..blocking.cascade import set_edge_probabilities
from ..graphs import load_network, graph_stats, to_csr
from ..logging_utils import RunLogger, RESULTS_DIR, append_csv
from ..sequential import DETECTORS, CONTAINERS, get_detector, get_container
from ..sequential.throttlers import THROTTLERS, get_throttler
from ..sequential.policy import run_episode, max_statistic
from ..sequential.simulator import Episode, exposure

COLUMNS = ["timestamp", "run_id", "network", "n_nodes", "n_edges", "prob_model", "p", "mode", "alpha", "kappa_min", "kappa_max", "kappa_null", "p0_scale",
           "benign_kappa_min", "benign_kappa_max", "n_seeds", "calib_n_seeds", "calib_kappa", "budget", "detector", "container",
           "throttler", "rho_min", "throttle_frac", "alpha_soft", "n_benign", "n_harmful", "fa_rate", "fa_se", "det_rate", "delay_mean", "harm_at_alarm", "harmw_at_alarm", "harm_final_harmful",
           "harm_cf_harmful", "saved_frac", "benign_loss", "benign_loss_frac", "kappa_mae", "n_intervened_mean",
           "t_detect_ms", "t_contain_ms", "throttled_rounds_harmful", "throttled_rounds_benign", "throttled_nodes_benign", "threshold", "seed", "git_commit", "notes"]

_G = {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True)
    p.add_argument("--prob-model", choices=["wc", "const", "tri"], default="wc")
    p.add_argument("--p", type=float, default=0.1)
    p.add_argument("--mode", choices=["block", "counter"], default="block")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--kappa-min", type=float, default=1.5)
    p.add_argument("--kappa-max", type=float, default=6.0)
    p.add_argument("--benign-kappa-min", type=float, default=1.0)
    p.add_argument("--benign-kappa-max", type=float, default=1.0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--calib-n-seeds", type=int, default=0, help="seed-set size used to calibrate threshold baselines (0 = same as --n-seeds)")
    p.add_argument("--calib-kappa", type=float, default=None, help="fix the benign multiplier used for calibration (default: the deployment benign distribution)")
    p.add_argument("--n-benign", type=int, default=300)
    p.add_argument("--n-harmful", type=int, default=300)
    p.add_argument("--n-calib", type=int, default=300)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--max-rounds", type=int, default=60)
    p.add_argument("--detectors", default="eprocess,sprt,cusum,size,growth,excess,logistic")
    p.add_argument("--containers", default="adaptive,greedy,proximity,degree")
    p.add_argument("--pairs", default="", help="explicit detector:container[:throttler] triples, comma separated (overrides the default grid)")
    p.add_argument("--rho-min", type=float, default=0.2, help="throttle factor applied to selected exposures")
    p.add_argument("--throttle-frac", type=float, default=0.5, help="fraction of the exposure mass to throttle (hub/random throttlers)")
    p.add_argument("--alpha-soft", type=float, default=0.5, help="soft evidence gate 1/alpha_soft for throttling (1 = always on)")
    p.add_argument("--default-container", default="greedy")
    p.add_argument("--sprt-kappa", type=float, default=2.0)
    p.add_argument("--grid-eta", type=float, default=0.25)
    p.add_argument("--kappa-null", type=float, default=1.0, help="inflated null multiplier for a robustness margin")
    p.add_argument("--p0-scale", type=float, default=1.0, help="misspecification: detectors use p0 * p0-scale while cascades follow p0 (values < 1 = baseline underestimated)")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--pool", type=int, default=300)
    p.add_argument("--horizon", type=int, default=0, help="planning horizon in live hops for the dominator planner (0 = unlimited)")
    p.add_argument("--replan-every", type=int, default=1, help="adaptive container re-plans every k rounds")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--tag", default="")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def make_detector(name, args):
    if name == "eprocess":
        return get_detector(name, alpha=args.alpha, kappa_min=max(1.05, args.kappa_min / 1.2), kappa_max=args.kappa_max * 1.2, eta=args.grid_eta, kappa_null=args.kappa_null)
    if name in ("sprt", "cusum"):
        return get_detector(name, alpha=args.alpha, kappa=args.sprt_kappa, kappa_null=args.kappa_null)
    return get_detector(name, alpha=args.alpha)


def episode_params(i, args, rng, harmful: bool, n_seeds: int, calib: bool = False):
    elig = _G["elig"]
    seeds = rng.choice(elig, size=n_seeds, replace=False)
    if harmful:
        kappa = float(np.exp(rng.uniform(np.log(args.kappa_min), np.log(args.kappa_max))))
    else:
        kappa = float(rng.uniform(args.benign_kappa_min, args.benign_kappa_max))
        if calib and args.calib_kappa is not None:
            kappa = float(args.calib_kappa)
    return seeds, kappa, int(rng.integers(2**31))


def _run_one(task):
    i, seeds, kappa, ep_seed, harmful = task
    det = copy.deepcopy(_G["det"])
    con = copy.deepcopy(_G["con"])
    thr = copy.deepcopy(_G.get("thr"))
    ep = Episode(_G["g"], kappa, seeds, np.random.default_rng(ep_seed), mode=_G["mode"])
    r = run_episode(ep, det, con, _G["budget"], _G["max_rounds"], g_det=_G["g_det"], throttler=thr)
    r.update({"i": i, "harmful": harmful, "kappa": kappa, "seeds": [int(s) for s in seeds]})
    return r


def _calib_one(task):
    i, seeds, kappa, ep_seed = task
    det = copy.deepcopy(_G["det"])
    ep = Episode(_G["g"], kappa, seeds, np.random.default_rng(ep_seed), mode="block")
    return max_statistic(ep, det, _G["max_rounds"], g_det=_G["g_det"])


def _features_one(task):
    """Per-round feature rows for training the logistic early detector."""
    i, seeds, kappa, ep_seed, harmful = task
    det = copy.deepcopy(_G["det"])
    det.reset()
    ep = Episode(_G["g"], kappa, seeds, np.random.default_rng(ep_seed), mode="block")
    rows = []
    rounds = 0
    while ep.alive and rounds < _G["max_rounds"]:
        inactive = ~(ep.active | ep.good | ep.blocked)
        frontier = ep.frontier
        size_before = ep.harm
        new = ep.step()
        rounds += 1
        det.t += 1
        exposed, log1mq = exposure(_G["g_det"], frontier, inactive, det.kappas)
        outcomes = np.zeros(len(exposed), dtype=np.int8)
        if len(new):
            outcomes[np.isin(exposed, new)] = 1
        rows.append((det.features(log1mq, outcomes, size_before), int(harmful)))
    return rows


def pmap(fn, tasks, n_jobs):
    if n_jobs <= 1 or len(tasks) < 8:
        return [fn(t) for t in tasks]
    with mp.get_context("fork").Pool(n_jobs) as pool:
        return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (n_jobs * 8)))


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or f"{args.network.replace(':', '-')}_{args.mode}"
    rl = RunLogger("sequential", tag, results_dir=args.results_dir or RESULTS_DIR, config=vars(args))
    log = rl.log

    G = load_network(args.network, seed=args.seed)
    st = graph_stats(G)
    g = set_edge_probabilities(to_csr(G), args.prob_model, args.p, seed=args.seed)
    log.info("network %s: %s", args.network, st)
    from ..sequential.containers import scaled
    g_det = scaled(g, args.p0_scale) if args.p0_scale != 1.0 else g
    _G.update(g=g, g_det=g_det, mode=args.mode, budget=args.budget, max_rounds=args.max_rounds, elig=np.flatnonzero(g.degree() > 0))

    rng = np.random.default_rng(args.seed)
    calib_seeds = args.calib_n_seeds or args.n_seeds
    calib = [(i,) + episode_params(i, args, rng, False, calib_seeds, calib=True) for i in range(args.n_calib)]
    calib_h = [(i,) + episode_params(i, args, rng, True, calib_seeds) for i in range(args.n_calib)]
    episodes = [(i,) + episode_params(i, args, rng, False, args.n_seeds) + (False,) for i in range(args.n_benign)]
    episodes += [(args.n_benign + i,) + episode_params(i, args, rng, True, args.n_seeds) + (True,) for i in range(args.n_harmful)]

    # ---- detectors (with calibration where needed) ----
    det_names = [d for d in args.detectors.split(",") if d]
    con_names = [c for c in args.containers.split(",") if c]
    if args.pairs:
        pairs = [tuple((p.split(":") + ["none"])[:3]) for p in args.pairs.split(",")]
    else:
        pairs = [(d, args.default_container, "none") for d in det_names] + [("eprocess", c, "none") for c in con_names if c != args.default_container]
        pairs += [("never", "none", "none"), ("immediate", args.default_container, "none")]
    seen = set()
    pairs = [p for p in pairs if not (p in seen or seen.add(p))]
    detectors = {}
    for d in {p[0] for p in pairs}:
        det = make_detector(d, args)
        if d == "logistic":
            from sklearn.linear_model import LogisticRegression
            _G["det"] = det
            t0 = time.time()
            half = args.n_calib // 2
            rows = pmap(_features_one, [c + (False,) for c in calib[:half]] + [c + (True,) for c in calib_h[:half]], args.n_jobs)
            X = np.array([f for r in rows for f, _ in r])
            y = np.array([l for r in rows for _, l in r])
            det.model = LogisticRegression(max_iter=2000).fit(X, y)
            _G["det"] = det
            stats = pmap(_calib_one, calib[half:], args.n_jobs)
            k = int(np.ceil((len(stats) + 1) * (1 - args.alpha)))
            det.threshold = float(np.sort(stats)[min(k, len(stats)) - 1])
            log.info("calibrated %s on %d benign cascades (train acc %.3f): threshold=%.4g (%.1fs)", d, len(stats), det.model.score(X, y), det.threshold, time.time() - t0)
        elif det.needs_calibration:
            _G["det"] = det
            t0 = time.time()
            stats = pmap(_calib_one, calib, args.n_jobs)
            k = int(np.ceil((len(stats) + 1) * (1 - args.alpha)))
            det.threshold = float(np.sort(stats)[min(k, len(stats)) - 1])
            log.info("calibrated %s on %d benign cascades (seeds=%d): threshold=%.4g (%.1fs)", d, len(stats), calib_seeds, det.threshold, time.time() - t0)
        detectors[d] = det

    # ---- run all pairs ----
    for d, c, th in pairs:
        det = detectors[d]
        con = get_container(c, seed=args.seed, n_samples=args.samples, pool=args.pool, horizon=args.horizon, replan_every=args.replan_every)
        thr = get_throttler(th, rho_min=args.rho_min, frac=args.throttle_frac, alpha_soft=args.alpha_soft, seed=args.seed)
        _G["det"], _G["con"], _G["thr"] = det, con, thr
        log.info("=== %s + %s + throttle:%s ===", d, c, th)
        t0 = time.time()
        rows = pmap(_run_one, episodes, args.n_jobs)
        with rl.instance_writer(f"{d}__{c}__{th}") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        ben = [r for r in rows if not r["harmful"]]
        har = [r for r in rows if r["harmful"]]
        fa = np.mean([r["alarm_round"] is not None for r in ben]) if ben else float("nan")
        det_rate = np.mean([r["alarm_round"] is not None for r in har]) if har else float("nan")
        delays = [r["alarm_round"] for r in har if r["alarm_round"] is not None]
        haa = [r["harm_at_alarm"] for r in har if r["alarm_round"] is not None]
        hwa = [r["harmw_at_alarm"] for r in har if r["alarm_round"] is not None and r.get("harmw_at_alarm") is not None]
        hf = np.array([r["harm_final"] for r in har], dtype=float)
        hcf = np.array([r["harm_counterfactual"] for r in har], dtype=float)
        bl = np.array([r["harm_counterfactual"] - r["harm_final"] for r in ben], dtype=float)
        bcf = np.array([r["harm_counterfactual"] for r in ben], dtype=float)
        kmae = [abs(r["kappa_hat"] - r["kappa"]) for r in har if r["kappa_hat"] is not None]
        row = dict(network=args.network, n_nodes=st["n_nodes"], n_edges=st["n_edges"], prob_model=args.prob_model, p=args.p, mode=args.mode,
                   alpha=args.alpha, kappa_min=args.kappa_min, kappa_max=args.kappa_max, kappa_null=args.kappa_null, p0_scale=args.p0_scale, benign_kappa_min=args.benign_kappa_min,
                   benign_kappa_max=args.benign_kappa_max, n_seeds=args.n_seeds, calib_n_seeds=calib_seeds, calib_kappa=("" if args.calib_kappa is None else args.calib_kappa), budget=args.budget,
                   detector=d, container=c, throttler=th, rho_min=args.rho_min, throttle_frac=args.throttle_frac, alpha_soft=args.alpha_soft,
                   n_benign=len(ben), n_harmful=len(har),
                   throttled_rounds_harmful=round(float(np.mean([r["throttled_rounds"] for r in har])), 3) if har else "",
                   throttled_rounds_benign=round(float(np.mean([r["throttled_rounds"] for r in ben])), 3) if ben else "",
                   throttled_nodes_benign=round(float(np.mean([r["throttled_nodes"] for r in ben])), 3) if ben else "",
                   fa_rate=round(float(fa), 4), fa_se=round(float(np.sqrt(fa * (1 - fa) / max(len(ben), 1))), 4) if ben else "",
                   det_rate=round(float(det_rate), 4), delay_mean=round(float(np.mean(delays)), 3) if delays else "",
                   harm_at_alarm=round(float(np.mean(haa)), 3) if haa else "", harmw_at_alarm=round(float(np.mean(hwa)), 3) if hwa else "", harm_final_harmful=round(float(hf.mean()), 3) if har else "",
                   harm_cf_harmful=round(float(hcf.mean()), 3) if har else "", saved_frac=round(float(1 - hf.sum() / hcf.sum()), 4) if har and hcf.sum() > 0 else "",
                   benign_loss=round(float(bl.mean()), 3) if ben else "", benign_loss_frac=round(float(bl.sum() / bcf.sum()), 4) if ben and bcf.sum() > 0 else "",
                   kappa_mae=round(float(np.mean(kmae)), 3) if kmae else "", n_intervened_mean=round(float(np.mean([r["n_intervened"] for r in har])), 2) if har else "",
                   t_detect_ms=round(1000 * float(np.mean([r["t_detect"] for r in rows])), 3), t_contain_ms=round(1000 * float(np.mean([r["t_contain"] for r in rows])), 3),
                   threshold=("" if det.threshold is None else round(float(det.threshold), 4)), seed=args.seed, notes=args.notes,
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run_id=rl.run_id, git_commit=__import__("trace_e.logging_utils", fromlist=["git_commit"]).git_commit())
        append_csv(os.path.join(args.results_dir or RESULTS_DIR, "summary_sequential.csv"), row, COLUMNS)
        with open(os.path.join(rl.run_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        log.info("RESULT %s %s+%s+%s: FA=%.3f det=%.3f delay=%s harm@alarm=%s harm_final=%s (cf %s) saved=%s benign_loss=%s kappa_mae=%s (%.0fs)",
                 args.network, d, c, th, fa, det_rate, row["delay_mean"], row["harm_at_alarm"], row["harm_final_harmful"], row["harm_cf_harmful"],
                 row["saved_frac"], row["benign_loss"], row["kappa_mae"], time.time() - t0)
    rl.finish()
    return rl.run_dir


if __name__ == "__main__":
    main()
