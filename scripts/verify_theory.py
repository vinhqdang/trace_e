"""Empirical checks of Theorems 1 and 2 of docs/METHOD.md.

1. False-alarm rate of the e-process on benign cascades for a grid of alpha
   (must stay at or below alpha), including benign cascades below baseline
   (dominance) and an inflated-null margin.
2. Harm at alarm on harmful cascades as a function of kappa, against the
   lower bound (any valid rule) and the upper bound of Theorem 2.

    python scripts/verify_theory.py --network cagrqc --prob-model wc --n 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from trace_e.blocking.cascade import set_edge_probabilities
from trace_e.graphs import load_network, to_csr
from trace_e.logging_utils import RESULTS_DIR
from trace_e.sequential.detectors import get_detector
from trace_e.sequential.simulator import Episode, exposure


def run_until_alarm(g, det, kappa, seeds, ep_seed, max_rounds=80):
    ep = Episode(g, kappa, seeds, np.random.default_rng(ep_seed))
    det.reset()
    rounds = 0
    first_cross = {}
    while ep.alive and rounds < max_rounds:
        inactive = ~(ep.active | ep.good | ep.blocked)
        fr = ep.frontier
        size_before = ep.harm
        new = ep.step()
        rounds += 1
        exposed, l = exposure(g, fr, inactive, det.kappas)
        out = np.zeros(len(exposed), dtype=np.int8)
        if len(new):
            out[np.isin(exposed, new)] = 1
        det.threshold = np.inf  # never stop: record the whole path
        det.update(l, out, size_before)
        for a in ALPHAS:
            if a not in first_cross and det.log_E >= np.log(1 / a):
                first_cross[a] = (rounds, ep.harm)
    return first_cross, ep.harm, rounds


ALPHAS = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="cagrqc")
    p.add_argument("--prob-model", default="wc")
    p.add_argument("--p", type=float, default=0.1)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--kappas", default="1.5,2,3,4,6")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    g = set_edge_probabilities(to_csr(load_network(args.network)), args.prob_model, args.p)
    elig = np.flatnonzero(g.degree() > 0)
    rng = np.random.default_rng(args.seed)
    kmin, kmax = 1.25, 8.0
    det = get_detector("eprocess", alpha=0.05, kappa_min=kmin, kappa_max=kmax, eta=0.25)
    J = len(det.grid)
    out_dir = os.path.join(RESULTS_DIR, "theory")
    os.makedirs(out_dir, exist_ok=True)
    report = {"network": args.network, "prob_model": args.prob_model, "n": args.n, "n_seeds": args.n_seeds, "grid_size": J}

    # ---- Theorem 1: false alarms under the null and under dominance ----
    t0 = time.time()
    for label, bk in [("kappa=1", (1.0, 1.0)), ("kappa~U[0.5,1]", (0.5, 1.0))]:
        counts = {a: 0 for a in ALPHAS}
        for i in range(args.n):
            kappa = float(rng.uniform(*bk))
            seeds = rng.choice(elig, size=args.n_seeds, replace=False)
            fc, _, _ = run_until_alarm(g, det, kappa, seeds, int(rng.integers(2**31)))
            for a in fc:
                counts[a] += 1
        report[f"fa[{label}]"] = {str(a): counts[a] / args.n for a in ALPHAS}
        print(label, "false-alarm rate:", {a: round(counts[a] / args.n, 4) for a in ALPHAS}, f"({time.time() - t0:.0f}s)")

    # ---- Theorem 2: harm at alarm vs kappa ----
    alpha = 0.05
    res = {}
    for kappa in [float(k) for k in args.kappas.split(",")]:
        harms, delays, detected, cf = [], [], 0, []
        for i in range(args.n // 4):
            seeds = rng.choice(elig, size=args.n_seeds, replace=False)
            fc, final, rounds = run_until_alarm(g, det, kappa, seeds, int(rng.integers(2**31)))
            cf.append(final)
            if alpha in fc:
                detected += 1
                delays.append(fc[alpha][0])
                harms.append(fc[alpha][1] - args.n_seeds)
        kj = det.grid[det.grid <= kappa].max() if (det.grid <= kappa).any() else det.grid[0]
        c = (kj * np.log(kj) - kj + 1) / kj
        lower = (np.log(1 / alpha) - np.log(2)) / np.log(kappa)  # beta -> 0 form
        upper_no_overshoot = (np.log(1 / alpha) + np.log(J)) / c
        res[str(kappa)] = {"detected_frac": detected / (args.n // 4), "mean_harm_at_alarm": float(np.mean(harms)) if harms else None,
                           "median_harm_at_alarm": float(np.median(harms)) if harms else None, "mean_delay_rounds": float(np.mean(delays)) if delays else None,
                           "mean_final_no_intervention": float(np.mean(cf)), "lower_bound": float(lower), "upper_bound_no_overshoot": float(upper_no_overshoot),
                           "kappa_grid_point": float(kj)}
        print(f"kappa={kappa}: detected {detected / (args.n // 4):.3f}, harm@alarm mean {np.mean(harms) if harms else float('nan'):.1f} "
              f"(median {np.median(harms) if harms else float('nan'):.0f}), lower bound {lower:.1f}, upper bound (no overshoot) {upper_no_overshoot:.1f}, "
              f"final spread w/o intervention {np.mean(cf):.1f}")
    report["harm_at_alarm"] = res
    path = os.path.join(out_dir, f"theory_{args.network}_{args.prob_model}_s{args.n_seeds}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
