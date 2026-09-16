"""Empirical check of Theorem 5: throttling leaves the number of activations before a valid
decision essentially unchanged but lowers their influence-weighted sum.

For fixed kappa, run the e-process (no containment) with throttling always on
(alpha_soft = 1) under none / hub / uniform / random policies and record, at
the first crossing of 1/alpha: the activation count H_tau, the influence-
weighted harm W_tau = sum over activated non-seed nodes of (1 + kappa * expected children), and the
number of rounds. Also records the benign cost of each policy.

    python scripts/verify_throttle.py --network cagrqc --n 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from trace_e.blocking.cascade import set_edge_probabilities
from trace_e.graphs import load_network, to_csr
from trace_e.logging_utils import RESULTS_DIR
from trace_e.sequential.containers import get_container
from trace_e.sequential.detectors import get_detector
from trace_e.sequential.policy import run_episode
from trace_e.sequential.simulator import Episode
from trace_e.sequential.throttlers import get_throttler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="cagrqc")
    p.add_argument("--prob-model", default="wc")
    p.add_argument("--p", type=float, default=0.1)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--kappas", default="2,3,4")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rho-min", type=float, default=0.1)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    g = set_edge_probabilities(to_csr(load_network(args.network)), args.prob_model, args.p)
    elig = np.flatnonzero(g.degree() > 0)
    det = get_detector("eprocess", alpha=args.alpha, kappa_min=1.25, kappa_max=8.0)
    con = get_container("none")
    policies = {"none": get_throttler("none"), "hub": get_throttler("hub", rho_min=args.rho_min, frac=args.frac, alpha_soft=1.0),
                "uniform": get_throttler("uniform", rho_min=args.rho_min, alpha_soft=1.0),
                "random": get_throttler("random", rho_min=args.rho_min, frac=args.frac, alpha_soft=1.0, seed=args.seed)}
    report = {"args": vars(args), "rows": []}
    for kappa in [float(k) for k in args.kappas.split(",")] + [1.0]:
        rng = np.random.default_rng(args.seed)
        params = [(rng.choice(elig, size=args.n_seeds, replace=False), int(rng.integers(2**31))) for _ in range(args.n)]
        for name, thr in policies.items():
            H, W, R, det_n, benign_loss = [], [], [], 0, []
            for seeds, es in params:
                ep = Episode(g, kappa, seeds, np.random.default_rng(es))
                r = run_episode(ep, det, con, budget=0, max_rounds=80, throttler=thr)
                if kappa == 1.0:
                    benign_loss.append(r["harm_counterfactual"] - r["harm_final"])
                    if r["alarm_round"] is not None:
                        det_n += 1
                    continue
                if r["alarm_round"] is not None:
                    det_n += 1
                    H.append(r["harm_at_alarm"] - args.n_seeds)
                    W.append(r["harmw_at_alarm"])
                    R.append(r["alarm_round"])
            row = {"kappa": kappa, "policy": name, "detected": det_n / args.n,
                   "H_at_alarm": float(np.mean(H)) if H else None, "W_at_alarm": float(np.mean(W)) if W else None,
                   "rounds": float(np.mean(R)) if R else None, "benign_loss": float(np.mean(benign_loss)) if benign_loss else None}
            report["rows"].append(row)
            print(json.dumps(row))
    out = os.path.join(RESULTS_DIR, "theory", f"throttle_{args.network}_{args.prob_model}_rho{args.rho_min}_f{args.frac}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
