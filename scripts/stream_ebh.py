"""Concurrent-cascade experiment for Theorem 4: harm-weighted e-BH across a stream.

M cascades (a fraction rho harmful) run concurrently. At every round each
cascade's e-process is updated; the platform then applies (weighted) e-BH to
the current e-values of the not-yet-treated cascades and intervenes on the
rejected ones with the dominator planner. Reported: realised FDR (benign
among treated), power (harmful treated), total harmful spread, total benign
loss. Compared: per-cascade threshold 1/alpha (no multiplicity control),
unweighted e-BH, harm-weighted e-BH (weights proportional to the expected
next-round spread of the frontier under kappa_hat).

    python scripts/stream_ebh.py --network cagrqc --M 400 --rho 0.2
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
from trace_e.sequential.containers import get_container
from trace_e.sequential.detectors import get_detector
from trace_e.sequential.simulator import Episode, exposure


def ebh(evalues: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted e-BH: reject the k* largest w_c e_c with w_(k) e_(k) >= M/(k alpha)."""
    M = len(evalues)
    we = weights * evalues
    order = np.argsort(-we)
    k_star = 0
    for k in range(1, M + 1):
        if we[order[k - 1]] >= M / (k * alpha):
            k_star = k
    rej = np.zeros(M, dtype=bool)
    rej[order[:k_star]] = True
    return rej


def harm_weight(ep: Episode, kappa_hat: float) -> float:
    """Expected number of next-round activations of the frontier under kappa_hat (harm potential)."""
    g = ep.g
    inactive = ~(ep.active | ep.good | ep.blocked)
    tot = 0.0
    for u in ep.frontier:
        lo, hi = g.indptr[u], g.indptr[u + 1]
        nb = g.indices[lo:hi]
        tot += float(np.minimum(1.0, kappa_hat * g.weights[lo:hi])[inactive[nb]].sum())
    return tot


def run(args, rule: str):
    g = set_edge_probabilities(to_csr(load_network(args.network)), args.prob_model, args.p)
    elig = np.flatnonzero(g.degree() > 0)
    rng = np.random.default_rng(args.seed)
    eps, dets, harmful, kappas = [], [], [], []
    for c in range(args.M):
        h = rng.random() < args.rho
        kappa = float(np.exp(rng.uniform(np.log(args.kappa_min), np.log(args.kappa_max)))) if h else 1.0
        seeds = rng.choice(elig, size=args.n_seeds, replace=False)
        eps.append(Episode(g, kappa, seeds, np.random.default_rng(int(rng.integers(2**31)))))
        d = get_detector("eprocess", alpha=args.alpha, kappa_min=1.25, kappa_max=args.kappa_max * 1.2)
        d.reset()
        dets.append(d)
        harmful.append(h)
        kappas.append(kappa)
    harmful = np.array(harmful)
    treated = np.zeros(args.M, dtype=bool)
    treat_round = np.full(args.M, -1)
    con = [get_container("adaptive", seed=args.seed, n_samples=args.samples, horizon=args.horizon, replan_every=2) for _ in range(args.M)]
    for c in range(args.M):
        con[c].reset(args.budget)
    t0 = time.time()
    for r in range(args.max_rounds):
        alive = [c for c in range(args.M) if eps[c].alive]
        if not alive:
            break
        # containment for already-treated cascades, then advance and update evidence
        for c in alive:
            if treated[c]:
                eps[c].intervene(con[c].act(eps[c], dets[c].kappa_hat()))
        logE = np.full(args.M, -np.inf)
        weights = np.ones(args.M)
        for c in alive:
            ep, det = eps[c], dets[c]
            inactive = ~(ep.active | ep.good | ep.blocked)
            fr = ep.frontier
            new = ep.step()
            if treated[c]:
                continue
            exposed, l = exposure(g, fr, inactive, det.kappas)
            out = np.zeros(len(exposed), dtype=np.int8)
            if len(new):
                out[np.isin(exposed, new)] = 1
            det.threshold = np.inf
            det.update(l, out, 0)
        cand = np.array([c for c in range(args.M) if not treated[c]])
        if len(cand) == 0:
            continue
        ev = np.array([np.exp(min(dets[c].log_E, 700)) for c in cand])
        if rule == "per_cascade":
            rej = ev >= 1 / args.alpha
        else:
            w = np.ones(len(cand))
            if rule == "weighted_ebh":
                hw = np.array([harm_weight(eps[c], dets[c].kappa_hat()) + 1e-9 for c in cand])
                w = hw / hw.mean()
            rej = ebh(ev, w, args.alpha)
        for c, rj in zip(cand, rej):
            if rj:
                treated[c] = True
                treat_round[c] = r + 1
                eps[c].intervene(con[c].act(eps[c], dets[c].kappa_hat()))
    finals = np.array([ep.harm for ep in eps], dtype=float)
    cfs = np.array([ep.counterfactual_harm() for ep in eps], dtype=float)
    n_treated = int(treated.sum())
    fdp = float((treated & ~harmful).sum() / max(n_treated, 1))
    return {
        "rule": rule, "M": args.M, "n_harmful": int(harmful.sum()), "treated": n_treated, "false_discoveries": int((treated & ~harmful).sum()),
        "FDP": fdp, "power": float((treated & harmful).sum() / max(harmful.sum(), 1)),
        "harmful_spread": float(finals[harmful].sum()), "harmful_spread_no_intervention": float(cfs[harmful].sum()),
        "saved_frac": float(1 - finals[harmful].sum() / max(cfs[harmful].sum(), 1)),
        "benign_loss": float((cfs[~harmful] - finals[~harmful]).sum()), "mean_treat_round": float(treat_round[treated].mean()) if n_treated else None,
        "time_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="cagrqc")
    p.add_argument("--prob-model", default="wc")
    p.add_argument("--p", type=float, default=0.1)
    p.add_argument("--M", type=int, default=400)
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--kappa-min", type=float, default=1.5)
    p.add_argument("--kappa-max", type=float, default=4.0)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=60)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    out_dir = os.path.join(RESULTS_DIR, "stream")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for rep in range(args.reps):
        args.seed = args.seed + rep
        for rule in ["per_cascade", "ebh", "weighted_ebh"]:
            r = run(args, rule)
            r["rep"] = rep
            rows.append(r)
            print(json.dumps(r))
    path = os.path.join(out_dir, f"stream_{args.network}_M{args.M}_rho{args.rho}_a{args.alpha}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
