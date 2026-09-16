"""Pareto view of the throttling sweep: final harmful spread vs benign cost per throttler setting.

    python scripts/pareto_throttle.py [--network cagrqc]

Writes results/figures/throttle_pareto_<network>.png and prints a markdown table.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from trace_e.logging_utils import RESULTS_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="cagrqc")
    p.add_argument("--setting", default="throttle_sweep")
    p.add_argument("--prob-model", default="wc")
    args = p.parse_args()
    df = pd.read_csv(os.path.join(RESULTS_DIR, "summary_sequential.csv"))
    df = df[(df.network == args.network) & (df.notes.fillna("") == args.setting) & (df.prob_model == args.prob_model)]
    df["throttler"] = df["throttler"].fillna("none")
    df = df.sort_values("timestamp").groupby(["throttler", "rho_min", "throttle_frac", "container"], as_index=False).tail(1)
    if df.empty:
        print("no rows")
        return
    df["label"] = df.apply(lambda r: r.throttler if r.throttler == "none" else f"{r.throttler} rho={r.rho_min:g} f={r.throttle_frac:g}", axis=1)
    cols = ["label", "fa_rate", "det_rate", "harm_at_alarm", "harmw_at_alarm", "harm_final_harmful", "saved_frac", "benign_loss", "throttled_rounds_benign"]
    print(df[cols].sort_values("harm_final_harmful").to_markdown(index=False, floatfmt=".3f"))
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"none": "black", "hub": "#1b6ca8", "uniform": "#c2410c", "random": "#6b7280"}
    for thr, g in df.groupby("throttler"):
        g = g.sort_values("benign_loss")
        ax.plot(g.benign_loss, g.harm_final_harmful, "o-" if thr != "none" else "s", color=colors.get(thr, "gray"), label=thr, ms=6)
        for _, r in g.iterrows():
            if thr != "none":
                ax.annotate(f"{r.rho_min:g}/{r.throttle_frac:g}" if thr != "uniform" else f"{r.rho_min:g}", (r.benign_loss, r.harm_final_harmful), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("benign cost: activations suppressed per benign cascade")
    ax.set_ylabel("final harmful spread (mean nodes)")
    ax.set_title(f"{args.network} ({args.prob_model}): active throttling, e-process + adaptive containment")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", f"throttle_pareto_{args.network}_{args.prob_model}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
