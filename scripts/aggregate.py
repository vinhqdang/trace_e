"""Aggregate results/summary_*.csv into results/REPORT.md (latest run per network x method)."""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

SRC_COLS = ["top1", "top3", "top5", "ed", "rr", "css", "fit_time_s", "infer_time_per_instance_s"]


def latest(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    return df.groupby(keys, as_index=False).tail(1)


def nonsingleton_metrics(run_id: str, method: str) -> dict | None:
    """Recompute top-1/top-5/ed/rr from the per-instance log, excluding outbreaks with a single infected node."""
    path = os.path.join(RES, "runs", run_id, f"instances_{method}.jsonl.gz")
    if not os.path.exists(path):
        return None
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("n_inf", 2) >= 2:
                rows.append(r)
    if not rows:
        return None
    d = pd.DataFrame(rows)
    return {"n": len(d), "top1": d.top1.mean(), "top5": d.top5.mean(), "ed": d.ed.mean(), "rr": d.rr.mean()}


def source_report(path: str) -> str:
    df = pd.read_csv(path)
    df = df[df["notes"].fillna("").str.strip() == ""] if "notes" in df else df
    out = ["## Problem A: source detection", "",
           "Protocol: continuous-time SIR, beta=1.3, sigma=0.5 (train), T=0.85, nu=1; test sigma=0. "
           "Latest run per network and method. top-k = top-k accuracy, ed = mean error distance (hops), "
           "rr = mean reciprocal rank, css = mean 90% credible-set size (probabilistic methods only).", ""]
    for net, d in latest(df, ["network", "method"]).groupby("network", sort=False):
        d = d.sort_values("top1", ascending=False)
        r0 = d.iloc[0]
        out.append(f"### {net}  (N={int(r0.n_nodes)}, E={int(r0.n_edges)}, test outbreaks={int(r0.n_instances)}, "
                   f"avg outbreak size={r0.avg_outbreak:.2f}, test sims/node={int(r0.test_sims)})")
        out.append("")
        out.append("| method | top1 | top3 | top5 | ed | rr | css | fit (s) | infer (s/inst) | top1 (n_inf>=2) | top5 (n_inf>=2) | ed (n_inf>=2) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        n_ns = None
        for _, r in d.iterrows():
            css = "" if pd.isna(r.css) else f"{r.css:.2f}"
            ns = nonsingleton_metrics(r.run_id, r.method)
            ns_cells = f"{ns['top1']:.3f} | {ns['top5']:.3f} | {ns['ed']:.3f}" if ns else " | | "
            n_ns = ns["n"] if ns else n_ns
            out.append(f"| {r.method} | {r.top1:.3f} ± {r.top1_se:.3f} | {r.top3:.3f} | {r.top5:.3f} | {r.ed:.3f} | {r.rr:.3f} | {css} | {r.fit_time_s:.1f} | {r.infer_time_per_instance_s:.5f} | {ns_cells} |")
        if n_ns:
            out.append("")
            out.append(f"The last three columns exclude outbreaks where only the source was infected at T ({n_ns} of {int(r0.n_instances)} test outbreaks remain).")
        out.append("")
    return "\n".join(out)


def blocking_report(path: str) -> str:
    df = pd.read_csv(path)
    df["mode"] = df["notes"].str.extract(r"mode=(\w+)")
    df["prob"] = df["notes"].str.extract(r"prob=(\w+)")
    df["budget"] = df["T"].astype(int)
    out = ["## Problem B: misinformation blocking", "",
           "Independent cascade with weighted-cascade probabilities unless noted. saved = 1 - (bad adopters after intervention / without). "
           "Latest run per network, mode, method and budget; mean ± s.e. over seed-set instances.", ""]
    for (net, mode), d in latest(df, ["network", "mode", "prob", "method", "budget"]).groupby(["network", "mode"], sort=False):
        r0 = d.iloc[0]
        budgets = sorted(d.budget.unique())
        out.append(f"### {net} / {mode} mode  (N={int(r0.n_nodes)}, E={int(r0.n_edges)}, prob={r0.prob}, "
                   f"instances={int(r0.n_instances)}, no-intervention spread={r0.avg_outbreak:.2f} nodes)")
        out.append("")
        out.append("| method | " + " | ".join(f"saved@{b}" for b in budgets) + " | select time (s) |")
        out.append("|---|" + "---|" * (len(budgets) + 1))
        piv = d.pivot_table(index="method", columns="budget", values=["top1", "top1_se", "infer_time_per_instance_s"], aggfunc="first")
        order = piv["top1"][budgets[-1]].sort_values(ascending=False).index
        for m in order:
            cells = [f"{piv['top1'][b][m]:.3f} ± {piv['top1_se'][b][m]:.3f}" for b in budgets]
            out.append(f"| {m} | " + " | ".join(cells) + f" | {piv['infer_time_per_instance_s'][budgets[-1]][m]:.3f} |")
        out.append("")
    return "\n".join(out)


def sequential_report(path: str) -> str:
    df = pd.read_csv(path)
    df["setting"] = df["notes"].fillna("main")
    out = ["## Problem C: sequential detect-and-contain (AVID vs baselines)", "",
           "Independent cascade streams, half benign (kappa=1 unless noted) and half harmful (kappa log-uniform in [kappa_min, kappa_max]). "
           "FA = fraction of benign cascades acted on (target <= alpha); det = fraction of harmful cascades acted on; delay in rounds; "
           "harm@alarm = bad nodes when the detector fires; saved = 1 - final harmful spread / no-intervention spread; "
           "benign loss = activations suppressed on benign cascades. Latest run per configuration.", ""]
    keys = ["network", "setting", "alpha", "budget", "n_seeds", "calib_n_seeds", "benign_kappa_min", "detector", "container"]
    for (net, setting, alpha, budget, ns, cns, bkm), d in latest(df, keys).groupby(["network", "setting", "alpha", "budget", "n_seeds", "calib_n_seeds", "benign_kappa_min"], sort=False):
        r0 = d.iloc[0]
        out.append(f"### {net} / {setting}  (alpha={alpha}, budget={budget}, seeds={ns}, calib seeds={cns}, benign kappa >= {bkm}, "
                   f"harmful kappa in [{r0.kappa_min}, {r0.kappa_max}], prob={r0.prob_model}, {int(r0.n_benign)} benign / {int(r0.n_harmful)} harmful)")
        out.append("")
        out.append("| detector | container | FA | det | delay | harm@alarm | final harm | saved | benign loss | kappa MAE | detect ms | contain ms |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        d = d.sort_values(["saved_frac"], ascending=False)
        for _, r in d.iterrows():
            f = lambda x, k=3: "" if pd.isna(x) or x == "" else f"{float(x):.{k}f}"  # noqa: E731
            out.append(f"| {r.detector} | {r.container} | {f(r.fa_rate)} ± {f(r.fa_se)} | {f(r.det_rate)} | {f(r.delay_mean, 2)} | {f(r.harm_at_alarm, 1)} | "
                       f"{f(r.harm_final_harmful, 1)} | {f(r.saved_frac)} | {f(r.benign_loss, 2)} | {f(r.kappa_mae, 2)} | {f(r.t_detect_ms, 2)} | {f(r.t_contain_ms, 1)} |")
        out.append("")
    return "\n".join(out)


def stream_report() -> str:
    files = sorted(glob.glob(os.path.join(RES, "stream", "*.json")))
    if not files:
        return ""
    out = ["## Concurrent cascades: FDR control and harm-weighted e-BH (Theorem 4)", "",
           "M cascades run concurrently (fraction rho harmful). At every round the platform selects cascades to act on from the current "
           "e-values: per-cascade threshold 1/alpha (no multiplicity control), e-BH, and harm-weighted e-BH (weights proportional to the "
           "frontier's expected next-round spread). FDP = benign among treated; power = harmful treated; saved = harmful spread removed; "
           "mean ± s.d. over repetitions.", ""]
    for fp in files:
        d = json.load(open(fp))
        a = d["args"]
        out.append(f"### {a['network']} (M={a['M']}, rho={a['rho']}, alpha={a['alpha']}, seeds={a['n_seeds']}, budget={a['budget']}, kappa in [{a['kappa_min']}, {a['kappa_max']}])")
        out.append("")
        out.append("| rule | FDP | power | saved | benign loss | treated |")
        out.append("|---|---|---|---|---|---|")
        df = pd.DataFrame(d["rows"])
        for rule, g in df.groupby("rule", sort=False):
            f = lambda c: f"{g[c].mean():.3f} ± {g[c].std(ddof=0):.3f}"  # noqa: E731
            out.append(f"| {rule} | {f('FDP')} | {f('power')} | {f('saved_frac')} | {g['benign_loss'].mean():.1f} | {g['treated'].mean():.1f} |")
        out.append("")
    return "\n".join(out)


def theory_report() -> str:
    files = sorted(glob.glob(os.path.join(RES, "theory", "*.json")))
    if not files:
        return ""
    out = ["## Theory checks (Theorems 1 and 2)", ""]
    for fp in files:
        r = json.load(open(fp))
        out.append(f"### {r['network']} ({r['prob_model']}, {r['n_seeds']} seeds, {r['n']} benign cascades per null, grid size {r['grid_size']})")
        out.append("")
        out.append("False-alarm rate of the e-process (must be <= alpha):")
        out.append("")
        fa_keys = [k for k in r if k.startswith("fa[")]
        alphas = list(r[fa_keys[0]].keys())
        out.append("| null | " + " | ".join(f"alpha={a}" for a in alphas) + " |")
        out.append("|---|" + "---|" * len(alphas))
        for k in fa_keys:
            out.append(f"| {k[3:-1]} | " + " | ".join(f"{r[k][a]:.4f}" for a in alphas) + " |")
        out.append("")
        out.append("Harm at alarm (bad nodes beyond the seeds when the e-process crosses 1/alpha, alpha=0.05) against the Theorem 2 bounds:")
        out.append("")
        out.append("| kappa | detected | mean harm@alarm | median | lower bound (any valid rule) | upper bound (no overshoot) | final spread w/o intervention |")
        out.append("|---|---|---|---|---|---|---|")
        for k, v in r["harm_at_alarm"].items():
            m = "" if v["mean_harm_at_alarm"] is None else f"{v['mean_harm_at_alarm']:.1f}"
            md = "" if v["median_harm_at_alarm"] is None else f"{v['median_harm_at_alarm']:.0f}"
            out.append(f"| {k} | {v['detected_frac']:.3f} | {m} | {md} | {v['lower_bound']:.1f} | {v['upper_bound_no_overshoot']:.1f} | {v['mean_final_no_intervention']:.1f} |")
        out.append("")
    return "\n".join(out)


def main():
    parts = ["# Results report", "", "Generated by `python scripts/aggregate.py` from `results/summary_*.csv`.", ""]
    s = os.path.join(RES, "summary_source.csv")
    b = os.path.join(RES, "summary_blocking.csv")
    if os.path.exists(s):
        parts.append(source_report(s))
    if os.path.exists(b):
        parts.append(blocking_report(b))
    q = os.path.join(RES, "summary_sequential.csv")
    if os.path.exists(q):
        parts.append(sequential_report(q))
    parts.append(stream_report())
    parts.append(theory_report())
    with open(os.path.join(RES, "REPORT.md"), "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", os.path.join(RES, "REPORT.md"))


if __name__ == "__main__":
    sys.exit(main())
