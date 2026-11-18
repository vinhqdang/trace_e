"""LaTeX tables for the empirical / SNAP-scale benchmark, generated from the per-episode logs of scripts/run_snap_benchmark.sh.

Paired quantities use the same live-edge realisation for every policy. 'sat' = 1 - sum(spread) / sum(unprotected spread).
A paired difference is called significant when |mean difference| > 2 standard errors (a screening rule, not a corrected test).
    python scripts/snap_tables.py > /tmp/snap_tables.txt
"""
from __future__ import annotations

import glob, gzip, json, os, sys
import numpy as np
import pandas as pd

TAG = "snap2"
NETS = [("cagrqc", "ca-GrQc"), ("cahepth", "ca-HepTh"), ("powergrid", "power-grid"), ("ba:2000:m=2", "BA(2000)")]
MODELS = [("const", "constant $p=0.1$"), ("wc", "weighted cascade")]
POL = [("ag", "AG"), ("gr", "GR"), ("lsbm", "LSBM"), ("proximity", "Proximity"), ("commit", "COMMIT"), ("defer_nopush", "DEFER-nopush"), ("defer", "DEFER")]


def load(tag=None):
    tag = tag or TAG
    cells = {}
    for d in sorted(glob.glob(f"results/runs/adaptive_*{tag}*")):
        cfg = json.load(open(f"{d}/config.json")); cfg = cfg.get("config", cfg)
        if cfg.get("tag") != tag: continue
        ep = pd.DataFrame([json.loads(l) for l in gzip.open(f"{d}/instances_episodes.jsonl.gz", "rt")])
        for b, g in ep.groupby("budget"):
            p = g.pivot_table(index=["inst", "draw"], columns="policy", values="spread")
            cells[(cfg["network"], cfg["prob_model"], int(b))] = p
    return cells


def sat(p, pol): return 1 - p[pol].sum() / p["none"].sum()


def diff(p, a, b):
    """mean paired saving of policy b over policy a, in percentage points of the unprotected spread (positive = b has the smaller
    spread, i.e. b is better), with its standard error and z."""
    x = p[a] - p[b]; base = p["none"].mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return 100 * x.mean() / base, 100 * se / base, (x.mean() / se if se > 0 else 0.0)


def main():
    cells = load(); out = []
    fmt = lambda m, s, z: (f"\\textbf{{{m:+.1f}}}" if abs(z) > 2 else f"{m:+.1f}") + f"\\,({s:.1f})"
    # main table: budget 20
    cols = "l" + "cc" * 4
    out.append("%% TABLE_MAIN")
    out.append("\\begin{tabular}{@{}" + "l" + "c" * 8 + "@{}}\n\\hline\n\\up & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in NETS) + " \\\\")
    out.append(" & " + " & ".join("const & WC" for _ in NETS) + " \\\\\n\\hline")
    best = {}
    for ni, (net, _) in enumerate(NETS):
        for mi, (mod, _) in enumerate(MODELS):
            p = cells[(net, mod, 20)]; best[(ni, mi)] = max(sat(p, k) for k, _ in POL)
    for k, lab in POL:
        row = []
        for ni, (net, _) in enumerate(NETS):
            for mi, (mod, _) in enumerate(MODELS):
                v = sat(cells[(net, mod, 20)], k); s = f"{v:.3f}"
                row.append(f"\\textbf{{{s}}}" if abs(v - best[(ni, mi)]) < 5e-4 else s)
        out.append((f"\\up " if k == POL[0][0] else "") + lab + " & " + " & ".join(row) + " \\\\")
    out.append("\\hline\n\\up DEFER$-$AG (pp) & " + " & ".join(fmt(*diff(cells[(net, mod, 20)], "ag", "defer")) for net, _ in NETS for mod, _ in MODELS) + " \\\\")
    out.append("\\down DEFER$-$COMMIT (pp) & " + " & ".join(fmt(*diff(cells[(net, mod, 20)], "commit", "defer")) for net, _ in NETS for mod, _ in MODELS) + " \\\\\n\\hline\n\\end{tabular}")
    # budget sweep (constant model)
    out.append("%% TABLE_SWEEP")
    out.append("\\begin{tabular}{@{}l" + "ccc" * 4 + "@{}}\n\\hline\n\\up & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in NETS) + " \\\\")
    out.append("$b$ & " + " & ".join("AG & DEFER & COMMIT" for _ in NETS) + " \\\\\n\\hline")
    for bi, b in enumerate((5, 10, 20, 50, 100)):
        row = []
        for net, _ in NETS:
            p = cells[(net, "const", b)]; v = {k: sat(p, k) for k in ("ag", "defer", "commit")}; mx = max(v.values())
            row += [(f"\\textbf{{{v[k]:.3f}}}" if abs(v[k] - mx) < 5e-4 else f"{v[k]:.3f}") for k in ("ag", "defer", "commit")]
        out.append((f"\\up " if bi == 0 else "") + f"{b} & " + " & ".join(row) + " \\\\")
    out.append("\\hline\n\\end{tabular}")
    # parity table
    out.append("%% TABLE_PARITY")
    out.append("\\begin{tabular}{@{}lcccccccc@{}}\n\\hline\n\\up & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in NETS) + " \\\\")
    out.append("$b$ & " + " & ".join("const & WC" for _ in NETS) + " \\\\\n\\hline")
    for bi, b in enumerate((5, 10, 20, 50, 100)):
        row = []
        for net, _ in NETS:
            for mod, _ in MODELS:
                p = cells[(net, mod, b)]; d = sat(p, "defer") - sat(p, "defer_nopush"); row.append(f"{d:+.3f}")
        out.append((f"\\up " if bi == 0 else "") + f"{b} & " + " & ".join(row) + " \\\\")
    out.append("\\hline\n\\end{tabular}")
    # tallies
    out.append("%% TALLY")
    for a, b, lab in (("ag", "defer", "DEFER vs AG"), ("commit", "defer", "DEFER vs COMMIT"), ("gr", "defer", "DEFER vs GR"), ("defer_nopush", "defer", "DEFER vs nopush")):
        z = [diff(p, a, b)[2] for p in cells.values()]
        out.append(f"{lab}: cells={len(z)} DEFER significantly better={sum(1 for x in z if x > 2)} worse={sum(1 for x in z if x < -2)}")
    ag_gain = [diff(cells[(net, "const", 20)], "ag", "defer")[0] for net, _ in NETS]
    out.append("const b=20 DEFER-AG pp: " + ", ".join(f"{g:+.2f}" for g in ag_gain))
    allg = [sat(p, "defer") - sat(p, "ag") for p in cells.values()]
    out.append(f"saved-fraction DEFER-AG over all {len(allg)} cells: min {min(allg):+.3f} max {max(allg):+.3f}; strictly ahead {sum(1 for g in allg if g > 5e-4)}, tied {sum(1 for g in allg if abs(g) <= 5e-4)}, behind {sum(1 for g in allg if g < -5e-4)}")
    pd_ = [abs(sat(p, "defer") - sat(p, "defer_nopush")) for p in cells.values()]
    out.append(f"max |DEFER - nopush| saved fraction: {max(pd_):.4f}; cells with diff > 0.005: {sum(1 for x in pd_ if x > 0.005)}")
    cb = [(k, sat(p, 'commit') - max(sat(p, 'defer'), sat(p, 'ag'))) for k, p in cells.items()]
    out.append("cells where COMMIT is best of AG/DEFER/COMMIT: " + str([(k, round(v, 3)) for k, v in cb if v > 5e-4]))
    # timing table (budget 20, constant probability), from the summary csv written by the same runs
    sm = pd.read_csv("results/summary_adaptive.csv"); sm = sm[sm.run_id.str.contains(TAG) & ~sm.run_id.str.contains("deg") & (sm.budget == 20) & (sm.prob_model == "const")]
    out.append("%% TABLE_TIMING")
    out.append("\\begin{tabular}{@{}lccccc@{}}\n\\hline\n\\up network ($n$) & AG & GR & DEFER & COMMIT & DEFER/AG \\\\\n\\hline")
    sizes = {"cagrqc": 5241, "cahepth": 9875, "powergrid": 4941, "ba:2000:m=2": 2000}
    for i, (net, lab) in enumerate(NETS):
        t = sm[sm.network == net].set_index("policy").time_s_per_episode
        out.append((f"\\up " if i == 0 else "") + f"{lab} ({sizes[net]}) & {t['ag']:.3f} & {t['gr']:.3f} & {t['defer']:.3f} & {t['commit']:.3f} & {t['defer'] / t['ag']:.1f}$\\times$ \\\\")
    out.append("\\hline\n\\end{tabular}")
    # high-degree-seed robustness and isolation-move ablation (separate runs)
    dc = load("snap2deg")
    if dc:
        out.append("%% TABLE_DEGREE")
        out.append("\\begin{tabular}{@{}l" + "ccc" * 4 + "@{}}\n\\hline\n\\up & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in NETS) + " \\\\")
        out.append("$b$ & " + " & ".join("AG & DEFER & COMMIT" for _ in NETS) + " \\\\\n\\hline")
        for bi, b in enumerate((10, 20, 50)):
            row = []
            for net, _ in NETS:
                p = dc[(net, "const", b)]; v = {k: sat(p, k) for k in ("ag", "defer", "commit")}; mx = max(v.values())
                row += [(f"\\textbf{{{v[k]:.3f}}}" if abs(v[k] - mx) < 5e-4 else f"{v[k]:.3f}") for k in ("ag", "defer", "commit")]
            out.append((f"\\up " if bi == 0 else "") + f"{b} & " + " & ".join(row) + " \\\\")
        out.append("\\hline\n\\end{tabular}")
        out.append("%% TABLE_ISOCUT")
        out.append("\\begin{tabular}{@{}lcccc@{}}\n\\hline\n\\up network & AG & isocut & DEFER & defer\\_cut \\\\\n\\hline")
        for i, (net, lab) in enumerate(NETS):
            p = dc[(net, "const", 20)]; v = {k: sat(p, k) for k in ("ag", "isocut", "defer", "defer_cut")}
            out.append((f"\\up " if i == 0 else "") + lab + " & " + " & ".join(f"{v[k]:.3f}" for k in ("ag", "isocut", "defer", "defer_cut")) + " \\\\")
        out.append("\\hline\n\\end{tabular}")
        out.append("%% TALLY_DEGREE")
        for a, b, lab in (("ag", "defer", "DEFER vs AG"), ("commit", "defer", "DEFER vs COMMIT"), ("ag", "isocut", "isocut vs AG"), ("defer", "defer_cut", "defer_cut vs DEFER")):
            z = [diff(p, a, b)[2] for p in dc.values()]
            out.append(f"{lab}: cells={len(z)} second significantly better={sum(1 for x in z if x > 2)} worse={sum(1 for x in z if x < -2)}")
        allg = [sat(p, "defer") - sat(p, "ag") for p in dc.values()]
        out.append(f"degree seeds DEFER-AG saved fraction over {len(allg)} cells: ahead {sum(1 for g in allg if g > 5e-4)}, tied {sum(1 for g in allg if abs(g) <= 5e-4)}, behind {sum(1 for g in allg if g < -5e-4)}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
