"""Figures for the DEFER and TWIG papers, generated from the repository's result files.

    python scripts/paper_figures.py            # writes paper/defer/figures/*.pdf and paper/twig/figures/*.pdf
"""
from __future__ import annotations

import glob, gzip, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import snap_tables as st

BLUE, ORANGE, AQUA, MAGENTA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#8d8c86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42, "axes.titlesize": 8.5, "axes.titleweight": "regular"})
DEFER_DIR, TWIG_DIR = "paper/defer/figures", "paper/twig/figures"
os.makedirs(DEFER_DIR, exist_ok=True); os.makedirs(TWIG_DIR, exist_ok=True)


def save(fig, d, name):
    fig.savefig(os.path.join(d, name)); plt.close(fig); print("saved", d, name)


# ------------------------------------------------------------------ DEFER paper
def fig_adaptivity_gap():
    rows = json.load(open("results/theory/adaptivity_gap.json")); L = 30
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    cols = {0.02: BLUE, 0.05: AQUA, 0.1: ORANGE, 0.2: VIOLET}
    ds = np.array([2 ** k for k in np.linspace(1, 5.3, 80)])
    for p, c in cols.items():
        ax.plot(ds, L / (L + 1) * (1 - (1 - p) ** ds) / p, color=c, lw=1.4)
        r = [x for x in rows if abs(x["p"] - p) < 1e-9]
        ax.plot([x["Delta"] for x in r], [x["measured_ratio"] for x in r], "o", color=c, ms=4.5, mec="#fcfcfb", mew=0.7)
        ax.text(ds[-1] * 1.05, L / (L + 1) * (1 - (1 - p) ** ds[-1]) / p, f"$p={p:g}$", color=INK2, va="center", fontsize=7.5)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels(["4", "8", "16", "32"]); ax.minorticks_off()
    ax.set_xlim(2, 52); ax.set_ylim(0, 30)
    ax.set_xlabel("branches $\\Delta$"); ax.set_ylabel("saving of DEFER / saving of one-shot")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Lines: theory ($L=30$). Points: measured (400 runs).", loc="left", fontsize=7.5, color=INK2)
    fig.tight_layout(); save(fig, DEFER_DIR, "fig_adaptivity_gap.pdf")


def fig_sop():
    d = pd.DataFrame([json.loads(l) for l in open("results/theory/star_of_paths_bench.jsonl")])
    series = [("ag", "AG", BLUE, "o"), ("commit", "COMMIT", ORANGE, "s"), ("ilp1", "ILP-1", AQUA, "^"), ("ilp_roll", "ILP-S", MAGENTA, "v"), ("defer", "DEFER", VIOLET, "D")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, D in zip(axes, (4, 8, 16)):
        cfg = [(p, b) for p in (0.05, 0.1, 0.2) for b in (1, 2)]
        for key, name, col, mk in series:
            y = []
            for p, b in cfg:
                g = d[(d.Delta == D) & (d.p == p) & (d.b == b)]
                y.append(1 - g[key].sum() / g["none"].sum())
            ax.plot(range(len(cfg)), y, color=col, marker=mk, ms=5.5 if key != "defer" else 7, lw=0, label=name, mec="#fcfcfb", mew=0.6, zorder=3 if key == "defer" else 2)
        ax.set_xticks(range(len(cfg))); ax.set_xticklabels([f"{p:g}\n{b}" for p, b in cfg], fontsize=7)
        ax.set_title(f"$\\Delta={D}$ branches", loc="left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("$p$ (top) and budget $b$ (bottom)", fontsize=7)
    axes[0].set_ylabel("fraction of spread saved"); axes[0].set_ylim(0, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=5, handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.09, 1, 1)); save(fig, DEFER_DIR, "fig_sop.pdf")


def regime_cells():
    cells = []
    for dd in sorted(glob.glob("results/runs/adaptive_*regime*")):
        cfg = json.load(open(f"{dd}/config.json")); cfg = cfg.get("config", cfg)
        ep = pd.DataFrame([json.loads(l) for l in gzip.open(f"{dd}/instances_episodes.jsonl.gz", "rt")])
        for b, g in ep.groupby("budget"):
            p = g.pivot_table(index=["inst", "draw"], columns="policy", values="spread")
            x = p["ag"] - p["defer"]; se = x.std(ddof=1) / np.sqrt(len(x)); n0 = p["none"].mean()
            cells.append(dict(fam=cfg["network"].split(":")[0], p=cfg["p"], b=int(b), n0=n0, gain=100 * x.mean() / n0, z=(x.mean() / se if se > 0 else 0.0)))
    return pd.DataFrame(cells)


def fig_regime():
    r = regime_cells(); fams = ["tree", "grid", "rgg", "ws", "ba"]; names = {"tree": "random tree", "grid": "grid", "rgg": "geometric", "ws": "Watts-Strogatz", "ba": "Barabasi-Albert"}
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.0))
    vmax = max(1.0, r.gain.max())
    for ax, f in zip(axes, fams):
        g = r[r.fam == f]; ps = sorted(g.p.unique()); bs = sorted(g.b.unique())
        M = np.array([[g[(g.p == p) & (g.b == b)].gain.iloc[0] for b in bs] for p in ps]); Z = np.array([[g[(g.p == p) & (g.b == b)].z.iloc[0] for b in bs] for p in ps])
        ax.imshow(np.clip(M, 0, None), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("v", ["#f4f2fb", VIOLET]), vmin=0, vmax=vmax, aspect="auto")
        for i in range(len(ps)):
            for j in range(len(bs)):
                sig = Z[i, j] > 2
                ax.text(j, i, f"{M[i, j]:.1f}" + ("*" if sig else ""), ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.55 * vmax else INK, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(len(bs))); ax.set_xticklabels(bs); ax.set_yticks(range(len(ps))); ax.set_yticklabels([f"{p:g}" for p in ps])
        ax.set_title(names[f], loc="left", fontsize=7.5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("budget $b$", fontsize=7)
    axes[0].set_ylabel("edge probability $p$", fontsize=7)
    fig.tight_layout(w_pad=0.6); save(fig, DEFER_DIR, "fig_regime_heatmap.pdf")


def fig_zone():
    r = regime_cells(); r["src"] = "synthetic scan"
    sn = []
    for (net, mod, b), p in st.load().items():
        x = p["ag"] - p["defer"]; se = x.std(ddof=1) / np.sqrt(len(x)); n0 = p["none"].mean()
        sn.append(dict(fam="empirical", p=0, b=b, n0=n0, gain=100 * x.mean() / n0, z=(x.mean() / se if se > 0 else 0.0), src="empirical networks"))
    a = pd.concat([r, pd.DataFrame(sn)], ignore_index=True); a["ratio"] = a.n0 / a.b
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for src, col, mk in (("synthetic scan", VIOLET, "o"), ("empirical networks", ORANGE, "s")):
        g = a[a.src == src]
        for sig, fill in ((False, False), (True, True)):
            h = g[(g.z > 2) == sig]
            ax.scatter(h.ratio, h.gain, s=26, marker=mk, facecolor=col if fill else "none", edgecolor=col, linewidth=1.0, label=f"{src}" + (", significant" if sig else ", not significant") if True else None, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xscale("log"); ax.set_xlabel("unprotected spread per unit of budget, $n_0/b$"); ax.set_ylabel("DEFER $-$ AG (points of spread saved)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); save(fig, DEFER_DIR, "fig_regime_scatter.pdf")


def fig_ilp():
    d = pd.read_csv("results/summary_adaptive.csv"); d = d[d.run_id.str.contains("adaptive_ilp_")]
    p = d.pivot_table(index=["network", "p", "budget"], columns="policy", values="saved_frac"); t = d.pivot_table(index=["network", "p", "budget"], columns="policy", values="time_s_per_episode")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (a, b, la, lb, col) in zip(axes[:2], (("ag", "ilp1", "AG", "ILP-1", AQUA), ("defer", "ilp_roll", "DEFER", "ILP-S", MAGENTA))):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, ls="--"); ax.scatter(p[a], p[b], s=22, color=col, edgecolor="#fcfcfb", linewidth=0.5, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xlabel(f"saved fraction, {la}"); ax.set_ylabel(f"saved fraction, {lb}")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axes[2]
    for i, (k, lab, col) in enumerate((("ag", "AG", BLUE), ("defer", "DEFER", VIOLET), ("ilp1", "ILP-1", AQUA), ("ilp_roll", "ILP-S", MAGENTA))):
        v = t[k].values; ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), v, s=16, color=col, edgecolor="#fcfcfb", linewidth=0.4, zorder=3)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(["AG", "DEFER", "ILP-1", "ILP-S"], fontsize=7); ax.set_ylabel("seconds per episode"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.8); save(fig, DEFER_DIR, "fig_ilp.pdf")


# ------------------------------------------------------------------ TWIG paper
def frontier_profile(G, order):
    pos = {v: i for i, v in enumerate(order)}; last = {v: max([pos[v]] + [pos[u] for u in G.neighbors(v)]) for v in order}
    return [sum(1 for v in order[: i + 1] if last[v] >= i) for i in range(len(order))]


def fig_frontier():
    from trace_e.blocking.treewidth import elimination_order
    from trace_e.graphs import load_empirical
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, (nm, lab) in zip(axes, (("karate", "karate (34 vertices)"), ("iceland", "Iceland (75)"), ("dolphin", "dolphin (62)"))):
        G = nx.convert_node_labels_to_integers(load_empirical(nm)); o = elimination_order(G)
        fw, rv = frontier_profile(G, o), frontier_profile(G, list(reversed(o)))
        ax.plot(range(1, len(o) + 1), fw, color=ORANGE, lw=1.6, label=f"forward (width {max(fw) - 1})"); ax.plot(range(1, len(o) + 1), rv, color=VIOLET, lw=1.8, label=f"reversed (width {max(rv) - 1})")
        ax.set_title(lab, loc="left"); ax.set_xlabel("step $i$"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("frontier size $|F_i|$"); fig.tight_layout(w_pad=0.8); save(fig, TWIG_DIR, "fig_frontier_profiles.pdf")


def bell(n):
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row: nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def fig_bell():
    fe = json.load(open("results/theory/twig_feasibility.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]; ws = np.arange(1, 23); ax.plot(ws, [bell(w + 2) for w in ws], color=GREY, lw=1.4); ax.set_yscale("log")
    for nm, lab, col in (("karate", "karate: 2.5 s", BLUE), ("iceland", "Iceland: 55 s", AQUA), ("dolphin", "dolphin: memory limit", ORANGE)):
        w = fe["networks"][nm]["width"]; ax.plot([w], [bell(w + 2)], "o", color=col, ms=7, mec="#fcfcfb", mew=0.8); ax.annotate(lab, (w, bell(w + 2)), xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=7.5, color=INK)
    ax.set_xlabel("width $w$"); ax.set_ylabel("state bound $B_{w+2}$"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axes[1]; bt = fe["binary_trees"]
    ax.plot([x["height"] for x in bt], [x["realised_width"] for x in bt], "o-", color=VIOLET, lw=1.6, ms=5, label="realised width (reversed min-fill-in)")
    ax.axhline(1, color=BLUE, lw=1.4); ax.text(7.1, 1.15, "treewidth $=1$", ha="right", fontsize=7.5, color=BLUE)
    ax.set_xlabel("height $h$ of complete binary tree"); ax.set_ylabel("width"); ax.set_ylim(0, 7); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=1.0); save(fig, TWIG_DIR, "fig_bell_states.pdf")


def rebuild_instance(fam, idx):
    import twig_optimality_gap as tg
    rng = random.Random(tg.SEED + len(fam))
    for i in range(idx + 1):
        out = tg.instance(rng, fam)
    return out


def fig_gap():
    d = pd.read_csv("results/theory/twig_optimality_gap.csv")
    fig = plt.figure(figsize=(7.2, 3.0)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0]); fams = ["sparse", "medium", "dense"]; rng = np.random.default_rng(1)
    for i, f in enumerate(fams):
        g = d[d.family == f]; y = np.where(g.excess < 1e-9, 5e-4, g.excess * 100)  # exact optimum plotted on a floor line
        ok = g.excess < 1e-9
        ax.scatter(i + rng.uniform(-0.25, 0.25, ok.sum()), np.full(ok.sum(), 0.04), s=8, color=GREY, alpha=0.7, zorder=2)
        ax.scatter(i + rng.uniform(-0.2, 0.2, (~ok).sum()), g.excess[~ok] * 100, s=26, color=ORANGE, edgecolor="#fcfcfb", linewidth=0.5, zorder=3)
        ax.text(i, 0.022, f"{int(ok.sum())}/{len(g)} optimal", ha="center", fontsize=7.2, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(0.015, 150); ax.set_yticks([0.04, 0.1, 1, 10, 100]); ax.set_yticklabels(["optimal", "0.1%", "1%", "10%", "100%"])
    ax.set_xticks(range(3)); ax.set_xticklabels(fams); ax.set_ylabel("excess spread of greedy"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("(a) 300 instances; grey = greedy optimal", loc="left", fontsize=8)
    # (b) largest-excess instance
    j = d.excess.idxmax(); row = d.loc[j]; G, wfn, seeds, b, order, w = rebuild_instance(row.family, int(row.i))
    ax = fig.add_subplot(gs[1]); pos = nx.spring_layout(G, seed=3, k=1.2)
    ge, oe = eval(row.greedy_set), eval(row.opt_set)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=INK2, width=[0.4 + 2.2 * wfn(u, v) for u, v in G.edges], alpha=0.65)
    for v in G.nodes:
        kind = "seed" if v in seeds else "plain"
        ax.scatter(*pos[v], s=250 if v in seeds else 170, marker="s" if v in seeds else "o", facecolor="#f4f2fb" if v in seeds else "white", edgecolor=INK, linewidth=1.0, zorder=3)
        ax.text(*pos[v], str(v), ha="center", va="center", fontsize=6.5, zorder=5)
    for v in ge: ax.scatter(*pos[v], s=340, marker="o", facecolor="none", edgecolor=ORANGE, linewidth=2.2, zorder=4)
    for v in oe: ax.scatter(*pos[v], s=470, marker="o", facecolor="none", edgecolor=VIOLET, linewidth=2.2, ls="--", zorder=4)
    ax.axis("off"); ax.set_title(f"(b) largest gap: greedy {row.spread_greedy:.3f} vs optimum {row.spread_opt:.3f}", loc="left", fontsize=8)
    ax.text(0.0, -0.02, "square: seed; orange ring: greedy set; violet dashed ring: optimal set; edge width: probability", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
    fig.tight_layout(); save(fig, TWIG_DIR, "fig_gap.pdf")


def fig_orders():
    d = json.load(open("results/theory/twig_scaling.json"))["orders"]
    d = [x for x in d if x["graph"] not in ("grid 3x200",)]
    d.sort(key=lambda x: x["width_min_fill_reversed"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(d))
    for i, x in enumerate(d):
        ax.plot([x["width_best_layout"], x["width_min_fill_reversed"]], [i, i], color=GRID, lw=2.0, zorder=1)
    ax.scatter([x["width_min_fill_reversed"] for x in d], y, s=34, color=ORANGE, zorder=3, label="reversed min-fill-in order")
    ax.scatter([x["width_best_layout"] for x in d], y, s=34, color=VIOLET, marker="D", zorder=3, label="best greedy layout")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([f"{x['graph']} ($n={x['n']}$)" for x in d], fontsize=7)
    ax.set_xlabel("frontier width $w$ of the order (log scale)"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.tight_layout(); save(fig, TWIG_DIR, "fig_orders.pdf")


def fig_scaling():
    d = json.load(open("results/theory/twig_scaling.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    for fam, col, mk in (("grid 3 x m", BLUE, "o"), ("random tree", ORANGE, "s")):
        r = [x for x in d["n_scaling"] if x["family"] == fam and x["outcome"] == "ok"]
        ax.plot([x["n"] for x in r], [x["seconds"] for x in r], marker=mk, color=col, lw=1.4, ms=5, label=("grid $3\\times m$ (width 3)" if fam.startswith("grid") else "random trees (width 3-10)"))
    ns = np.array([100, 5000]); ax.plot(ns, 2.2e-4 * ns, color=GREY, ls="--", lw=0.9); ax.text(1200, 0.05, "slope 1", color=INK2, fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("vertices $n$"); ax.set_ylabel("seconds per evaluation"); ax.legend(frameon=False, fontsize=6.8, loc="upper left"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axes[1]
    r = [x for x in d["w_scaling"] if x["outcome"] == "ok"]
    ax.plot([x["width"] for x in r], [x["seconds"] for x in r], "o-", color=VIOLET, lw=1.6, ms=5, label="grid strips, $n=240$")
    pb = [x for x in d["powergrid_balls"] if x["outcome"] == "ok"]
    ax.scatter([x["width"] for x in pb], [x["seconds"] for x in pb], s=22, color=AQUA, marker="^", label="power-grid subnetworks", zorder=3)
    ws = np.arange(2, 13); ax.plot(ws, [r[0]["seconds"] * 4.0 ** (w - 2) for w in ws], color=GREY, ls="--", lw=0.9); ax.text(9.1, 0.01, "$\\times4$ per unit", color=INK2, fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("frontier width $w$"); ax.legend(frameon=False, fontsize=6.8, loc="upper left"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axes[2]
    fe = d["feasible"]
    names = ["karate", "iceland", "dolphin"]
    old = {"karate": 2.5, "iceland": 54.7, "dolphin": None}
    ax.bar(np.arange(3) - 0.2, [old[k] if old[k] else np.nan for k in names], width=0.38, color=ORANGE, label="reversed min-fill-in order")
    ax.bar(np.arange(3) + 0.2, [fe[k]["seconds"] for k in names], width=0.38, color=VIOLET, label="best greedy layout")
    ax.text(2 - 0.2, 0.4, "memory\nlimit", ha="center", va="bottom", fontsize=6.8, color=ORANGE)
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(["karate", "Iceland", "dolphin"], fontsize=7.5); ax.set_ylabel("seconds"); ax.legend(frameon=False, fontsize=6.5, loc="upper left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.9); save(fig, TWIG_DIR, "fig_scaling.pdf")


def fig_accuracy():
    d = json.load(open("results/theory/twig_accuracy.json"))
    fig, ax = plt.subplots(figsize=(3.6, 2.7)); cols = [BLUE, ORANGE, AQUA, MAGENTA]; mks = ["o", "s", "^", "v"]
    th = np.array([100, 1000, 10000])
    for i, x in enumerate(d):
        ax.plot(th, [x["theta"][str(t)]["rel_rmse"] for t in th], marker=mks[i % 4], color=cols[i % 4], lw=1.4, ms=5, label=x["instance"].split(" (")[0])
    ax.plot(th, 1.0 * th ** -0.5, color=GREY, ls="--", lw=0.9); ax.text(400, 0.012, "slope $-1/2$", color=INK2, fontsize=7, rotation=-24)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("samples $\\theta$"); ax.set_ylabel("relative RMS error of the estimate"); ax.legend(frameon=False, fontsize=6.8, loc="lower left"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, TWIG_DIR, "fig_accuracy.pdf")


def fig_planners():
    d = pd.read_csv("results/theory/twig_planners.csv")
    d = d[d.exact_greedy > 1e-9].copy()  # drop degenerate rows where blocking leaves zero spread
    d["excess"] = 100 * (d.ag_mean - d.exact_greedy) / d.exact_greedy
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, b in zip(axes, (2, 4)):
        g = d[d.budget == b]; graphs = list(dict.fromkeys(g.graph)); rng = np.random.default_rng(2)
        for k, (theta, col, mk) in enumerate(((100, ORANGE, "o"), (1000, VIOLET, "D"))):
            h = g[g.theta == theta]
            for i, gr in enumerate(graphs):
                v = h[h.graph == gr].excess.values
                ax.scatter(i + (k - 0.5) * 0.28 + rng.uniform(-0.05, 0.05, len(v)), v, s=22, color=col, marker=mk, edgecolor="#fcfcfb", linewidth=0.4, zorder=3, label=f"AG, $\\theta={theta}$" if i == 0 else None)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(range(len(graphs))); ax.set_xticklabels(graphs, fontsize=6.8, rotation=20, ha="right"); ax.set_title(f"budget $b={b}$", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("exact spread of AG's plan,\n% above exact greedy"); axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=0.8); save(fig, TWIG_DIR, "fig_planners.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["gap_curve", "sop", "regime", "zone", "ilp", "frontier", "bell", "gap", "orders", "scaling", "accuracy", "planners"]
    fn = dict(gap_curve=fig_adaptivity_gap, sop=fig_sop, regime=fig_regime, zone=fig_zone, ilp=fig_ilp, frontier=fig_frontier, bell=fig_bell, gap=fig_gap, orders=fig_orders, scaling=fig_scaling, accuracy=fig_accuracy, planners=fig_planners)
    for w in which: fn[w]()
