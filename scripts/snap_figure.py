"""Figure: saved fraction versus budget for AG, COMMIT and DEFER on the four networks (constant p = 0.1).
Reads the per-episode logs of scripts/run_snap_benchmark.sh via scripts/snap_tables.py. Writes paper/defer/figures/defer_budget_sweep.pdf."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from snap_tables import load, sat, NETS

BUDGETS = (5, 10, 20, 50, 100)
SERIES = [("ag", "AG", "#2a78d6", "o", 1.6), ("commit", "COMMIT", "#eb6834", "s", 1.6), ("defer", "DEFER", "#6250d6", "D", 2.4)]
cells = load()
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=False)
for ax, (net, lab) in zip(axes, NETS):
    for key, name, col, mk, lw in SERIES:
        y = [sat(cells[(net, "const", b)], key) for b in BUDGETS]
        ax.plot(BUDGETS, y, color=col, marker=mk, markersize=4.5, linewidth=lw, label=name, markeredgecolor="#fcfcfb", markeredgewidth=0.8)
    ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.minorticks_off(); ax.set_title(lab, fontsize=8.5, loc="left"); ax.grid(axis="y", color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("budget $b$")
axes[0].set_ylabel("fraction of spread saved")
axes[0].legend(frameon=False, loc="lower right", fontsize=7.5, handlelength=1.6)
fig.tight_layout(w_pad=1.0)
os.makedirs("paper/defer/figures", exist_ok=True)
fig.savefig("paper/defer/figures/defer_budget_sweep.pdf")
print("saved")
