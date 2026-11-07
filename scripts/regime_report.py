"""Paired comparison of policies from run_adaptive episode logs (same instance+draw+budget => paired)."""
import glob, gzip, json, sys
import numpy as np
import pandas as pd

TAG = sys.argv[1] if len(sys.argv) > 1 else "regime"
A, B = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("defer", "ag")
rows = []
for d in sorted(glob.glob(f"results/runs/adaptive_*{TAG}*")):
    cfg = json.load(open(f"{d}/config.json"))
    cfg = cfg.get("config", cfg)
    ep = pd.DataFrame([json.loads(l) for l in gzip.open(f"{d}/instances_episodes.jsonl.gz", "rt")])
    for b, g in ep.groupby("budget"):
        p = g.pivot_table(index=["inst", "draw"], columns="policy", values="spread")
        base = p["none"]
        diff = (p[B] - p[A])  # positive = A has smaller spread = A better
        n = len(diff)
        rows.append(dict(net=cfg["network"], p=cfg["p"], b=b, none=base.mean(), **{f"{x}_saved": 1 - p[x].sum() / base.sum() for x in (A, B)},
                         gain=diff.mean(), se=diff.std(ddof=1) / np.sqrt(n), pct=100 * diff.sum() / base.sum(), n=n))
r = pd.DataFrame(rows)
r["z"] = r.gain / r.se.replace(0, np.nan)
r["verdict"] = np.where(r.z > 2, f"{A} wins", np.where(r.z < -2, f"{B} wins", "tie"))
pd.set_option("display.width", 200)
print(f"A={A} vs B={B}: gain = mean spread({B}) - spread({A}) (nodes saved by {A}); pct = gain as % of unprotected spread")
print(r.round(3).to_string(index=False))
print(r.verdict.value_counts().to_string())
