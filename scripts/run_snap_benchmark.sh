#!/usr/bin/env bash
# Empirical / SNAP-scale benchmark with the current code: every policy on the same realisations (paired comparisons).
set -uo pipefail
cd "$(dirname "$0")/.."
POL="none,ag,gr,lsbm,proximity,commit,defer,defer_nopush"
COMMON="--n-instances 20 --draws 5 --theta 100 --n-seeds 20 --seed-rule random --budgets 5,10,20,50,100 --tag snap2 --policies $POL"
for NET in cagrqc cahepth powergrid ba:2000:m=2; do
  python3 -m trace_e.eval.run_adaptive --network $NET --prob-model const --p 0.1 $COMMON
  python3 -m trace_e.eval.run_adaptive --network $NET --prob-model wc $COMMON
done
