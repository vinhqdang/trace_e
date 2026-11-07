#!/usr/bin/env bash
# Regime scan: where does deferral (DEFER) beat greedy (AG) and re-plan-and-commit (COMMIT)? Full grid reported, wins and losses.
set -uo pipefail
cd "$(dirname "$0")/.."
POL="none,ag,gr,defer,defer_nopush,commit"
COMMON="--n-instances 10 --draws 10 --theta 100 --tag regime --policies $POL"
for NET in tree:1000 grid:1024:shortcuts=0.0 rgg:1000:deg=4 ws:1000:k=4:p=0.05; do
  for P in 0.2 0.35 0.5; do
    python3 -m trace_e.eval.run_adaptive --network $NET --prob-model const --p $P --n-seeds 10 --budgets 5,20,50 $COMMON
  done
done
for P in 0.03 0.06 0.1; do
  python3 -m trace_e.eval.run_adaptive --network ba:1000:m=2 --prob-model const --p $P --n-seeds 10 --budgets 5,20,50 $COMMON
done
