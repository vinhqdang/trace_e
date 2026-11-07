#!/usr/bin/env bash
# ILP as the one-shot planner inside DEFER (same instances and realisations as run_ilp_baseline.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
POL="none,ag,defer,commit,ilp1,defer_ilp,defer_ilp_nopush,commit_ilp"
ILP="--ilp-k 48 --ilp-h 3 --ilp-branches 12 --ilp-time-limit 30"
COMMON="--n-instances 10 --draws 5 --theta 100 --tag deferilp"
run() { python3 -m trace_e.eval.run_adaptive --policies "$POL" $ILP $COMMON "$@"; }
run --network dolphin --prob-model const --p 0.1 --n-seeds 3 --budgets 2,4,8
run --network dolphin --prob-model const --p 0.2 --n-seeds 3 --budgets 2,4,8
run --network iceland --prob-model const --p 0.3 --n-seeds 2 --budgets 2,4,8
run --network karate --prob-model const --p 0.2 --n-seeds 2 --budgets 2,4
run --network workplace --prob-model const --p 0.05 --n-seeds 3 --budgets 2,4,8
run --network ba:150:m=2 --prob-model const --p 0.1 --n-seeds 3 --budgets 2,4,8
run --network highschool2013 --prob-model const --p 0.03 --n-seeds 3 --budgets 4,8
