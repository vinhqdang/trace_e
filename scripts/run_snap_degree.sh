#!/usr/bin/env bash
# Robustness runs with the current code: 50 highest-degree seeds, and the isolation-move ablations (isocut, defer_cut).
set -uo pipefail
cd "$(dirname "$0")/.."
POL="none,ag,commit,defer,isocut,defer_cut"
for NET in cagrqc cahepth powergrid ba:2000:m=2; do
  python3 -m trace_e.eval.run_adaptive --network $NET --prob-model const --p 0.1 --n-seeds 50 --seed-rule degree --budgets 10,20,50 \
    --n-instances 20 --draws 5 --theta 100 --tag snap2deg --policies $POL
done
