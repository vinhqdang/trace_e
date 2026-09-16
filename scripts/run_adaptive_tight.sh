#!/usr/bin/env bash
# Tight-budget regime for adaptive IMIN (where hedging costs one-shot planners the most).
set -uo pipefail
cd "$(dirname "$0")/.."
NETWORKS=${NETWORKS:-"cagrqc cahepth powergrid ba:2000:m=2"}
POLICIES=${POLICIES:-"none,ag,gr,lsbm,proximity,defer,defer_nopush,defer_gr,commit"}
PUSH=${PUSH:-1}
run() {
  python3 -m trace_e.eval.run_adaptive "$@"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log adaptive blocking results ($*)" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
}
for NET in $NETWORKS; do
  run --network "$NET" --prob-model const --p 0.1 --n-seeds 20 --budgets 5,10,20 --n-instances 20 --draws 5 --theta 100 --policies "$POLICIES"
  run --network "$NET" --prob-model const --p 0.2 --n-seeds 10 --budgets 10,20,40 --n-instances 20 --draws 5 --theta 100 --policies "$POLICIES"
done
