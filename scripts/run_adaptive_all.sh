#!/usr/bin/env bash
# Adaptive IMIN benchmark: DEFER vs one-shot AG / GR / LSBM / heuristics, same realisations and budgets.
set -uo pipefail
cd "$(dirname "$0")/.."
NETWORKS=${NETWORKS:-"cagrqc cahepth powergrid ba:2000:m=2"}
POLICIES=${POLICIES:-"none,ag,gr,lsbm,isocut,proximity,defer,defer_nopush,defer_gr,defer_cut,commit"}
COMMON=${COMMON:-"--n-instances 10 --draws 5 --theta 100"}
PUSH=${PUSH:-1}
run() {
  python3 -m trace_e.eval.run_adaptive "$@"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log adaptive blocking results ($*)" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
}
for NET in $NETWORKS; do
  run --network "$NET" --prob-model const --p 0.05 --n-seeds 20 --seed-rule random --budgets 5,10,20 --policies "$POLICIES" $COMMON
  run --network "$NET" --prob-model const --p 0.1 --n-seeds 20 --seed-rule random --budgets 10,20,50 --policies "$POLICIES" $COMMON
  run --network "$NET" --prob-model wc --n-seeds 20 --seed-rule random --budgets 10,20,50 --policies "$POLICIES" $COMMON
  run --network "$NET" --prob-model const --p 0.1 --n-seeds 20 --seed-rule degree --budgets 10,20,50 --policies "$POLICIES" $COMMON
  run --network "$NET" --prob-model const --p 0.05 --n-seeds 50 --seed-rule random --budgets 20,50,100 --policies "$POLICIES" $COMMON
done
