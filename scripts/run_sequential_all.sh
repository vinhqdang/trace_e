#!/usr/bin/env bash
# AVID benchmark suite: main comparison, calibration-shift robustness, dominance null, alpha sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
NETWORKS=${NETWORKS:-"cagrqc highschool2013 powergrid cahepth ba:2000:m=2"}
COMMON=${COMMON:-"--prob-model wc --n-seeds 3 --budget 10 --n-benign 300 --n-harmful 300 --n-calib 300 --samples 50 --pool 200"}
PUSH=${PUSH:-1}
run() {
  python3 -m trace_e.eval.run_sequential "$@"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log sequential detect-and-contain results ($*)" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
}
for NET in $NETWORKS; do
  run --network "$NET" $COMMON --notes main
  run --network "$NET" $COMMON --calib-n-seeds 1 --notes calib_shift
  run --network "$NET" $COMMON --benign-kappa-min 0.5 --notes dominance_null
done
for A in 0.01 0.02 0.1; do
  run --network cagrqc $COMMON --alpha $A --pairs eprocess:adaptive,sprt:greedy,cusum:greedy,size:greedy,excess:greedy,logistic:greedy --notes alpha_sweep
done
for B in 3 5 20 40; do
  run --network cagrqc $COMMON --budget $B --pairs eprocess:adaptive,eprocess:greedy,eprocess:proximity,eprocess:degree,immediate:greedy --notes budget_sweep
done
