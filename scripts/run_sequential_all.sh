#!/usr/bin/env bash
# AVID benchmark suite: main comparison, calibration-shift robustness, dominance null, alpha sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
NETWORKS=${NETWORKS:-"cagrqc highschool2013 powergrid cahepth ba:2000:m=2"}
COMMON=${COMMON:-"--prob-model wc --n-seeds 3 --budget 10 --kappa-min 1.5 --kappa-max 4 --n-benign 300 --n-harmful 300 --n-calib 300 --samples 30"}
# detection comparison uses no container (cheap); containment comparison fixes the e-process detector
DET_PAIRS="eprocess:none,sprt:none,cusum:none,size:none,growth:none,excess:none,logistic:none,never:none,immediate:none"
CON_PAIRS="eprocess:adaptive,eprocess:greedy,eprocess:proximity,eprocess:degree,immediate:adaptive"
PUSH=${PUSH:-1}
run() {
  python3 -m trace_e.eval.run_sequential "$@"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log sequential detect-and-contain results ($*)" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
}
for NET in $NETWORKS; do
  run --network "$NET" $COMMON --pairs "$DET_PAIRS" --notes main
  run --network "$NET" $COMMON --calib-n-seeds 1 --pairs "$DET_PAIRS" --notes calib_shift
  run --network "$NET" $COMMON --benign-kappa-min 0.5 --pairs "$DET_PAIRS" --notes dominance_null
  run --network "$NET" $COMMON --pairs "$CON_PAIRS" --notes main
done
for A in 0.01 0.02 0.1; do
  run --network cagrqc $COMMON --alpha $A --pairs "$DET_PAIRS" --notes alpha_sweep
done
for B in 3 5 20 40; do
  run --network cagrqc $COMMON --budget $B --pairs eprocess:adaptive,eprocess:greedy,eprocess:proximity,eprocess:degree,immediate:greedy --notes budget_sweep
done
