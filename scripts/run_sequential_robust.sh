#!/usr/bin/env bash
# Robustness of false-alarm control: baseline probabilities misspecified for all detectors
# (cascades follow p0, detectors believe p0 * P0_SCALE), with and without the inflated-null margin.
set -uo pipefail
cd "$(dirname "$0")/.."
NETWORKS=${NETWORKS:-"cagrqc highschool2013 powergrid"}
COMMON=${COMMON:-"--prob-model wc --n-seeds 3 --budget 10 --kappa-min 1.5 --kappa-max 4 --n-benign 300 --n-harmful 300 --n-calib 300 --samples 30"}
DET_PAIRS="eprocess:none,sprt:none,cusum:none,size:none,growth:none,excess:none,logistic:none"
PUSH=${PUSH:-1}
run() {
  python3 -m trace_e.eval.run_sequential "$@"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log sequential robustness results ($*)" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
}
for NET in $NETWORKS; do
  for S in 0.8 0.67; do
    run --network "$NET" $COMMON --p0-scale $S --pairs "$DET_PAIRS" --notes "misspec_p0x$S"
    run --network "$NET" $COMMON --p0-scale $S --kappa-null 1.25 --pairs "eprocess:none,sprt:none,cusum:none" --notes "misspec_p0x${S}_margin1.25"
  done
done
