#!/usr/bin/env bash
# Follow-up sequential experiments: power-matched alphas, container ablations, budget sweep with the bounded-horizon planner.
set -uo pipefail
cd "$(dirname "$0")/.."
COMMON=${COMMON:-"--prob-model wc --n-seeds 3 --kappa-min 1.5 --kappa-max 4 --n-benign 300 --n-harmful 300 --n-calib 300 --samples 30 --horizon 4 --replan-every 2"}
DET_PAIRS="eprocess:none,sprt:none,cusum:none,size:none,growth:none,excess:none,logistic:none"
ABL_PAIRS="eprocess:adaptive,eprocess:greedy,eprocess:greedy_p0,eprocess:adaptive_p0,eprocess:adaptive_commit,eprocess:proximity,immediate:adaptive"
PUSH=${PUSH:-1}
run() {
  python3 -m trace_e.eval.run_sequential "$@"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log sequential follow-up results ($*)" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
}
for A in 0.2 0.5; do
  run --network cagrqc $COMMON --budget 10 --alpha $A --pairs "$DET_PAIRS" --notes alpha_sweep
done
for NET in highschool2013 powergrid cahepth ba:2000:m=2; do
  run --network "$NET" $COMMON --budget 10 --pairs "$ABL_PAIRS" --notes ablation
done
for B in 3 5 20 40; do
  run --network cagrqc $COMMON --budget $B --pairs "$ABL_PAIRS" --notes budget_sweep
done
