#!/usr/bin/env bash
# Active throttling (Theorem 5): Pareto sweep of final harm vs benign cost for hub / uniform / random throttling.
set -uo pipefail
cd "$(dirname "$0")/.."
NET=${NET:-cagrqc}
COMMON=${COMMON:-"--prob-model wc --n-seeds 3 --budget 10 --kappa-min 1.5 --kappa-max 4 --n-benign 300 --n-harmful 300 --n-calib 50 --samples 30 --horizon 4 --replan-every 2"}
ALPHA_SOFT=${ALPHA_SOFT:-0.5}
PUSH=${PUSH:-1}
run() {
  python3 -m trace_e.eval.run_sequential --network "$NET" $COMMON --alpha-soft $ALPHA_SOFT "$@"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log throttling sweep ($NET $*)" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
}
run --pairs eprocess:adaptive:none --notes throttle_sweep
for R in 0.1 0.3; do
  for F in 0.3 0.5 0.7; do
    run --pairs eprocess:adaptive:hub,eprocess:adaptive:random --rho-min $R --throttle-frac $F --notes throttle_sweep
  done
done
for R in 0.3 0.5 0.7 0.85; do
  run --pairs eprocess:adaptive:uniform --rho-min $R --notes throttle_sweep
done
