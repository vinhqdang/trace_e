#!/usr/bin/env bash
# Run the Problem B baseline suite (both intervention modes) and commit after each run.
set -uo pipefail
cd "$(dirname "$0")/.."

NETWORKS=${NETWORKS:-"karate dolphin iceland fraternity workplace highschool2013 powergrid"}
MODES=${MODES:-"block counter"}
METHODS=${METHODS:-"random,degree,pagerank,proximity,reach,greedy,greedy_dom"}
EXTRA=${EXTRA:-"--budgets 1,2,5,10 --n-instances 20 --n-mc 1000 --greedy-samples 200"}
PUSH=${PUSH:-1}

for NET in $NETWORKS; do
  for MODE in $MODES; do
    echo "=== $NET / $MODE ==="
    python3 -m trace_e.eval.run_blocking --network "$NET" --mode "$MODE" --methods "$METHODS" $EXTRA
    git add -A results
    git commit -q -m "Log blocking results for $NET ($MODE mode)" || true
    if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
  done
done
