#!/usr/bin/env bash
# Run the Problem A baseline suite on the empirical networks and commit the
# results after each network so partial progress is always logged.
set -uo pipefail
cd "$(dirname "$0")/.."

NETWORKS=${NETWORKS:-"karate iceland dolphin fraternity workplace highschool2013 powergrid"}
METHODS=${METHODS:-"random,degree,jordan,distance,rumor,netsleuth,dmp,sme,mcs,mlp,gcn,gcn_skip,igcn"}
TEST_SIMS=${TEST_SIMS:-100}
EXTRA=${EXTRA:-""}
PUSH=${PUSH:-1}

for NET in $NETWORKS; do
  TS=$TEST_SIMS
  if [ "$NET" = "powergrid" ]; then TS=${POWERGRID_TEST_SIMS:-10}; fi
  echo "=== $NET ==="
  python3 -m trace_e.eval.run_source --network "$NET" --methods "$METHODS" --test-sims "$TS" $EXTRA
  git add results
  git commit -q -m "Log source-detection results for $NET" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
done
