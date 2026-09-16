#!/usr/bin/env bash
# Detectability over time: sweep the observation time T on one network.
set -uo pipefail
cd "$(dirname "$0")/.."
NET=${NET:-dolphin}
METHODS=${METHODS:-"random,jordan,rumor,netsleuth,dmp,mcs,gcn_skip"}
TS=${TS:-"0.25 0.5 0.85 1.5 2.5 4.0"}
TEST_SIMS=${TEST_SIMS:-50}
PUSH=${PUSH:-1}
for T in $TS; do
  python3 -m trace_e.eval.run_source --network "$NET" --methods "$METHODS" --T "$T" --test-sims "$TEST_SIMS" --tag "${NET}_T${T}" --notes "T_sweep"
  python3 scripts/aggregate.py
  git add -A results && git commit -q -m "Log T-sweep results for $NET at T=$T" || true
  if [ "$PUSH" = "1" ]; then git push -q origin main || true; fi
done
