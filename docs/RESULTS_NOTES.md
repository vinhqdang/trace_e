# Results notes (running log)

## DEFER: adaptive influence minimisation (2026-09-16)

Protocol (`trace_e/eval/run_adaptive.py`): every policy runs on the same
live-edge realisations with the same total budget; one-shot planners see the
seeds at time 0; adaptive policies observe activations round by round.
theta = 100 samples, 10 instances x 5 realisations per cell.

**What the first pass taught.** The deferral lemma holds exactly (lazy
commitment of AdvancedGreedy's plan reproduces AdvancedGreedy node for node).
Two confounders had to be removed before the effect of adaptivity could be
measured: (i) the plan returned by any sample-based greedy varies a lot
between sample draws at theta = 50-100 (on Epinions the same AG run gave
13 or 31 remaining nodes depending on the draw), so DEFER now keeps the
previous plan's leftover unless the new plan beats it on the current
samples (plan protection, the hypothesis of Theorem 1); (ii) frontier
isolation moves in the planner helped on ca-HepTh but hurt on Epinions, so
the default planner is AdvancedGreedy itself and the isolation variant is
reported as `defer_cut`.

**Where adaptivity pays: low activation probability, tight budget.**
ca-GrQc, constant p = 0.05, 20 random seeds (no intervention 34.4):

| budget | AG | GR | LSBM | proximity | DEFER | DEFER no push-down | COMMIT |
|---|---|---|---|---|---|---|---|
| 5 | 25.8 | 25.9 | 25.8 | 25.5 | **24.8** (uses 4.7) | 28.0 | 28.0 |
| 10 | 21.2 | 21.3 | 20.9 | 20.5 | **19.2** (uses 9.5) | 20.5 | 20.5 |
| 20 | 14.6 | 14.7 | 13.6 | 13.4 | 14.2 (uses 18.4) | 14.3 | 14.3 |

DEFER is the only adaptive variant that beats the one-shot planners here, and
it does so while spending less budget; the push-down rule (defer an exposed
node when protecting its children later is cheaper in expectation) is what
makes the difference, exactly the mechanism of the adaptivity-gap
construction. Without push-down, adaptivity brings nothing on this graph.

**Adaptivity gap on the star-of-paths family** (`results/theory/adaptivity_gap.json`):
the measured ratio of nodes saved by DEFER to nodes saved by the best
one-shot blocker tracks the theoretical (1-(1-p)^Delta)/p closely, e.g.
Delta = 16, p = 0.05: 12.1 measured vs 11.2 theory; Delta = 32, p = 0.02:
20.2 vs 23.8.

**Scalability.** soc-Epinions1 (75,879 nodes, 811k directed edges),
weighted cascade, 20 seeds, budget 50: AdvancedGreedy 0.05-0.2 s, DEFER
0.3-0.8 s per episode (planning once per round on the current reachable
region), pure Python.

**Earlier configuration (dominator planner with isolation moves, p = 0.1,
budgets 20/50/100).** At budget 20 that variant beat AG by 10.2 ± 6.0 nodes
(paired, 70% of realisations) and GR by 6.0 ± 5.1; at budgets 50 and 100 all
planners seal the seeds and coincide. The full suite
(`scripts/run_adaptive_all.sh`: 4 graphs x {p = 0.05 tight, p = 0.1,
weighted cascade, degree seeds, 50 seeds}) is running and regenerates
`results/REPORT.md` after each configuration, including paired differences
with standard errors.

## Theorems 1-2 (ca-GrQc, `results/theory/`)

* False-alarm rate of the e-process across alpha in {0.2, 0.1, 0.05, 0.02,
  0.01, 0.005}: 0.026, 0.016, 0.010, 0.0025, 0.0013, 0.0013 under kappa = 1;
  lower still when benign cascades are below baseline (dominance null). The
  guarantee holds with slack because most benign cascades die before the
  evidence can cross the threshold.
* Harm at alarm (bad nodes beyond the seeds) vs. the Theorem 2 bounds at
  alpha = 0.05: kappa = 2: 18.6 in [3.3, 32.1]; kappa = 3: 11.8 in
  [2.1, 13.3]; kappa = 4: 11.1 (median 9) vs upper bound 9.8 without the
  overshoot term; kappa = 6: 11.0 vs 6.2 (overshoot dominates). The
  no-intervention spread of the same cascades is 800-4000 nodes.

## Detection (ca-GrQc, detector + no container)

| setting | e-process FA / det / harm@alarm | CUSUM (calibrated) FA / det / harm@alarm | size threshold FA |
|---|---|---|---|
| in-distribution | 0.007 / 0.79 / 17.7 | 0.05 / 0.90 / 11.5 | 0.067 |
| calibrated with 1 seed, deployed with 3 | 0.003 / 0.71 / 17.8 | 0.14 / 0.83 / 12.2 | 0.137 |

Calibrated thresholds are sharper in distribution but their false-alarm
rate triples under a mild shift of the seed-set size; the e-process needs
no calibration data and stays below alpha. The timing oracle (`immediate`)
has harm@alarm 8.9 (round 1).

## Containment (ca-GrQc, e-process detector, budget 10)

| container | saved fraction | final harmful spread (no intervention 1819) |
|---|---|---|
| adaptive frontier (AVID) | 0.479 | 949 |
| one-shot dominator greedy with learned kappa | 0.479 | 949 |
| one-shot greedy with baseline p0 (ablation) | 0.298 | 1277 |
| proximity | 0.428 | 1042 |
| degree | 0.039 | 1749 |

Planning with the e-process's transmissibility estimate instead of the
benign baseline is worth 18 points of saved fraction. Adaptive deferral and
one-shot planning coincide on 234 of 237 detected cascades at this budget:
either the first plan seals the frontier or the cascade has already broken
out (130 of 237 escape with mean final size 2170 and mean harm@alarm 20).
The value of deferral has to be measured in the intermediate regime (budget
sweep, slower cascades), which is queued in `scripts/run_sequential_extra.sh`.

## Vertex blocking (Problem B, ca-HepTh, const p = 0.1, 20 seeds, 3 instances)

Exact dominator-tree gains with cut-aware isolation moves (`greedy_dom`)
vs. CELF with a candidate pool (`greedy`) vs. proximity, saved fraction at
budget 50: 0.85 / 0.82 / 0.80, with selection time 3 s / 36 s / 0.3 s.
Without isolation moves, one-node greedy loses to proximity on this
instance (0.61 vs 0.69 at budget 20), which is the non-submodular
"coordinated cut" effect described in `docs/METHOD.md`.

## Concurrent cascades (ca-GrQc, M = 400, rho = 0.2, alpha = 0.1, 3 repetitions)

| rule | FDP | power | saved |
|---|---|---|---|
| per-cascade threshold | 0.03-0.12 | 0.79-0.86 | 0.44-0.57 |
| e-BH | 0.000 | 0.63-0.77 | 0.31-0.37 |
| harm-weighted e-BH | 0.00-0.03 | 0.69-0.78 | 0.39-0.42 |

Both e-BH variants keep the false discovery proportion under alpha; the
harm weights add 3-10 points of saved fraction at no cost in FDP.

## One-shot IMIN: what was tried against AdvancedGreedy / GreedyReplace (2026-09-16, evening)

All on ca-HepTh and ca-GrQc, constant p = 0.1, 20 random seeds, budget 20;
quality = spread on 300 *fresh* live-edge realisations divided by the
no-intervention spread (lower is better), averaged over 3 instances.

| method | ca-HepTh | ca-GrQc | note |
|---|---|---|---|
| AG, theta = 100 | 0.477 | 0.432 | |
| AG, theta = 300 | 0.358 | 0.427 | |
| AG, theta = 1000 | 0.328 | 0.422 | |
| GR, theta = 300 | 0.358 | 0.415 | |
| SWAP (1-swap local search from AG) | = AG or worse (overfits the samples) | | |
| LCB greedy (mean − s.e.) | worse than AG | | |
| cross-validated greedy | worse than AG | | |
| bagged AG (5 x 60, 10 x 30) | = AG at equal total samples | | |
| ISOCUT / CUTGREEDY (isolation and min-cut group moves) | = AG (moves never win by ratio); 200x slower for CUTGREEDY | | |
| PH-CUT (scenario min-cuts + progressive hedging) | = AG fallback; Lagrangian dual bound near 0 | | |
| RAG (racing greedy, adaptive theta up to 1500) | 0.324 | 0.422 | = AG(1000), 1.5x its time; the top-20 candidates stay statistically tied |

**Diagnosis.** Cascade sizes are heavy-tailed (ca-HepTh: median 665,
mean 478, 62% of realisations contain a giant component). The sample
objective of AG is far below its true objective: theta = 60 gives F = 31 on
its own samples vs 205 on fresh ones; theta = 300 gives 65 vs 98. Greedy
therefore selects on sampling noise, and the true quality improves
monotonically with theta. Among the top candidates at each greedy step the
gains are statistically indistinguishable even at 1500 samples, so racing
cannot save samples, and the exact per-scenario optimum (min-cut) is 0 with
budget 20 (each scenario alone is trivially sealed), so the wait-and-see
bound is vacuous: the difficulty of IMIN is entirely in the coupling of
scenarios. The one lever that reliably improves quality is more samples;
AG's cost is linear in theta.
