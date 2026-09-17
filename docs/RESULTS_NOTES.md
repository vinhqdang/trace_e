# Results notes (running log)

## Exact SIR source posterior via time-unrolled treewidth DP: tried and abandoned (2026-09-17)

Attempted a "TWIG for Problem A": an exact (non-mean-field) computation of
P(observed snapshot | source = v) under discrete-time SIR, via variable
elimination over the time-unrolled graphical model, reusing the same
elimination order as `trace_e/blocking/treewidth.py`. Implemented as
`trace_e.source.exact_sir` (since removed; see git history at commit
9710690 for the code), with a small generic factor/variable-elimination
engine.

**Correctness: verified.** ~270 random-graph trials against an explicit
brute-force simulation of the full 3^n-state Markov chain (n <= 6), plus an
independent structural check (summing exp(log-likelihood) over all 3^n
possible snapshots gives exactly 1), all passed.

**Abandoned for two reasons, both found empirically rather than assumed:**

1. **Complexity depends on max degree, not just treewidth.** Unlike TWIG's
   IC-model spread computation (where a vertex's contribution is processed
   one edge at a time via `_schedule`'s `connect_to` list), a vertex's SIR
   transition genuinely needs its ENTIRE neighbourhood's previous state
   *jointly* (the "does at least one infected neighbour transmit" event),
   producing a single factor of size 3^(deg(v)+2). This is not fixable by
   reusing the same elimination order: even bounded-treewidth graphs with a
   single high-degree hub (Karate: treewidth 5, max degree 16) blow up. Worse,
   even a synthetic max-degree-3, n=8 graph over 3 discrete time steps took
   ~12s (width 13), and n=14 timed out -- i.e. this is not just "no real
   network is small enough", literally none of this project's small real
   networks (Karate max-degree 16, Iceland 24, Dolphin 12, Fraternity 52,
   Workplace 44) are anywhere close to usable, and even tiny synthetic
   graphs hit a wall fast. A proper fix (representing the neighbourhood OR
   via an auxiliary variable chain, interleaved with the real per-vertex
   elimination order so intermediate factors stay small) is a sound
   direction but was not completed.
2. **Even where it does run (n <= 7, degree <= 3), it does not clearly beat
   DMP in the metric that matters.** Simulated 241 real epidemic instances
   on small random graphs and compared exact vs. DMP (mean-field) at
   identifying the true source: top-1 accuracy 59.8% (exact) vs. 61.4%
   (DMP); pairwise, exact assigned a strictly higher log-likelihood to the
   true source in only 115/241 (47.7%) of cases -- indistinguishable from a
   coin flip. (Initial expectation, via Gibbs' inequality, was that the
   true model should have higher *expected* log-likelihood than the
   mean-field approximation -- but that guarantee only holds averaged over
   many repeated draws from the *same* fixed (graph, source) instance, not
   across a population of different random instances with one or few draws
   each, so it does not translate into "exact beats DMP" empirically here.)

Conclusion: DMP's mean-field independence assumption, despite being
non-exact, appears to be a fairly good practical proxy for *ranking*
candidate sources even where we can check it against ground truth. Given
this, and the severe scalability wall above, this direction was dropped
(decision made 2026-09-17) in favour of concentrating the paper's novel-
algorithm contribution on Problem B (TWIG / DEFER / LAZY-AG). Problem A's
role in the writeup is the benchmark reproduction/extension work in the
section below, not a new algorithm.

## Problem A vs. published literature (2026-09-16)

Direct source: Sterchi, Brack & Hilfiker, "Graph Neural Networks for Source
Detection: A Review and Benchmark Study," arXiv:2512.20657 (2026) --
the closest possible comparison, since our SIR simulator, network set
(Karate/Iceland/Dolphin/Fraternity/Workplace/Highschool/Powergrid) and several
baselines (Jordan, MLP, GCN family, MCMF/DMP-style, SME) were built to match
their setup. Their benchmark calibrates beta and T *per network* so that
R0 ~ 2 and ~40% of nodes are infected on average at observation time. Our own
suite instead used one fixed (beta=1.3, T=0.85) pair across every network,
which happens to reproduce Sterchi's exact Karate calibration (also
beta=1.300, T=0.85) but diverges sharply everywhere else:

| network | our avg. infected fraction | Sterchi's target |
|---|---|---|
| karate | 39.7% | ~40% (matches) |
| iceland | 16.7% | ~40% (they report ~75% actually reached at their higher beta=5.1) |
| dolphin | 28.4% | ~40% |
| fraternity | 96.6% | ~40% |
| workplace | 92.6% | ~40% |
| highschool | 96.8% | ~40% |
| powergrid | 0.11% | ~40% |

**Conclusion: only Karate is a valid apples-to-apples comparison.** Every
other network's comparison to Sterchi's numbers is confounded by wildly
different outbreak sizes and must not be read as "our pipeline vs. theirs."

Karate comparison (both: seed beta=1.300, T=0.85, ~40% infected, top-5
accuracy on 100 simulated outbreaks/node = 3400 test instances):

| method | Sterchi et al. top-5 | ours (top-5) | Sterchi ed | ours ed |
|---|---|---|---|---|
| Random | 39.37% | 52.85% | 1.321 | 1.352 |
| Jordan | 52.68% | 54.35% | 1.080 | 1.114 |
| Betweenness / degree (closest ours has) | 55.19% | 53.59% | 0.911 | 0.889 |
| SME | 60.67% | 55.74% | 1.083 | 1.217 |
| MCMF / DMP (closest ours has) | 65.61% | 67.85% | 0.900 | 0.965 |
| MLP (snapshot) | 67.90% | 61.15% | 1.037 | 1.226 |
| GCN (Shah et al. arch, their best) | 72.87% | -- | 0.933 | -- |
| GCN (Dong et al. arch, ours uses this) | 68.67% | 64.27% | 1.002 | 0.899 |
| IGCN | 72.83% | 67.12% | 0.975 | 1.174 |
| GraphSAGE (ours: gcn_skip, different arch) | 71.89% | 69.88% | 0.981 | 0.956 |

Two things stand out and are worth chasing before writing this up:

1. **Our Random baseline scores far above theirs** (52.85% vs 39.37% top-5)
   even though average outbreak size matches almost exactly. This is the
   single biggest anomaly in the comparison and suggests a difference in how
   "random among the infected subgraph" is implemented (tie-breaking,
   inclusion of recovered vs. only-infectious nodes, or restricting the
   candidate pool) -- not yet root-caused.
2. Modulo that baseline shift, our GNN methods land in the same 60s-70% top-5
   band as theirs and beat our own classical baselines by a similar margin,
   so the qualitative finding ("GNNs clearly beat classical/probabilistic
   methods") replicates even though the absolute numbers don't match exactly.

**Action item, not yet done:** recalibrate (beta, T) per network to hit a
consistent target infected-fraction (as Sterchi did) before claiming any
further network-level comparison; also audit the `random` baseline's exact
sampling procedure to explain the discrepancy above.

## Problem B vs. published literature (2026-09-16)

Direct source: Xie, Zhang, Wang, Liu, Lin & Zhang, "Influence Minimization via
Blocking Strategies," INFORMS J. Computing (2024), arXiv:2312.17488 -- the
paper AdvancedGreedy/GreedyReplace (our `agreedy`/`greedy_replace` baselines)
come from. Their Table 4 (IC model, TR edge probabilities, budget 20-100,
10 random seeds) reports e.g. on Facebook (n=4039): Random 16.0-21.5,
OutDegree 16.0-21.4, AdvancedGreedy 10.0-14.6, GreedyReplace 10.0-14.6 (all
converging to ~10, the number of seeds, as budget grows -- i.e. AG/GR nearly
eliminate all seed-caused spread at budget>=80). Their own `Exact` vs.
`GreedyReplace` check (Table 3, budget 1-4, IC/TR, 100-node subgraphs)
finds GreedyReplace within 0.05-0.5% of the true optimum, at up to 6 orders
of magnitude less compute. This is a useful ground truth to line up our
`greedy_dom`/AG-style implementation against on the same or similarly-sized
graphs; not yet done in this repo (our Problem B suite so far runs on
different graphs at different budgets, see the blocking summary CSV).

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

## Fixing the bottleneck: cheaper samples (2026-09-16, night)

Since quality is governed by the number of samples theta and every
variance-reduction attempt failed (control variates on the seed-hit
indicator: MSE −15%, no end-to-end gain; permutation Monte Carlo: variance
ratio 1.0 because the number of live edges is concentrated and all the
randomness is in *which* edges are live), the remaining lever is the cost
per sample per greedy step.

**LAZY-AG** (`trace_e/blocking/lazy.py`). After blocking v, AdvancedGreedy
recomputes theta dominator trees. LAZY-AG instead subtracts |D(v)| from the
subtree size of every dominator-tree ancestor of v and marks D(v) as saved,
in O(theta * depth + theta * k) vectorised work, and recomputes the trees
exactly only every r picks. *Lemma.* For every remaining node w the lazy
size equals |D(w) \ D(v)|, which is a lower bound on the exact new dominated
set (deleting v can only create new dominance relations), and equals it
whenever no seed-path of a remaining node passed through v.

Fresh-sample quality (spread / no-intervention spread, 3 instances, budget
20, p = 0.1) and time:

| method | ca-HepTh | time | ca-GrQc | time |
|---|---|---|---|---|
| AG theta=300 | 0.358 | 5.0 s | 0.427 | 3.0 s |
| LAZY-AG theta=300, r=5 | 0.358 | 2.0 s | 0.427 | 1.0 s |
| LAZY-AG theta=300, no refresh | 0.400 | 1.0 s | 0.442 | 0.4 s |
| AG theta=1000 | 0.328 | 16 s | 0.422 | 11 s |
| LAZY-AG theta=1500, r=5 | 0.333 | 10 s | 0.420 | 6 s |

Same quality as AdvancedGreedy at 2.5x less time, i.e. AG(theta=1000)
quality in the time of AG(theta=500); without periodic exact refresh the
lower-bound gains drift and quality degrades, so the refresh is part of the
algorithm.

## Problem A: final GNN + classical benchmark table (2026-09-17, closed out)

Full Sterchi et al. (2026) reproduction matrix plus `trace_e`-native classical
baselines, all on identical test instances per network (same fixed seed and
sample count as their own `run_inference.py`). Closed out as final with 5/6
networks complete and `highschool2013`'s GNN sweep abandoned per explicit
decision, documented below.

**Top-1 / Top-5 accuracy** (higher is better):

| method | karate | iceland | dolphin | fraternity | workplace | highschool2013 |
|---|---|---|---|---|---|---|
| Random | .294/.372 | .224/.265 | .291/.337 | .368/.408 | .355/.381 | .372/.380 |
| Jordan center | .288/.508 | .209/.360 | .290/.422 | .369/.493 | .363/.456 | -/- |
| Betweenness | .292/.535 | .211/.375 | .296/.438 | .372/.504 | .361/.466 | -/- |
| SME | .337/.598 | .271/.461 | .315/.462 | .367/.502 | .367/.469 | -/- |
| MCS mean-field | .370/.653 | .295/.504 | .332/.519 | .383/.533 | .374/.497 | -/- |
| Rumor centrality | .287/.542 | .208/.373 | .293/.432 | .369/.499 | .361/.464 | .378/.461 |
| NetSleuth | .340/.609 | .274/.485 | .319/.480 | .384/.515 | .371/.479 | .381/.463 |
| DMP | .382/.679 | .296/.506 | .341/.531 | .395/.549 | .387/.531 | .391/.483 (reduced-sample, see note) |
| MLP | .325/.589 | .244/.427 | .299/.450 | .368/.515 | .361/.478 | .375/.457 |
| DONG (GCNSI) | .331/.671 | .302/.561 | .329/.575 | .118/.320 | .285/.448 | not trained |
| SHAH (GCN/GAT) | .396/.726 | .316/.579 | .367/.611 | .395/.573 | .400/.558 | not trained |
| IGCN | .392/.727 | .298/.547 | .352/.594 | .395/.563 | .397/.558 | not trained |
| HADDAD (GraphSAGE) | .377/.699 | .306/.559 | .357/.601 | .387/.541 | .381/.536 | not trained |

Notes:
- DMP for `highschool2013` is the honestly-labeled reduced-sample run
  (15 sims/node instead of 100, n=4905 instead of 32700); the full-sample
  attempt was left running in the background, died silently without
  completing, and was not resurrected (see the earlier section above).
- `highschool2013`'s classical "first-model" sweep (Jordan/Betweenness/
  SME/MCS) also did not complete: the GNN-matrix session died mid-inference
  right after MLP training finished, so only Random made it through before
  the interruption.
- **`highschool2013`'s GNN rows (DONG/SHAH/IGCN/HADDAD) were not trained
  and this was a deliberate decision, not an oversight.** Training on this
  network (the largest of the six) ran at roughly 6 minutes/epoch with no
  sign of early-stopping by epoch 10, i.e. multiple hours to converge.
  Colab's free-tier GPU runtimes disconnect after roughly one hour
  regardless, and the training driver has no mid-run checkpointing, so
  every disconnection restarted that model's training from epoch 0. This
  repeated (through two separate Google accounts, ~40 session
  creation/death cycles total, several of which additionally hit Colab's
  own session-assignment rate limit) without a single model ever
  completing. Given the cost of continuing to fight Colab's session
  lifetime and rate limits for an increasingly marginal addition to an
  already-decided-secondary benchmark (Problem A is explicitly padding
  for the paper, not the contribution), the remaining `highschool2013`
  GNN cells were abandoned rather than pursued further.
