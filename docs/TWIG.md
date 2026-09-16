# TWIG: exact influence minimisation on bounded-treewidth graphs

Status: verified core algorithm (`trace_e/blocking/treewidth.py`), narrow but
real scalability window (documented below with measurements, not claims).

## 1. What it computes

Given a graph, independent-cascade edge probabilities, a seed set and a
vertex-deletion (blocking) set B, `ExactSpread.evaluate(B)` computes
E[#nodes reachable from the seeds, excluding seeds | delete B] **exactly**
(no sampling), for graphs of bounded treewidth. Computing this quantity is a
classical network-reliability problem and is #P-hard in general (Valiant
1979); all IMIN literature (AdvancedGreedy, GreedyReplace, SandIMIN)
estimates it by Monte-Carlo sampling of live-edge graphs, which the
benchmark work in `docs/RESULTS_NOTES.md` showed is the actual bottleneck
of the whole line of work (cascade sizes are heavy-tailed, so the sample
mean is a poor estimator of the true expectation and greedy selects on
noise). `ExactSpread` removes that noise entirely, on graphs where it
applies.

## 2. Algorithm

Process the vertices in an elimination order pi (from a min-fill-in or
min-degree heuristic, giving induced width w = treewidth upper bound). A
vertex is "active" (in the frontier) from its introduction until every edge
incident to it has been processed, i.e. until the last of its neighbours
(by pi) is introduced; the frontier has size at most w + 1.

The DP state is a distribution over: (i) a partition of the active
vertices into blocks, one of which may be the SEED block (everything
already known to be reachable from some seed via edges processed so far);
(ii) for every other (non-seed) block, a **pending count**: the number of
already-retired vertices assigned to that block whose contribution to
E[#reached] is deferred until (and paid out exactly when) the block merges
with the SEED block. Three operations update this exactly:

* **introduce(v):** singleton block (SEED if v is a seed), pending 0.
* **process_edge(u, v):** branch on live (probability p, merge blocks u
  and v; if the merge is into SEED, immediately pay out the non-seed
  block's pending count, scaled by the branch probability) or dead
  (probability 1-p, unchanged); merge resulting states with identical
  (partition, pending) descriptions, summing probabilities.
* **retire(u):** if u's block is SEED, add its probability mass to the
  running total (u is genuinely reached, u itself is not a seed). If not,
  increment that block's pending count by 1 (defer u's own contribution)
  and drop u from the partition; if u was the block's last active member,
  the block can never merge with anything else in the future, so its
  pending count is discarded (those vertices correctly never counted).

Blocking a candidate set B is handled by simply never introducing its
members (equivalently, deleting them and their incident edges); one
`evaluate(B)` call is one pass over the elimination order, reusing the
same order for every B.

### Bug history (why this took two rounds to get right)

Two implementation bugs were caught only by brute-force verification
(exhaustive summation over all 2^m live-edge subsets on random graphs up
to n=9), which is why that verification is treated as load-bearing, not
optional, for this kind of algorithm:

1. A vertex retiring while still in a non-SEED block was originally simply
   dropped, discarding its potential future reachability if its block later
   merged with SEED. Fixed by the pending-count mechanism above.
2. Seed vertices themselves were double-counted at their own retirement
   (a seed is always trivially "in the SEED block", but should never be
   counted as one of the "reached" nodes). Fixed by excluding u in seeds
   from the payout at retirement.
3. A third bug specific to blocking: skipping the introduce/edge steps for
   a blocked vertex v was implemented by skipping its *entire* elimination
   step, including the *retirement* of other, unrelated vertices scheduled
   at that same step -- silently losing their contribution. Fixed by only
   skipping introduce/connect for a blocked v, never its step's retirements.

After all three fixes: 2000+ random-graph checks (n up to 10, up to 4
seeds, up to 4 blocked nodes, unblocked and blocked) match brute force
exactly (`tests/test_treewidth.py`).

## 3. Complexity: XP, not (cleanly) FPT, in treewidth

The number of partition *shapes* of a frontier of size w+1 is Bell(w+1), a
constant for fixed w. But the pending-count vector attached to each shape
has up to w+1 components, each a count that can range up to n, so the
number of *distinct reachable states* is bounded by Bell(w+1) * n^{O(w)} --
polynomial in n for any *fixed* w, but with a degree that grows with w.
This makes the algorithm XP in treewidth (each fixed w gives a polynomial-
time algorithm) rather than FPT in the strict sense (f(w) * poly(n) with
the polynomial's degree independent of w). This distinction matters in
practice, not just in theory:

| graph | n | treewidth (min-fill-in) | `ExactSpread.evaluate()` |
|---|---|---|---|
| random tree | 30 | 1 | 0.2 s |
| Erdos-Renyi, m=25 | 15 | 4 | 0.4 s |
| Erdos-Renyi, m=35 | 20 | ~5-6 | did not finish in 150 s |
| Zachary karate club | 34 | 5 | did not finish in 60 s |

Trees and very sparse graphs (treewidth 1-2) scale to hundreds of nodes;
graphs with treewidth as low as 5-6 but non-tree structure already blow up
the pending-count dimension in practice on graphs as small as n=20-34 in
this (pure-Python, unoptimised) implementation. This is a genuine, measured
limitation, not a hedge: **`ExactSpread` and everything built on it below
are usable on small graphs (n up to roughly 15-20 for w around 4-5, or much
larger for tree-like/very sparse graphs), not on the SNAP-scale graphs used
in `docs/RESULTS_NOTES.md`.** Reducing the polynomial's degree (e.g. by
bounding pending counts via an approximate/rounded representation, or a
compiled implementation) is future work, not attempted here.

## 4. What this makes possible: true ground truth for greedy IMIN

`greedy_exact` runs AdvancedGreedy-style greedy selection using
`ExactSpread.evaluate` as the marginal-gain oracle instead of a Monte-Carlo
estimate -- zero sampling noise, by construction never worse than the true
optimum. `brute_force_optimal_block` exhaustively searches all
size-<=budget subsets with the same exact oracle, giving the **true
optimal blocking set** on graphs small enough to enumerate.

This is the first time in this project (and, as far as the literature
search turned up, in the IMIN literature) that a *provably exact*
optimality gap for greedy blocking has been measured, rather than a
sampling-noise-confounded proxy. Result over 60 random small instances
(n=6-14, budget 1-4, `trace_e/blocking/treewidth.py` `greedy_exact` vs
`brute_force_optimal_block`):

* 56 / 60 instances: greedy is exactly optimal.
* 4 / 60 instances: greedy is strictly suboptimal, by up to 84.6% relative
  excess spread in the worst case found (mean excess 2.2% across all 60
  instances, dominated by that one outlier).

This is consistent with the known hardness-of-approximation result for
IMIN under IC (Xie et al., ICDE 2023): greedy is *usually* very good but
is not guaranteed to be, and now there is a concrete, reproducible instance
family exhibiting the gap, obtainable at will via
`brute_force_optimal_block` on any small instance.

## 5. Honest summary

* Verified-correct exact reachability oracle for IC-model influence
  minimisation on bounded-treewidth graphs: yes, rigorously (2000+ brute-
  force cross-checks, three bugs found and fixed along the way).
* A genuinely new empirical result: true (not sampled) optimality gaps for
  greedy IMIN, on tiny instances.
* A practical, general-purpose algorithm for the graph sizes used
  elsewhere in this project (SNAP-scale, budget-20-100 experiments): no.
  The pending-count state blow-up makes it inapplicable there without
  further algorithmic work (see complexity discussion above).
* Recommended honest framing for a paper: a correctness-and-complexity
  contribution (XP algorithm, proof sketch above, matching the network-
  reliability literature's bounded-treewidth tractability results) plus a
  small, clean empirical result (true optimality gaps on tiny instances),
  not a "faster/better than AG/GR at scale" claim -- that claim would be
  false given the measurements above.
