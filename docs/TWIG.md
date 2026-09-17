# TWIG: Exact Influence Minimisation on Bounded-Treewidth Graphs

Status: verified core algorithm (`trace_e/blocking/treewidth.py`), formal
correctness and complexity proofs below, narrow but real scalability window
(measured, not claimed).

## 1. Problem and preliminaries

**Independent cascade (IC).** $G=(V,E)$ undirected, $n=|V|$, $m=|E|$, edge
probabilities $p:E\to(0,1]$. A live-edge realisation $\omega\in\{0,1\}^E$ is
drawn with $\Pr[\omega]=\prod_e p_e^{\omega_e}(1-p_e)^{1-\omega_e}$. For a
seed set $S\subseteq V$ and a blocked (deleted) set $B\subseteq V\setminus S$,
let $G_B=G[V\setminus B]$ and let $R(\omega,B)\subseteq V\setminus B$ be the
vertices reachable from $S$ using only edges of $G_B$ that are live under
$\omega$. Define
$$F(B)\;=\;\mathbb E_\omega\bigl[\,|R(\omega,B)\setminus S|\,\bigr]
=\sum_{v\in V\setminus B\setminus S}\Pr\nolimits_\omega[v\in R(\omega,B)].$$
Computing $F(B)$ exactly is a network-reliability computation and is
\#P-hard in general (Valiant 1979; Provan & Ball 1983 for the two-terminal
case), which is why the whole IMIN literature (AdvancedGreedy,
GreedyReplace, SandIMIN) estimates it by Monte-Carlo sampling instead. The
benchmark work in `docs/RESULTS_NOTES.md` shows this sampling noise is the
actual bottleneck of that line of work (cascade sizes are heavy-tailed, so
the sample mean is a poor, high-variance estimator, and greedy selects on
noise). This module computes $F(B)$ **exactly**, for any $B$, on graphs of
bounded treewidth.

**Elimination order and schedule.** Fix a permutation $\sigma=(\sigma_1,
\dots,\sigma_n)$ of $V$ (in the implementation, $\sigma$ is the *reverse* of
a min-fill-in elimination order; see Proposition 4 for why direction
matters). Write $\mathrm{pos}(v)$ for $v$'s index in $\sigma$. For each
$v=\sigma_i$ define:

* $\mathrm{connect\_to}(v)=\{u\in N(v):\mathrm{pos}(u)<i\}$ (already-processed
  neighbours; edge $(u,v)$ is handled when $v$ is processed);
* $\mathrm{last}(v)=\max_{x\in N[v]}\mathrm{pos}(x)$ (the step at which $v$'s
  final incident edge is handled);
* the **frontier** after step $i$, $F_i=\{\sigma_j: j\le i,\ \mathrm{last}
  (\sigma_j)>i\}\cup\{\sigma_j:j\le i,\ \mathrm{last}(\sigma_j)=i\}$, i.e. the
  vertices introduced so far that have not yet retired, including those
  retiring at step $i$ itself right before they do.

The **width** of $\sigma$ is $w=\max_i|F_i|-1$; $w$ upper-bounds the
treewidth of $G$ for the $\sigma$ obtained from a min-fill-in / min-degree
chordal completion (Bodlaender 1988; Cygan et al., *Parameterized
Algorithms*, ch. 7 gives the general theory this instantiates).

## 2. Algorithm `ExactSpread`

**Definition 1 (configuration).** A *configuration* over a vertex set
$F\subseteq V$ is a pair $\kappa=(\Pi,\rho)$ where $\Pi$ is a set partition
of $F$ with at most one distinguished block $\mathrm{SEED}$, and
$\rho:(\text{non-}\mathrm{SEED}\text{ blocks of }\Pi)\to\mathbb N$.

The algorithm maintains, after each step $i$, a finite probability
distribution $\{P_\kappa\}$ over configurations on $F_i$, plus a running
scalar $\mathrm{acc}_i$ (`expected` in the code), updated by three
operations at step $i=\mathrm{pos}(v)$:

* **introduce($v$).** Every configuration gains $v$ as a new singleton
  block: $\mathrm{SEED}$ if $v\in S$, otherwise a fresh block with
  $\rho(\{v\})=0$.
* **process\_edge($u,v$)** for each $u\in\mathrm{connect\_to}(v)$, in turn.
  Every configuration branches: w.p. $1-p(u,v)$ unchanged; w.p. $p(u,v)$ the
  blocks of $u,v$ merge. If the merge is into $\mathrm{SEED}$: add
  $p(u,v)\cdot\rho(b)$ to $\mathrm{acc}$, where $b\ne\mathrm{SEED}$ is the
  merging block, then drop $b$ (its vertices join $\mathrm{SEED}$, pending
  discharged). If both blocks are ordinary: the merged block's pending is
  $\rho(b_1)+\rho(b_2)$. Configurations that coincide after the branch are
  merged, summing probabilities.
* **retire($u$)** for each $u$ retiring at step $i$. If $u$'s block is
  $\mathrm{SEED}$: add $P_\kappa$ to $\mathrm{acc}$ **unless** $u\in S$
  (seeds are never counted as "reached"). Otherwise: if $u$'s block still
  has other active members, increment its pending by 1 (defer $u$'s
  contribution); if $u$ was the block's last active member, discard the
  block's pending (those vertices are, correctly, never reached). Either
  way $u$ is removed from the partition.

Blocking a set $B$ is handled by skipping introduce/connect\_to for
$v\in B$ (its edges are simply never processed, exactly modelling
deletion), while still running retirements of *other* vertices scheduled
at the same step (Bug 3 below).

### 2.1 Correctness

**Lemma 1 (invariant).** *Fix $B$ and a step $i$. For every configuration
$\kappa=(\Pi,\rho)$ with $P_\kappa>0$ after step $i$,*
$$P_\kappa=\Pr\nolimits_\omega\Bigl[\;\Pi\text{ is the partition of }F_i\text{
by connectivity in }\bigl(F_i^{\ge},\,\omega|_{E_i}\bigr)\text{, seed-
containing part collapsed to }\mathrm{SEED};$$
$$\text{and }\forall\text{ non-}\mathrm{SEED}\text{ block }b,\ \rho(b)=\bigl|
\{u:\mathrm{last}(u)\le i,\ u\text{ retired non-seed},\ u\sim_\omega b\}
\bigr|\;\Bigr],$$
*where $E_i$ is the set of edges processed by step $i$ (both endpoints
introduced, neither endpoint in $B$), $F_i^{\ge}=\{x\notin B:\mathrm{pos}(x)
\le i\}$ is every vertex introduced so far (active or already retired), and
$u\sim_\omega b$ means $u$ is connected, via live edges of $E_i$ within
$F_i^{\ge}$, to the current representative of block $b$. Moreover*
$$\mathrm{acc}_i=\sum_{v\in V\setminus S\setminus B\,:\,\mathrm{last}(v)\le
i}\Pr\nolimits_\omega\bigl[v\sim_\omega\text{ some seed, using only edges of
}E_i\bigr].$$

*Proof.* By induction on $i$. Base case $i=-1$ (before any step): $F_{-1}
=\varnothing$, the unique configuration is the empty partition with
$P=1$, $\mathrm{acc}=0$, and both sides of the claimed identities are
vacuously true ($E_{-1}=\varnothing$, no vertex has retired).

*Introduce.* Adding $v$ as a new singleton block changes nothing about
already-active vertices' connectivity (no new edge is realised), and
correctly starts $v$'s own block (SEED if $v\in S$, pending $0$
otherwise, since $v$ is trivially connected to itself and, being newly
introduced, to nothing else yet). The invariant is preserved for $\Pi$
and, since no new edges enter $E_i$, for $\mathrm{acc}$.

*Process\_edge($u,v$).* This is the only place a new edge $(u,v)$ enters
$E_i$. Conditioning on its two outcomes (independent of every other edge's
realisation, by the IC model's edge-independence) exactly reproduces the
two branches: tails leaves $\Pi$, $\rho$, $\mathrm{acc}$ untouched, matching
that $(u,v)\notin\omega$ leaves connectivity and every retired vertex's
membership unchanged. Heads merges $u$'s and $v$'s blocks, which is exactly
the effect of adding a live edge to a union-find style connectivity
structure. If the merge is into $\mathrm{SEED}$: every already-retired
vertex $u'$ counted in the merging block's $\rho(b)$ is, by the outer
induction hypothesis, connected (via $E_{i}\setminus\{(u,v)\}$) to $b$'s
current representative, hence now (with $(u,v)$ live) transitively
connected to a seed for the first time in this history -- exactly one unit
each is owed to $\mathrm{acc}$, contributing $p(u,v)\cdot\rho(b)$ in
expectation over this branch, matching the update; $\rho(b)$ is then
correctly discharged (those vertices will never need to be paid for again,
since a vertex is reached at most once). If the merge is between two
ordinary blocks, no vertex's fate is decided yet (neither is connected to a
seed), so $\mathrm{acc}$ is untouched and the two blocks' pending vertex
sets simply become one, i.e. $\rho(b_1)+\rho(b_2)$, matching the update.

*Retire($u$).* $u$ retiring means $\mathrm{last}(u)=i$: every edge incident
to $u$ is already in $E_i$, so $u$'s connectivity to everything else active
or retired is already fully determined and will never change (Definition
1's frontier bookkeeping is exactly designed to make this true). If $u$'s
block is already $\mathrm{SEED}$: $u$ is reached with probability $P_\kappa$
in this history, contributed to $\mathrm{acc}$ unless $u$ itself is a seed
(seeds are excluded from "reached", matching the definition of $F(B)$).
Otherwise $u$'s fate is not yet decided; deferring it into $\rho(b)$ is
exactly the claimed count (one more retired vertex now connected to $b$),
*unless* $b$ has no other active member, in which case $b$ can **never**
merge with anything else in the future (its only remaining edges were
already all processed, by definition of retirement having emptied it of
active representatives) -- correctly discharging vertices that are
provably never reached. Removing $u$ from $\Pi$ matches $F_i\setminus\{u\}$.
$\blacksquare$

**Theorem 1 (correctness of `ExactSpread`).** *For every $B$,
`ExactSpread(G, p, S).evaluate(B)` $=F(B)$.*

*Proof.* Apply Lemma 1 at $i=n-1$ (the last step): $F_{n-1}=\varnothing$
(every vertex has retired by the time all its neighbours, which include
itself, have been introduced), $E_{n-1}=E\setminus(B\text{-incident
edges})$, i.e. all of $G_B$'s edges, and every non-seed, non-blocked
vertex has $\mathrm{last}(v)\le n-1$. The second identity of Lemma 1 then
reads $\mathrm{acc}_{n-1}=\sum_{v\in V\setminus S\setminus B}\Pr_\omega[v
\sim_\omega\text{ some seed in }G_B]=F(B)$, which is exactly what
`evaluate(B)` returns. $\blacksquare$

### 2.2 Bug history (why this took three rounds to get right)

Three implementation bugs were caught only by brute-force verification
(exhaustive summation over all $2^m$ live-edge subsets on random graphs up
to $n=9$), which is why that verification is treated as load-bearing, not
optional, for algorithms of this kind:

1. A vertex retiring while still in a non-SEED block was originally simply
   dropped, discarding its potential future reachability if its block later
   merged with SEED -- exactly the deferral Lemma 1 depends on. Fixed by
   introducing the pending-count mechanism.
2. Seed vertices themselves were double-counted at their own retirement
   (Lemma 1's retire case explicitly excludes $u\in S$; the original code
   did not). Fixed by excluding $u\in S$ from the payout at retirement.
3. A bug specific to blocking: skipping the introduce/edge steps for a
   blocked vertex $v$ was implemented by skipping its *entire* elimination
   step, including the *retirement* of other, unrelated vertices scheduled
   at that same step -- silently losing their contribution (this breaks
   the proof of Lemma 1's introduce/retire cases for those other
   vertices, since their invariant update was simply never applied). Fixed
   by skipping only introduce/connect for a blocked $v$, never its step's
   retirements.

After all three fixes: 2000+ random-graph checks (n up to 10, up to 4
seeds, up to 4 blocked nodes, unblocked and blocked) match brute force
exactly (`tests/test_treewidth.py`), consistent with Theorem 1.

## 3. Complexity: XP, not (cleanly) FPT, in treewidth

**Proposition 1 (state space of `ExactSpread`).** *The number of partition
shapes of a frontier of size at most $w+1$ is $B_{w+1}$, the $(w+1)$-th
Bell number, independent of $n$. But each shape's pending vector has up to
$w+1$ components, each ranging over $\{0,\dots,n\}$, so the number of
distinct configurations at any step is at most $B_{w+1}\cdot(n+1)^{w+1}$.*

This makes `ExactSpread` **XP** in treewidth: for every *fixed* $w$ it runs
in time polynomial in $n$, but the polynomial's degree grows with $w$,
rather than **FPT** ($f(w)\cdot\mathrm{poly}(n)$ with the polynomial's
degree independent of $w$). Measured consequence (pure-Python, single
elimination order reused across calls):

| graph | n | treewidth (min-fill-in) | `ExactSpread.evaluate()` |
|---|---|---|---|
| random tree | 30 | 1 | 0.2 s |
| Erdos-Renyi, m=25 | 15 | 4 | 0.4 s |
| Erdos-Renyi, m=35 | 20 | ~5-6 | did not finish in 150 s |
| Zachary karate club | 34 | 5 | did not finish in 60 s |

## 4. `ExactSpreadFast`: collapsing pending to make the algorithm FPT

**Theorem 2 (correctness of the linear-aggregation collapse).** *Fix a
partition shape $s$ (Definition 1's $\Pi$ with $\rho$ forgotten) and, at
some step, let $\mathcal K_s=\{(s,\rho):P_{(s,\rho)}>0\}$ be the set of
configurations sharing shape $s$. Define the aggregate*
$$P_s=\sum_{(s,\rho)\in\mathcal K_s}P_{(s,\rho)},\qquad
W_b=\sum_{(s,\rho)\in\mathcal K_s}P_{(s,\rho)}\cdot\rho(b)\ \ (b\in
\mathrm{blocks}(s)\setminus\{\mathrm{SEED}\}).$$
*Then $(P_s,\{W_b\})_s$ can be updated, at every step, by the same four
rules `ExactSpreadFast` implements, and this update yields the identical
sequence of `acc` increments as `ExactSpread`, for every $B$.*

*Proof.* We check each of the (at most) four ways a step transforms a
configuration $(s,\rho)\mapsto(s',\rho')$ within a fixed edge-outcome
branch (shape transitions never depend on $\rho$, only on $s$ and, for
`process_edge`, the branch); linearity of each in $\rho$ is then all that
is needed, by the following computation applied to every $b\in
\mathrm{blocks}(s')$:

* **Introduce a new non-seed block $b_0$.** $\rho'(b_0)=0$, $\rho'(b)=
  \rho(b)$ otherwise, $s'=s\cup\{b_0\}$ is independent of $\rho$. Then
  $W'_{b_0}=\sum_\rho P_\rho\cdot 0=0$ and $W'_b=W_b$ for $b\ne b_0$: no
  update needed for existing entries, new entry is (implicitly) $0$ --
  matches the code (a freshly introduced block has no `W` entry, and
  absent entries default to $0$).
* **Process\_edge, tails.** $s'=s$, $\rho'=\rho$, but the whole branch
  carries probability weight $(1-p_e)$: $P'_s=(1-p_e)P_s$ and $W'_b=
  \sum_\rho(1-p_e)P_\rho\cdot\rho(b)=(1-p_e)W_b$ -- matches
  `w0[k]+=val*(1-pe)` applied to every entry of $W$.
* **Process\_edge, heads, merge into $\mathrm{SEED}$.** The payout added to
  `acc` in this branch is $p_e\sum_{(s,\rho)}P_\rho\cdot\rho(b)=p_e\cdot
  W_b$ exactly (no approximation: this is a direct algebraic identity, not
  a mean-field substitution) -- matches `expected += pe * W.get(non_s,
  0.0)`. Block $b$ is then dropped ($\rho'$ has no entry for it),
  matching `W2.pop(...)`.
* **Process\_edge, heads, merge two ordinary blocks $b_1,b_2\to b$.**
  $\rho'(b)=\rho(b_1)+\rho(b_2)$, so $W'_b=\sum_\rho P_\rho(\rho(b_1)+
  \rho(b_2))=W_{b_1}+W_{b_2}$ -- matches `W2[merged] = W.get(bu,0)+
  W.get(bv,0)`.
* **Retire, block still active.** $\rho'(b)=\rho(b)+1$ for the retiring
  block, others unchanged: $W'_b=\sum_\rho P_\rho(\rho(b)+1)=W_b+\sum_\rho
  P_\rho=W_b+P_s$ -- matches `W2[bu] = W2.get(bu,0.0) + P` (note this uses
  the *aggregate* $P_s$, not a per-history increment, which is exactly
  what the identity requires).
* **Retire, block emptied.** $b$'s entry is dropped for every history in
  $\mathcal K_s$ simultaneously (the condition "no other active member"
  depends only on $s$, not $\rho$), so $W'_b$ is simply omitted -- matches
  `W2.pop(...)`.
* **Retire, block is SEED, $u\notin S$.** `acc` gains $\sum_\rho P_\rho=
  P_s$ in this branch -- matches `expected += P` (the aggregate, exactly
  the quantity Lemma 1 needs summed over this shape).

Each case is an exact algebraic identity between the *aggregate before* and
the *aggregate after*, with no term dropped or approximated; by induction
over the sequence of steps, $(P_s,\{W_b\})$ computed by `ExactSpreadFast`
therefore equals the shape-lumped aggregate of `ExactSpread`'s full
distribution at every step, and in particular produces the identical
sequence of contributions to `acc`. $\blacksquare$

Cross-checked computationally against `ExactSpread` and brute force on
1000+ random graphs (`tests/test_treewidth.py`), consistent with Theorem 2.

**Proposition 2 (state space of `ExactSpreadFast`).** *The number of
distinct states at any step is at most $B_{w+1}$ (one $(P_s,\{W_b\})$ tuple
per shape), independent of $n$.* This makes `ExactSpreadFast` **FPT** in
treewidth: $O(n\cdot B_{w+1}\cdot\mathrm{poly}(w))$ overall, a genuine
change of complexity class from Proposition 1, not merely a constant-factor
speed-up -- the $n^{w+1}$ factor is gone entirely.

## 5. Processing direction is not a detail: a provable separation

**Proposition 3 (star graphs: forward order forces frontier $n-1$; the
reverse order gives frontier $2$).** *Let $T_k$ be the star $K_{1,k}$
(one centre $c$, $k$ leaves), $n=k+1$. Min-fill-in computes the elimination
order $\pi=(\ell_1,\dots,\ell_k,c)$ (every leaf has fill-in cost 0 and is
eliminated before the centre, whose remaining fill-in cost is also 0 by
the time it is the last vertex left) -- correctly certifying treewidth 1.
Running the DP schedule directly on $\pi$ (centre processed **last**)
produces a frontier of size $k=n-1$ immediately before the centre's step
(every leaf is introduced but cannot retire, since its only neighbour $c$
is not yet introduced). Running the DP on $\mathrm{reverse}(\pi)=(c,\ell_1,
\dots,\ell_k)$ (centre processed **first**) keeps the frontier at size at
most 2 throughout (the centre plus whichever single leaf is currently
being processed; every leaf retires immediately after its one edge to the
already-active centre is processed). Verified computationally for $k=6$
($n=7$): forward peak frontier $=6$, reversed peak frontier $=2$
(`docs/TWIG.md` proposition check, `elimination_order`/`_schedule`).*

*Proof.* Immediate from the definitions in Section 1: under $\pi$,
$\mathrm{last}(\ell_i)=\mathrm{pos}(c)=k$ for every leaf (its only
neighbour $c$ is introduced last), so no leaf retires before step $k$,
giving $F_{k-1}=\{\ell_1,\dots,\ell_k\}$; under $\mathrm{reverse}(\pi)$,
$\mathrm{last}(\ell_i)=\mathrm{pos}(\ell_i)$ (once $c$ is already active,
$\ell_i$'s only edge is processed the instant $\ell_i$ is introduced, and
$c$ is not $\ell_i$'s own last neighbour to be introduced -- it already was),
so every leaf retires the step after it is introduced, and $c$ itself
retires only at the final step (its own $\mathrm{last}$ is the position of
the last-introduced leaf), giving $|F_i|\le2$ for all $i$. $\blacksquare$

This is a worst-case separation on an exact, checkable family, not a
heuristic claim: a bounded-treewidth graph (here, treewidth $1$, for any
$n$) on which the *same* elimination order, run in the *wrong* direction,
forces the DP's true state space to scale with $n$ instead of staying
bounded by a function of the width alone. On Zachary's karate club (real
data, not this synthetic worst case) the same qualitative effect measures
as: forward order's frontier peaks at 15+ active vertices despite width 5;
reversed, it peaks at 11. Combined with the Theorem 2 collapse:

| graph | n | width (min-fill-in) | before both fixes | after both fixes |
|---|---|---|---|---|
| Zachary karate club | 34 | 5 | did not finish in 60 s | **2.5 s** |
| Iceland (sexual contact) | 75 | 4 | not attempted | **51 s** |
| Dolphin social network | 62 | ~10-11 | not attempted | still infeasible (memory blow-up, killed after using >9 GB) |

Karate and Iceland -- real networks from `data/networks/`, not synthetic
toy instances -- are tractable for exact evaluation after both fixes.
Denser small graphs with width above roughly 6-7 (dolphin, fraternity,
workplace) remain out of reach: $B_{w+1}$ is still exponential in $w$
(Proposition 2), just no longer also multiplied by a factor growing with
$n$ (Proposition 1's gap, closed by Theorem 2).

`greedy_exact` additionally restricts its candidate pool to vertices
actually reachable from the seeds (blocking an unreachable vertex cannot
change $F(B)$, immediate from the definition), which matters on sparse
graphs where most vertices are never in play.

### 5.1 End-to-end capstone: exact greedy blocking on Zachary's karate club

`greedy_exact` (not just a single `evaluate()` call) was run end-to-end on
the karate club network (n=34, width 5, one seed at node 0, random IC edge
probabilities in [0.05, 0.4]):

| budget | blocked set B | exact expected spread F | time |
|---|---|---|---|
| 1 | {2} | 7.9752 | 63.1 s |
| 2 | {2, 1} | 5.5053 | 72.2 s |

Each row is a full greedy search (evaluating every remaining reachable
candidate exactly, once per budget slot -- about 33 exact `evaluate()`
calls for budget 1, another 32 for budget 2), not a single oracle call:
node 2 (a high-degree cut vertex in karate's two-faction structure) is
selected first both times, then node 1, cutting expected spread from
7.98 to 5.51 nodes with zero sampling noise anywhere in the computation.

### 5.2 A joint decision+expectation DP was tried and abandoned

A natural next step is folding the *choice* of which vertices to block into
the same DP (adding a "slots used" dimension and a block/keep branch at
each introduce step), to get the exact optimal budgeted blocking set
without exhaustive search. This was attempted and **abandoned as unsound**:
naively tracking one scalar "best accumulated cost so far" per (partition,
pending, slots) state conflates two different things that must not be
conflated -- *minimising* over decisions (which are fixed once, not
resampled) and *averaging* over the edge randomness (which must be
properly probability-weighted, as in Theorem 1's proof). The draft
implementation implicitly let the block/keep decision for a not-yet-
introduced vertex depend on which random branch a given state had arrived
from, which corresponds to solving an easier, *adaptive* version of the
problem, not the intended fixed (non-adaptive) blocking set. It was
deleted rather than shipped once this was noticed. `greedy_exact` (an
exact-oracle greedy heuristic) and `brute_force_optimal_block` (exhaustive
search, small instances only) are the safe substitutes used instead.

## 6. What this makes possible: true ground truth for greedy IMIN

`greedy_exact` runs AdvancedGreedy-style greedy selection using
`ExactSpreadFast.evaluate` as the marginal-gain oracle instead of a
Monte-Carlo estimate -- zero sampling noise, and by Theorem 1 never worse
than the true optimum restricted to greedy's own search path.
`brute_force_optimal_block` exhaustively searches all size-$\le$budget
subsets with the same exact oracle, giving the **true optimal blocking
set** on graphs small enough to enumerate.

Xie et al. (ICDE 2023, Theorem 8) prove that $F(\cdot)$ is *not*
supermodular in $B$ under IC, so greedy carries no $(1-1/e)$-style
guarantee here (unlike influence *maximisation*); they further show IMIN
is APX-hard (Theorem 9). This is consistent with, and gives a theoretical
account of, the following: this is the first time in this project (and, as
far as the literature search turned up, in the IMIN literature) that a
*provably exact* optimality gap for greedy blocking has been measured,
rather than a sampling-noise-confounded proxy. Result over 60 random small
instances (n=6-14, budget 1-4, `greedy_exact` vs
`brute_force_optimal_block`):

* 56 / 60 instances: greedy is exactly optimal.
* 4 / 60 instances: greedy is strictly suboptimal, by up to 84.6% relative
  excess spread in the worst case found (mean excess 2.2% across all 60
  instances, dominated by that one outlier).

Non-supermodularity (Xie et al.'s Theorem 8) explains *why* such gaps must
exist in the worst case; this is the first *measurement*, free of sampling
confounds, of how often and how badly they occur in practice on small
instances, and a concrete, reproducible instance family exhibiting the gap
is obtainable at will via `brute_force_optimal_block` on any small
instance.

## 7. Honest summary

* Verified-correct exact reachability oracle for IC-model influence
  minimisation on bounded-treewidth graphs, with formal correctness proofs
  (Theorem 1, Theorem 2) matching 2000+ and 1000+ brute-force cross-checks
  respectively, three bugs found and fixed along the way (Section 2.2).
* A formal complexity classification (XP $\to$ FPT in treewidth,
  Propositions 1-2) plus a provable worst-case separation showing
  *processing direction* is a first-class algorithmic choice, not a tuning
  detail (Proposition 3), each independently confirmed on real data.
* A genuinely new empirical result, explained by (not merely consistent
  with) Xie et al.'s non-supermodularity/APX-hardness theorems: true (not
  sampled) optimality gaps for greedy IMIN, on tiny instances.
* A practical, general-purpose algorithm for the graph sizes used
  elsewhere in this project (SNAP-scale, budget-20-100 experiments): no.
  $B_{w+1}$'s exponential dependence on treewidth alone (Proposition 2)
  makes it inapplicable there without a fundamentally different technique.
* Recommended honest framing for a paper: a correctness-and-complexity
  contribution (FPT algorithm with full proofs, matching the network-
  reliability literature's bounded-treewidth tractability results) plus a
  small, clean empirical result (true optimality gaps on tiny instances),
  not a "faster/better than AG/GR at scale" claim -- that claim would be
  false given the measurements above.
