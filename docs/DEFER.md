# DEFER: Adaptive Influence Minimisation by Deferred Frontier Blocking

Status: algorithm specification v1 (2026-09-16). Implementation:
`trace_e/sequential/containers.py` (`AdaptiveFrontier`),
`trace_e/blocking/imin.py` (baselines AG, GR, LSBM on a shared sample set),
benchmark `trace_e/eval/run_adaptive.py`.

## 1. Problem

**IMIN (one-shot).** Directed graph $G=(V,E)$ with independent-cascade
probabilities $p(u,v)$, seed set $S$, budget $b$. Choose $B\subseteq V\setminus S$,
$|B|\le b$, to minimise $\sigma(S;B)=\mathbb E[\,|\text{nodes activated from }S\text{ in }G[V\setminus B]|\,]$.
NP-hard and hard to approximate (Xie et al. 2023). State of the art:
AdvancedGreedy (AG), GreedyReplace (GR) (Xie et al. 2023/2024) and the
Sandwich algorithms of Wang et al. (2024) whose lower bound LSBM is greedy
maximum coverage of dominator chains.

**Adaptive IMIN (this work).** The cascade unfolds in rounds. In round $t$
the controller observes the newly activated set $N_t$ (full-adoption
feedback: activations are observed, live edges are not), may block any
inactive unblocked nodes, and the total number of blocked nodes over all
rounds is at most $b$. Minimise the expected final spread. A one-shot
solution is the special case that spends everything in round $0$.

## 2. Algorithm DEFER

Input: $G$, $p$, seeds $S$, budget $b$, sample size $\theta$, a one-shot
planner $\mathcal P$ (any IMIN algorithm; default: dominator greedy with
isolation moves, alternatively GR).

For each round $t=0,1,2,\dots$ while the cascade is alive and budget remains:

1. **Frontier.** Let $F_t=N_t$ be the nodes activated in the last round
   (only they can activate new nodes under IC), $A_t$ the active set,
   $B_t$ the blocked set, $b_t=b-|B_t|$ the remaining budget.
2. **Sample.** Draw $\theta$ live-edge graphs, keeping only edges out of
   nodes reachable from $F_t$ avoiding $A_t\cup B_t$ (forward BFS per
   sample).
3. **Plan.** Run $\mathcal P$ on the instance "seeds $=F_t$, forbidden
   $=A_t\cup B_t$, budget $b_t$" using the samples: with the dominator
   planner, build one dominator tree per sample rooted at a super-source
   attached to $F_t$; the saving of blocking $v$ is its average subtree
   size, the saving of isolating a frontier node $u$ (blocking all of its
   exposed out-neighbours) is its average subtree size minus one; pick moves
   greedily by saving per budget unit, recomputing the trees only on
   samples in which a chosen node was reachable. Output a plan
   $P_t$, $|P_t|\le b_t$.
4. **Commit only what is exposed, and only if waiting is dearer.** Let
   $X_t$ be the inactive unblocked out-neighbours of $F_t$, $q_v$ the
   probability that $v\in X_t$ is activated in the coming round
   ($q_v=1-\prod_{u\in F_t}(1-p(u,v))$), $c_v$ the number of inactive
   unblocked out-neighbours of $v$, and $\lambda_t$ the shadow price of a
   budget unit (the smallest marginal saving among the picks of $P_t$; $0$
   if the plan did not exhaust the budget). Block
   $$C_t=\{v\in P_t\cap X_t:\ q_v\,(c_v+1/\lambda_t)\ \ge\ 1\}$$
   now (*push-down rule*); every other planned node is left uncommitted and
   the budget is carried over.
5. **Observe** $N_{t+1}$ and go to 1.

The push-down rule compares two ways of protecting what $v$ protects:
spending one unit on $v$ now, or waiting one round and, if $v$ does
activate (probability $q_v$), protecting its children instead at cost
$c_v$ while losing $v$ itself (one node). Waiting costs $q_v c_v$ budget
units and $q_v$ nodes in expectation; a budget unit is worth $\lambda_t$
nodes; so waiting is preferable iff $q_v c_v+q_v/\lambda_t<1$. Without the
rule ($C_t=P_t\cap X_t$) DEFER is exactly lazy commitment of the planner's
solution (Lemma 1); with it DEFER can exploit the randomness of the next
round, which is what the adaptivity gap of Theorem 2 is made of.

When the planner is deterministic given the samples and no new information
arrives, DEFER reduces to the one-shot planner's solution executed lazily
(Lemma 1). The two ablations used in the experiments are COMMIT (same loop
but block all of $P_t$ in step 4) and the one-shot planners themselves.

## 3. Analysis

**Lemma 1 (deferral is free).** Fix a live-edge realisation, the seeds and
any set $P$ with $|P|\le b$. Let $\pi_P$ be the policy that, in every round,
blocks the not-yet-blocked nodes of $P$ that are exposed to the current
frontier. Then $\pi_P$ produces exactly the same final active set as
blocking $P$ at time $0$, and blocks at most as many nodes.

*Proof.* On a fixed realisation a node $v$ becomes active in the first
round in which a frontier node has a live edge into it, i.e. the round after
it is first exposed. $\pi_P$ blocks $v\in P$ in that round, before its trial,
so $v$ never activates; every node outside $P$ faces the same set of blocked
nodes at each of its trials under both policies because a node of $P$ that is
never exposed never takes part in any trial. Hence the activation sequences
coincide. Nodes of $P$ that are never exposed are never blocked. $\square$

**Theorem 1 (DEFER weakly dominates its planner).** Let $P_0$ be the plan
of $\mathcal P$ at round $0$. If in every later round the planner returns a
plan whose sample-average saving is at least that of the leftover
$P_{t-1}\setminus C_{t-1}$ (guaranteed when the leftover is offered to the
planner as a candidate solution), then the expected final spread of DEFER is
at most that of the one-shot solution $P_0$, up to the sampling error of the
$\theta$ samples.

*Proof.* By Lemma 1 the policy that keeps $P_0$ and commits lazily has the
same spread as blocking $P_0$ at time 0. DEFER differs from it only by
replacing, at each round, the uncommitted leftover by a plan that is no worse
on the samples for the remaining budget; the committed prefix is identical.
Induction over rounds gives the claim on the samples; concentration of the
sample average (Xie et al. 2024, Theorem 2, applied per candidate plan) gives
the expectation statement. $\square$

**Lemma 2 (push-down is the myopic optimum).** Consider an exposed planned
node $v$ whose children (its inactive out-neighbours) jointly protect the
same set as $v$ minus $v$ itself, assume the children are exposed only
through $v$, and value a budget unit at $\lambda$ nodes. Among the two
policies "block $v$ now" and "block $v$'s children if and only if $v$
activates", the second has smaller expected loss (nodes lost plus
$\lambda$ times budget spent) iff $q_v(c_v+1/\lambda)<1$.

*Proof.* Blocking now costs $\lambda$ and loses nothing of $v$'s set.
Waiting loses $v$ (one node) and spends $c_v$ units with probability $q_v$,
and nothing otherwise, while the protected set is the same in both branches
by assumption: expected loss $q_v+\lambda q_vc_v$. Compare. $\square$

**Theorem 2 (adaptivity gap is unbounded).** For every $\Delta\ge2$ and
$p\in(0,1)$ there is an instance with maximum out-degree $\Delta$ on which
the best one-shot solution with budget $1$ has expected spread at least
$\bigl(1-(1-p)^{\Delta}\bigr)/p$ times that of DEFER with budget $1$
(up to an additive constant), i.e. the adaptivity gap of IMIN is at least
$(1-(1-p)^\Delta)/p$, which tends to $\Delta$ as $p\to0$.

*Proof.* Seed $s$ has out-neighbours $a_1,\dots,a_\Delta$ with
$p(s,a_i)=p$; each $a_i$ starts a private directed path of length $L$ with
edge probability $1$. Any one-shot solution with budget $1$ blocks one node;
blocking $a_i$ (or a node on its path) saves at most $L+1$ nodes and only if
$a_i$ is activated, which happens with probability $p$: expected spread
$\ge(\Delta p-p)L$. DEFER observes $N_1=\{a_i:\ (s,a_i)\text{ live}\}$ and
blocks the first path node of one activated branch; if $k$ branches are
active the spread is $(k-1)L+k$. Its expected spread is
$\mathbb E[(k-1)^+]L+\Delta p\le(\Delta p-1+(1-p)^\Delta)L+\Delta p$
(DEFER's plan at round $0$ is a branch head $a_i$ with $q=p$, $c=1$ and
$\lambda\approx pL$, so the push-down rule $p(1+1/(pL))<1$ defers it). The
ratio of one-shot to DEFER is at least
$\frac{(\Delta-1)p}{\Delta p-1+(1-p)^\Delta}$, which for $p\to0$ with
$\Delta p=c$ fixed tends to $c/(c-1+e^{-c})$ and, for fixed $\Delta$ as
$p\to0$, behaves like $\frac{(\Delta-1)p}{\binom{\Delta}{2}p^2}=\frac{2}{\Delta p}\to\infty$;
the stated bound $(1-(1-p)^\Delta)/p$ follows from comparing the saving
$L\cdot p$ of the best one-shot node with the saving
$L\cdot\Pr[k\ge1]=L(1-(1-p)^\Delta)$ of DEFER. $\square$

*Remark.* COMMIT (re-plan every round but commit the whole plan) is not
covered by Lemma 1: it can spend the budget in round $0$ on nodes that a
later observation would have shown to be irrelevant. In the experiments it
is often worse than the one-shot planner.

**Theorem 3 (complexity).** Let $n_t, m_t$ be the number of nodes and edges
reachable from $F_t$ in the union of the $\theta$ samples, $r$ the number
of rounds and $\alpha$ the inverse Ackermann function. With the dominator
planner, round $t$ costs $O(\theta\,m_t\,\alpha(m_t,n_t))$ for the trees
plus $O(b_t\,\theta'\,m_t\,\alpha)$ for the greedy picks, where
$\theta'\le\theta$ is the number of samples in which the picked node is
reachable; the total is $O\bigl(\sum_t(1+b_t)\theta m_t\alpha\bigr)$, which is
$O(r\,b\,\theta\,m\,\alpha(m,n))$ in the worst case and much smaller in
practice because $m_t$ is the size of the *current* reachable region and
$\sum_t b_t$ counts only uncommitted budget. One-shot AG costs
$O(b\,\theta\,m\,\alpha(m,n))$ (Xie et al. 2024); GR costs
$O(\min\{d^{out}_S,b\}\,\theta\,m\,\alpha)$ plus the replacement pass.

*Proof.* Each dominator tree is computed by the Cooper-Harvey-Kennedy
iteration on the reachable subgraph (near-linear; Lengauer-Tarjan gives the
$\alpha$ bound). Step 2 is a BFS per sample. Recomputation after a pick is
restricted to samples whose cached reach set contains the picked node, which
is exactly the set of samples whose dominator tree can change. Summation over
picks and rounds gives the bound. $\square$

## 4. What is new

* The adaptive variant of IMIN with node blocking and the observation that
  the *timing* of commitment, not re-planning, is what adaptivity buys:
  deferring is free (Lemma 1), committing early is harmful (COMMIT), and
  the gap to one-shot is unbounded (Theorem 2).
* DEFER turns any one-shot IMIN algorithm into an adaptive one that is never
  worse than it (Theorem 1) at the price of one planning call per round on
  the current reachable region (Theorem 3).
* The dominator planner prices single nodes and frontier isolations
  together; the isolation move is what makes a plan able to seal a frontier
  node whose children are individually cheap (the non-submodular
  coordinated-cut case).

## 5. Experimental protocol (`trace_e/eval/run_adaptive.py`)

Graphs: ca-GrQc, ca-HepTh, power grid, BA(2000), soc-Epinions1 where
feasible. Probabilities: constant $p\in\{0.05,0.1\}$ and weighted cascade.
Seeds: 20 or 50, random or highest degree. Budgets 20, 50, 100. For each
instance and each of 5 live-edge realisations every policy runs on the same
realisation with the same total budget (common random numbers). Policies:
none, AG, GR, LSBM, proximity, degree (one-shot); DEFER with the dominator
planner, DEFER with GR as planner, COMMIT, DEFER with a 4-hop planning
horizon. Reported: final spread beyond the seeds (mean and s.e.), saved
fraction, budget used, wall-clock per episode including planning.
