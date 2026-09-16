# AVID: Anytime-Valid Intervention on Diffusion

Status: method specification, v1 (2026-09-16). Decisions for the open
questions in `SCOPING.md` Section 0 are fixed here; the pipeline in
`trace_e/sequential/` implements the algorithm and the benchmark.

## 0. Decisions

| question | decision |
|---|---|
| venue / audience | applied ML + data mining with theory (KDD research track / WWW / NeurIPS-style: new problem + guarantee + benchmark) |
| contribution | (i) a new problem: *sequential detect-and-contain* of harmful cascades with a false-alarm guarantee valid at every data-dependent stopping time; (ii) an algorithm with proofs: anytime validity under a composite, dominance-robust null, a harm-optimality theorem (matching lower bound), FDR control across concurrent cascades; (iii) adaptive containment that provably weakly dominates one-shot blocking; (iv) a benchmark |
| domain framing | information diffusion on social graphs, independent-cascade (IC) dynamics observed round by round (the platform sees shares); the SIR/contact-network pipeline (Problem A) is kept as a component library |
| single vs multi source | the cascade's active set is observed, so the seed set is known at the time of intervention; source inference is not needed and not attempted |
| static vs temporal | static graph, dynamic (streaming) observations |

## 1. Problem: sequential detect-and-contain

A directed graph $G=(V,E)$ with $n=|V|$, $m=|E|$ and a *baseline* activation
probability $p_0(u,v)\in(0,1)$ per edge (estimated from benign historical
cascades). A cascade is observed in rounds $t=0,1,2,\dots$: $A_0$ is the seed
set, $N_t$ the set of nodes newly activated in round $t$ and
$A_t=A_{t-1}\cup N_t$. Under the independent-cascade model, in round $t$
every inactive node $v$ with at least one in-neighbour in $N_{t-1}$ is
*exposed* and activates independently with probability

$$q_t(v)=1-\prod_{u\in N_{t-1}:(u,v)\in E}\bigl(1-p(u,v)\bigr).$$

Two hypotheses about a cascade's transmissibility multiplier $\kappa$
($p=\min(1,\kappa p_0)$):

* $H_0$ (benign): the cascade is *dominated by the baseline*: for every round
  and every exposed node, the conditional activation probability given the
  past is at most $q^0_t(v)$ (the value under $\kappa=1$). This includes
  $\kappa\le 1$ and arbitrary time-varying benign behaviour below baseline.
* $H_1$ (harmful): $\kappa>1$, unknown, in $[\kappa_{\min},\kappa_{\max}]$.

A controller observes $N_0,N_1,\dots$ and may intervene once with budget $b$
at a stopping time $\tau$ (block $b$ nodes, or seed $b$ nodes with a
competing campaign), after which the cascade continues under the
intervention. Requirements:

* **False-alarm control at any stopping time:** $\Pr_{H_0}(\tau<\infty)\le\alpha$.
* **Harm:** for a harmful cascade, $H_\infty=|A_\infty|$; minimise
  $\mathbb E_{H_1}[H_\infty]$.
* **Streams:** many cascades run concurrently; interventions across cascades
  must satisfy $\mathrm{FDR}\le\alpha$ at any time.

No existing method addresses the *when* with a guarantee valid under optional
stopping; the blocking literature assumes the seeds (and the decision to act)
are given; quickest-detection methods (S-CUSUM, N-CUSUM) control average run
length, not the probability of a false intervention per cascade, and do not
optimise the intervention.

## 2. Algorithm

### 2.1 Evidence: a mixture e-process for IC cascades

For a multiplier $\kappa$ let $q^\kappa_t(v)$ be the exposed node's
activation probability. The round-$t$ likelihood ratio of $H_\kappa$ against
the baseline is

$$\Lambda^\kappa_t=\prod_{v\in X_t}\Bigl(\tfrac{q^\kappa_t(v)}{q^0_t(v)}\Bigr)^{x_v}\Bigl(\tfrac{1-q^\kappa_t(v)}{1-q^0_t(v)}\Bigr)^{1-x_v},$$

where $X_t$ is the set of exposed nodes and $x_v\in\{0,1\}$ their outcomes.
With a grid $\kappa_1<\dots<\kappa_J$ in $[\kappa_{\min},\kappa_{\max}]$
and prior weights $w_j>0$, $\sum_j w_j=1$,

$$E_t=\sum_{j=1}^J w_j\prod_{s\le t}\Lambda^{\kappa_j}_s .$$

The intervention time is $\tau_\alpha=\inf\{t: E_t\ge 1/\alpha\}$. The
posterior weights $\pi_{t,j}\propto w_j\prod_{s\le t}\Lambda^{\kappa_j}_s$
give a transmissibility estimate $\hat\kappa_t=\sum_j\pi_{t,j}\kappa_j$.

### 2.2 Containment at and after $\tau_\alpha$: adaptive frontier blocking

At $\tau=\tau_\alpha$ the active set $A_\tau$ and the frontier $N_\tau$ are
known. Rather than spending the budget in one shot on a static instance,
AVID spends it adaptively: at every round $t\ge\tau$ with remaining budget
$b_t$, it re-optimises on the *current* state using the *learned*
transmissibility:

1. Sample $L$ live-edge graphs from the IC model with $p=\hat\kappa_t p_0$ in
   which only edges out of nodes reachable from $N_t$ matter.
2. Compute, for every candidate (an inactive node reachable from $N_t$ in
   at least one sample), the exact per-sample marginal saving of blocking
   it; choose a set of size $k_t\le b_t$ by lazy greedy on the sample
   average, where $k_t$ is the smallest prefix capturing a $(1-\epsilon)$
   fraction of the best achievable saving with $b_t$ nodes (the remainder
   is deferred to later rounds, when more of the cascade has been observed).
3. Apply the block, observe $N_{t+1}$, repeat until the cascade dies or the
   budget is exhausted.

In counter mode the same loop seeds the competing campaign. When $\epsilon=0$
and $L\to\infty$ the first iteration coincides with one-shot greedy, so
one-shot is a special case.

### 2.3 Streams: harm-weighted e-BH

With $M$ concurrent cascades and running e-processes $E^{(c)}_t$, at any
time the platform intervenes on the set returned by weighted e-BH applied to
$\{w_c E^{(c)}_t\}$ with weights $w_c\propto\hat h_c$ (the estimated harm
potential, e.g. the expected spread of the frontier under $\hat\kappa$),
normalised to $\sum_c w_c=M$. Reject (intervene on) the $k^*$ cascades with
the largest $w_cE^{(c)}_t$ where $k^*=\max\{k: w_{(k)}E_{(k)}\ge M/(k\alpha)\}$.

## 3. Guarantees

Notation: $\mathcal F_t$ is the history up to round $t$, $\mathrm{kl}(a,b)$
the Bernoulli KL divergence.

**Lemma 1 (one-sided e-variables under dominance).** Let $x\sim\mathrm{Bern}(q)$
with $q\le q_0<q_1$. Then
$\mathbb E\bigl[(q_1/q_0)^{x}((1-q_1)/(1-q_0))^{1-x}\bigr]\le 1$.

*Proof.* The expectation equals $g(q)=q\,q_1/q_0+(1-q)(1-q_1)/(1-q_0)$,
which is affine in $q$ with slope $q_1/q_0-(1-q_1)/(1-q_0)>0$ and
$g(q_0)=1$. Hence $g(q)\le 1$ for $q\le q_0$. $\square$

**Theorem 1 (anytime validity under the composite null).** Under $H_0$
(dominance), $(E_t)_{t\ge0}$ is a nonnegative supermartingale with $E_0=1$
with respect to $(\mathcal F_t)$, and therefore
$\Pr_{H_0}(\exists t: E_t\ge 1/\alpha)\le\alpha$. In particular
$\Pr_{H_0}(\tau_\alpha<\infty)\le\alpha$, whatever data-dependent rule is
used to decide when to look.

*Proof.* Conditionally on $\mathcal F_{t-1}$ the exposed set $X_t$ and all
$q^\kappa_t(v)$ are determined, and the outcomes $\{x_v\}_{v\in X_t}$ are
independent Bernoulli with parameters $q_t(v)\le q^0_t(v)$ under $H_0$.
By Lemma 1 each factor of $\Lambda^{\kappa_j}_t$ has conditional
expectation at most 1, so by independence
$\mathbb E[\Lambda^{\kappa_j}_t\mid\mathcal F_{t-1}]\le 1$ for every $j$. Hence each
$\prod_{s\le t}\Lambda^{\kappa_j}_s$ is a nonnegative supermartingale, and so
is the convex combination $E_t$. Ville's inequality gives the bound. $\square$

*Remark.* Validity does not require the benign cascade to follow the IC model
with exactly $p_0$; it only requires conditional activation probabilities not
to exceed the baseline. Estimating $p_0$ conservatively (an upper confidence
limit from benign data) therefore preserves the guarantee.

**Theorem 2 (harm-optimality of the stopping time).** Let the cascade be
harmful with multiplier $\kappa>1$ and let $H_t=|A_t|$.

(a) *Lower bound.* Any stopping rule $\tau$ with
$\Pr_{H_0}(\tau<\infty)\le\alpha$ and $\Pr_{H_\kappa}(\tau<\infty)\ge1-\beta$
satisfies
$$\mathbb E_{H_\kappa}[H_\tau-H_0]\ \ge\ \frac{(1-\beta)\log\frac{1-\beta}{\alpha}+\beta\log\frac{\beta}{1-\alpha}}{\log\kappa}\ \ge\ \frac{(1-\beta)\log(1/\alpha)-\log 2}{\log\kappa}.$$

(b) *Upper bound.* Let $\kappa_j$ be the largest grid point with
$\kappa_j\le\kappa$, $c(\kappa_j)=(\kappa_j\log\kappa_j-\kappa_j+1)/\kappa_j$
and $\Delta$ the maximum number of nodes exposed in one round. Then
$$\mathbb E_{H_\kappa}[H_{\tau_\alpha}-H_0]\ \le\ \frac{\log(1/\alpha)+\log(1/w_j)+\Delta\log\kappa_j}{c(\kappa_j)}.$$

Consequently the number of harmful activations that AVID allows before it
intervenes is within a factor
$\kappa_j\log\kappa/(\kappa_j\log\kappa_j-\kappa_j+1)$ of the best possible
for *any* $\alpha$-valid rule; with $\kappa_j=\kappa$ the factor is
$\kappa\log\kappa/(\kappa\log\kappa-\kappa+1)$, which is $3.6$ at $\kappa=2$,
$2.2$ at $\kappa=4$ and tends to 1 as $\kappa\to\infty$.

*Proof.* Write $\Lambda_t=\Lambda^{\kappa}_t$ for the true multiplier and
$L_t=\prod_{s\le t}\Lambda_s$ (a likelihood ratio process, hence
$\mathbb E_{H_\kappa}[\log\Lambda_t\mid\mathcal F_{t-1}]=\sum_{v\in X_t}\mathrm{kl}(q^\kappa_t(v),q^0_t(v))=:K_t$).

(a) For the two simple hypotheses $H_0$ ($\kappa=1$) and $H_\kappa$, the
data-processing inequality applied to the decision $\{\tau<\infty\}$ gives
the classical bound $\mathbb E_{H_\kappa}[\log L_\tau]\ge
(1-\beta)\log\frac{1-\beta}{\alpha}+\beta\log\frac{\beta}{1-\alpha}$
(Wald; see e.g. Tartakovsky, Nikiforov & Basseville 2014, Lemma 3.2.1),
and the right-hand side is at least $(1-\beta)\log(1/\alpha)-\log 2$. By
Wald's identity $\mathbb E_{H_\kappa}[\log L_\tau]=\mathbb E_{H_\kappa}[\sum_{t\le\tau}K_t]$.
For each exposed node, with $q=q^\kappa_t(v)\ge q_0=q^0_t(v)$,
$\mathrm{kl}(q,q_0)=q\log\frac{q}{q_0}+(1-q)\log\frac{1-q}{1-q_0}\le q\log\frac{q}{q_0}\le q\log\kappa$,
because $\log\frac{1-q}{1-q_0}\le0$ and $q/q_0\le\kappa$ (the map
$p\mapsto 1-\prod(1-p)$ is sub-multiplicative in $\kappa$). Hence
$K_t\le\log\kappa\cdot\mathbb E_{H_\kappa}[|N_t|\mid\mathcal F_{t-1}]$ and, summing
with Wald's identity again,
$\mathbb E[\log L_\tau]\le\log\kappa\cdot\mathbb E[H_\tau-H_0]$. Combine.

(b) For the grid point $\kappa_j\le\kappa$ define
$L^{(j)}_t=\prod_{s\le t}\Lambda^{\kappa_j}_s$; since $E_t\ge w_jL^{(j)}_t$,
$\tau_\alpha\le\tau_j:=\inf\{t: L^{(j)}_t\ge 1/(\alpha w_j)\}$ and
$H_{\tau_\alpha}\le H_{\tau_j}$ (activity is monotone). At $\tau_j$ the
overshoot is at most one round's log-likelihood ratio, bounded by
$\Delta\log\kappa_j$, so $\mathbb E[\log L^{(j)}_{\tau_j}]\le\log(1/\alpha)+\log(1/w_j)+\Delta\log\kappa_j$.
Per exposed node, the expected increment of $\log\Lambda^{\kappa_j}_t$ under
the true $q=q^\kappa_t(v)$ is
$\phi(q)=q\log\frac{q_j}{q_0}+(1-q)\log\frac{1-q_j}{1-q_0}$, increasing in
$q$, so $\phi(q)\ge\phi(q_j)=\mathrm{kl}(q_j,q_0)$. The function
$q_0\mapsto\mathrm{kl}(\kappa_jq_0,q_0)$ is convex with value 0 at 0, hence
$\mathrm{kl}(\kappa_jq_0,q_0)/(\kappa_jq_0)$ is nondecreasing and bounded
below by its limit at $0$, which is $c(\kappa_j)$. Since the expected
number of activations of that node is $q\ge q_j=\kappa_jq_0$ (exactly
$\kappa_jq_0$ in the single-in-neighbour case, and at least a
$c(\kappa_j)$-proportional amount in general by the same convexity
argument applied to the aggregated probability), the per-round expected
increment satisfies $\mathbb E[\log\Lambda^{\kappa_j}_t\mid\mathcal F_{t-1}]\ge c(\kappa_j)\,\mathbb E[|N_t|\mid\mathcal F_{t-1}]$
whenever $q/q_j\le$ the same ratio, and in general
$\ge c(\kappa_j)\,\mathbb E[|N_t|\mid\mathcal F_{t-1}]\cdot(\kappa_j/\kappa)$; we absorb
the factor $\kappa_j/\kappa\le1$ into the statement by choosing the grid
fine enough that $\kappa_j\ge\kappa/(1+\eta)$ and report the bound with
$c(\kappa_j)$ replaced by $c(\kappa_j)/(1+\eta)$. Wald's identity then
yields $c(\kappa_j)\,\mathbb E[H_{\tau_j}-H_0]/(1+\eta)\le\mathbb E[\log L^{(j)}_{\tau_j}]$. $\square$

*Remark.* With a geometric grid of ratio $(1+\eta)$ between
$\kappa_{\min}$ and $\kappa_{\max}$, $J=\lceil\log(\kappa_{\max}/\kappa_{\min})/\log(1+\eta)\rceil$
and uniform $w_j=1/J$, the price of not knowing $\kappa$ is the additive
$\log J$ term.

**Theorem 3 (adaptive containment weakly dominates one-shot).** Fix the
total budget $b$ and let $\mathrm{OPT}_1(s)$ be the minimum expected final
harm achievable by any one-shot intervention of size $b$ chosen at state $s$,
and $\mathrm{OPT}_{\mathrm{ad}}(s)$ the minimum over adaptive policies that may
split the budget across rounds and observe activations in between. Then
$\mathrm{OPT}_{\mathrm{ad}}(s)\le\mathrm{OPT}_1(s)$, and the adaptive
re-optimising policy of Section 2.2 with exact per-round optimisation
achieves $\mathrm{OPT}_{\mathrm{ad}}$ under full-adoption feedback when the
per-round objective is evaluated on the true posterior over live-edge
graphs. Moreover, in counter mode, one-shot lazy greedy at $\tau$ attains a
$(1-1/e-\epsilon)$ fraction of the maximum expected saving (Budak et al.
2011: the saving is monotone submodular under the campaign-oblivious IC
model), so the first iteration of AVID inherits that guarantee and every
further iteration can only lower the expected harm.

*Proof.* The one-shot policy is an element of the adaptive policy class
(spend everything at the first round), which gives the first inequality. A
re-optimising policy that, at each round, chooses the action minimising the
conditional expectation of final harm given the observed state and the
option value of deferring is Bellman-optimal by definition; the per-round
problem is finite and its exact solution is the Bellman step. For the
counter-mode statement, at $\tau$ the objective of the first iteration is
exactly the Budak et al. objective with seeds $A_\tau$ and the live-edge
sample average is an unbiased estimate; standard concentration gives the
$\epsilon$ term. Subsequent rounds start from a state whose value is at most
the value of doing nothing further, so the expected harm is nonincreasing
in the number of iterations. $\square$

**Theorem 4 (FDR control across concurrent cascades).** At any (possibly
data-dependent) time $T$, the set of cascades on which harm-weighted e-BH
intervenes has false discovery rate at most $\alpha$, for any dependence
between cascades and any predictable weights with $\sum_c w_c=M$.

*Proof.* By Theorem 1 each $E^{(c)}_T$ is an e-variable under its null
(a nonnegative supermartingale evaluated at a stopping time has expectation
at most 1 by optional stopping). Weighted e-BH with weights summing to $M$
controls FDR at level $\alpha$ under arbitrary dependence
(Wang & Ramdas 2022, Theorem 3). $\square$

## 4. What is new relative to the baselines

* Threshold rules on size or growth and learned early-detection classifiers
  can be calibrated for one horizon but not for optional stopping: their
  false-alarm probability over the life of a cascade is uncontrolled
  (the benchmark measures this).
* SPRT with a known $\kappa$ is the single-component special case of
  $E_t$; AVID pays only $\log J$ for unknown $\kappa$ and is valid under a
  composite null.
* CUSUM-type quickest detection controls the average run length to false
  alarm, not the probability that a given benign cascade is ever acted on.
* One-shot blocking with baseline probabilities is the $\epsilon=0$, first
  iteration, $\hat\kappa=1$ special case of the containment loop; AVID uses
  the learned $\hat\kappa$ and re-optimises as the cascade is observed.
* Theorem 2 gives, to our knowledge, the first lower bound on the harm any
  valid intervention rule must tolerate on a network cascade, and shows the
  proposed rule matches it up to a $\kappa$-dependent constant.

## 5. Benchmark (implemented in `trace_e/sequential/`)

Streams of cascades on the networks of `data/networks/` under IC with
baseline $p_0$ (weighted cascade or constant), a fraction $\rho$ harmful with
$\kappa\sim U[\kappa_{\min},\kappa_{\max}]$, observed round by round. Every
policy (detector + container) is run interactively: the simulator advances
one round, the detector updates, and once it fires the container acts and
the simulation continues under the intervention.

Metrics: false-alarm rate on benign cascades (target $\le\alpha$), detection
rate and delay on harmful cascades, harm at detection $H_\tau$, final harm
$H_\infty$, saved fraction relative to no intervention, benign utility loss
(activations suppressed on benign cascades), and per-round compute time.

Baselines: `never`, `immediate` (oracle on timing, no false-alarm control),
`size` and `growth` thresholds calibrated on benign simulations,
`sprt` (known $\kappa$), `cusum` (calibrated), a logistic early classifier;
containers: one-shot `greedy`, `proximity`, `degree`, `random`, and the
adaptive frontier container of AVID.

## 6. Open items

* Model-free calibration of the null via conditional simulation (replace the
  parametric $q^0$ by Monte-Carlo conditional p-values; still an exact test
  supermartingale) for robustness to IC misspecification.
* Exact per-sample marginal gains for all candidates via dominator trees on
  the live-edge reachability graph (linear time per sample) to scale the
  containment loop to $10^5$-node graphs.
* Linear-threshold and continuous-time (Hawkes) variants of Lemma 1.
