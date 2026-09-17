"""Exact source-posterior computation on bounded-treewidth graphs for the
discrete-time SIR model, via variable elimination over the time-unrolled
dynamic graphical model.

Motivation. Computing P(observed snapshot | source = v) exactly under SIR
dynamics is the same #P-hard flavour of problem that makes people fall back
on either (a) Monte-Carlo simulation (SME, Antulov-Fantulin et al. 2015), or
(b) a mean-field / cavity approximation that assumes the states of a node's
neighbours are independent given the node's own history (DMP, Lokhov et al.
2014; MCMF, Sterchi et al. 2023). That independence assumption is exactly
correct when the contact graph is a tree (this is *why* DMP-style message
passing is exact on trees) and becomes an approximation as soon as the graph
has cycles, because two neighbours of a node can share a common ancestor,
inducing a correlation the mean-field update ignores.

This module removes that approximation for graphs of bounded treewidth, by
running exact variable elimination over the graph's own chordal completion
(the same elimination order used by ``trace_e.blocking.treewidth`` for the
IC-model spread computation) at *every discrete time step*, rather than
maintaining independent per-node marginals. Concretely:

* Variables are X_i^t in {S=0, I=1, R=2} for every node i and time step
  t = 0..K (K = round(T / dt)). X_i^0 is evidence (I for the source, S for
  everyone else), not a free variable.
* The transition psi_i(X_i^t | X_i^{t-1}, X_{N(i)}^{t-1}) factorises exactly
  like the graph's own adjacency: node i's next state depends only on its
  own current state and its immediate neighbours' current states. This
  means the "moralised" graphical model for one transition step has exactly
  the same edge structure as the contact graph itself, so the elimination
  order computed once for the static graph is reusable, unmodified, as the
  per-time-step elimination order here.
* We process time steps t = 1..K in order. At each step we (i) build that
  step's n transition factors (using the *previous* step's not-yet-
  eliminated variables), (ii) at t = K only, also multiply in a delta
  factor clamping X_i^K to the observed state, then (iii) eliminate every
  X_i^{t-1} variable (in the static elimination order) now that nothing
  later in time can depend on it -- exactly analogous to how a vertex
  "retires" once all its edges are processed in the IC-model DP, except
  here retirement happens on a rolling basis once per time step rather
  than once ever, since every node's state persists for the entire time
  horizon.

Because the moralised structure of every step matches the *static* graph's
adjacency, the width of this elimination order is the ORIGINAL graph's
treewidth plus a small constant, independent of K: this is the standard
"unrolled DBN has bounded treewidth" fact from the graphical-models
literature (e.g. the interface/frontier algorithm for dynamic Bayesian
networks). The DP is therefore polynomial in K and the number of time steps
never appears in an exponent -- unlike, say, tracking each node's full
"infection time" distribution jointly, which would multiply the per-node
alphabet size by K and *does* blow up (an earlier, abandoned design; see
``docs/EXACT_SIR.md``).

Correctness is checked in ``tests/test_exact_sir.py`` against an explicit
brute-force simulation of the full 3^n-state Markov chain on random graphs
small enough to enumerate (n <= 6).
"""
from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from ..blocking.treewidth import elimination_order
from .base import SourceDetector, register

S, I, R = 0, 1, 2


# ---------------------------------------------------------------------------
# Minimal variable-elimination engine (factors over {0,1,2}-valued variables)
# ---------------------------------------------------------------------------

class Factor:
    """A function from an assignment of a tuple of variables to a probability
    (or unnormalised weight), stored as a dict keyed by the assignment tuple
    (in the same order as ``scope``)."""

    __slots__ = ("scope", "table")

    def __init__(self, scope: tuple, table: dict):
        self.scope = scope
        self.table = table

    @staticmethod
    def constant(value: float) -> "Factor":
        return Factor((), {(): value})

    def multiply(self, other: "Factor") -> "Factor":
        if not self.scope:
            v = self.table.get((), 0.0)
            return Factor(other.scope, {k: v * p for k, p in other.table.items()})
        if not other.scope:
            v = other.table.get((), 0.0)
            return Factor(self.scope, {k: v * p for k, p in self.table.items()})
        shared = [v for v in self.scope if v in other.scope]
        new_scope = self.scope + tuple(v for v in other.scope if v not in self.scope)
        self_idx = {v: i for i, v in enumerate(self.scope)}
        other_idx = {v: i for i, v in enumerate(other.scope)}
        table: dict = {}
        for a_key, a_val in self.table.items():
            if a_val == 0.0:
                continue
            for b_key, b_val in other.table.items():
                if b_val == 0.0:
                    continue
                if shared and any(a_key[self_idx[v]] != b_key[other_idx[v]] for v in shared):
                    continue
                extra = tuple(b_key[other_idx[v]] for v in other.scope if v not in self_idx)
                new_key = a_key + extra
                table[new_key] = table.get(new_key, 0.0) + a_val * b_val
        return Factor(new_scope, table)

    def sumout(self, var) -> "Factor":
        if var not in self.scope:
            return self
        idx = self.scope.index(var)
        new_scope = self.scope[:idx] + self.scope[idx + 1:]
        table: dict = {}
        for key, val in self.table.items():
            new_key = key[:idx] + key[idx + 1:]
            table[new_key] = table.get(new_key, 0.0) + val
        return Factor(new_scope, table)


def _eliminate(factors: list[Factor], var) -> list[Factor]:
    """Multiply every factor mentioning ``var``, sum it out, return the
    updated factor list (with the merged/reduced factor appended)."""
    touching = [f for f in factors if var in f.scope]
    rest = [f for f in factors if var not in f.scope]
    if not touching:
        return factors
    merged = touching[0]
    for f in touching[1:]:
        merged = merged.multiply(f)
    reduced = merged.sumout(var)
    rest.append(reduced)
    return rest


# ---------------------------------------------------------------------------
# SIR transition factor construction
# ---------------------------------------------------------------------------

def _transition_probs(prev_self: int, prev_neighbors: Sequence[int], lam: Sequence[float], mu: float):
    """P(next state | prev_self, prev_neighbors) as a length-3 array."""
    if prev_self == R:
        return np.array([0.0, 0.0, 1.0])
    if prev_self == I:
        return np.array([0.0, 1.0 - mu, mu])
    # prev_self == S
    p_stay = 1.0
    for x_j, l in zip(prev_neighbors, lam):
        if x_j == I:
            p_stay *= (1.0 - l)
    return np.array([p_stay, 1.0 - p_stay, 0.0])


class ExactSIR:
    """Exact P(observed snapshot | source = v) via time-unrolled variable
    elimination, for a single fixed candidate source v."""

    def __init__(self, G, weight_fn, beta: float, nu: float, T: float, dt: float = 0.05, order=None):
        self.G = G
        self.weight_fn = weight_fn
        self.beta = beta
        self.nu = nu
        self.mu = 1.0 - math.exp(-nu * dt)
        self.K = max(1, int(round(T / dt)))
        self.dt = dt
        self.order = order if order is not None else elimination_order(G)
        self.neighbors = {v: list(G.neighbors(v)) for v in G.nodes}
        self.lam = {
            (u, v): 1.0 - math.exp(-beta * weight_fn(u, v) * dt)
            for v in G.nodes for u in self.neighbors[v]
        }

    def log_likelihood(self, source, observed: dict) -> float:
        """observed: dict node -> state in {0,1,2} (the full snapshot at t=K)."""
        n_nodes = self.G.number_of_nodes()
        prev_state = {v: (I if v == source else S) for v in self.G.nodes}
        factors: list[Factor] = []

        for t in range(1, self.K + 1):
            new_factors = []
            for v in self.G.nodes:
                nbrs = self.neighbors[v]
                if t == 1:
                    # t-1 = 0 is evidence: prev_self / prev_neighbors are known constants.
                    lam_v = [self.lam[(u, v)] for u in nbrs]
                    probs = _transition_probs(prev_state[v], [prev_state[u] for u in nbrs], lam_v, self.mu)
                    table = {(x,): probs[x] for x in range(3) if probs[x] > 0.0}
                    new_factors.append(Factor(((v, t),), table))
                else:
                    lam_v = [self.lam[(u, v)] for u in nbrs]
                    scope = ((v, t), (v, t - 1)) + tuple((u, t - 1) for u in nbrs)
                    table = {}
                    for assignment in itertools.product(range(3), repeat=len(nbrs) + 1):
                        self_prev = assignment[0]
                        nbr_prev = assignment[1:]
                        probs = _transition_probs(self_prev, nbr_prev, lam_v, self.mu)
                        for x in range(3):
                            if probs[x] > 0.0:
                                table[(x,) + assignment] = probs[x]
                    new_factors.append(Factor(scope, table))
            factors.extend(new_factors)

            if t == self.K:
                for v in self.G.nodes:
                    obs = observed[v]
                    factors.append(Factor(((v, t),), {(obs,): 1.0}))

            if t > 1:
                for v in self.order:
                    factors = _eliminate(factors, (v, t - 1))

        for v in self.order:
            factors = _eliminate(factors, (v, self.K))

        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        assert result.scope == (), (
            f"elimination did not fully reduce (n={n_nodes}, remaining scope={result.scope})"
        )
        p = result.table.get((), 0.0)
        return math.log(p) if p > 0.0 else float("-inf")

    def max_active_width(self, source, observed: dict) -> int:
        """Diagnostic: largest factor scope size seen during elimination
        (informal proxy for the induced width actually realised)."""
        n_nodes = self.G.number_of_nodes()
        prev_state = {v: (I if v == source else S) for v in self.G.nodes}
        factors: list[Factor] = []
        widest = 0
        for t in range(1, self.K + 1):
            new_factors = []
            for v in self.G.nodes:
                nbrs = self.neighbors[v]
                if t == 1:
                    lam_v = [self.lam[(u, v)] for u in nbrs]
                    probs = _transition_probs(prev_state[v], [prev_state[u] for u in nbrs], lam_v, self.mu)
                    table = {(x,): probs[x] for x in range(3) if probs[x] > 0.0}
                    new_factors.append(Factor(((v, t),), table))
                else:
                    lam_v = [self.lam[(u, v)] for u in nbrs]
                    scope = ((v, t), (v, t - 1)) + tuple((u, t - 1) for u in nbrs)
                    table = {}
                    for assignment in itertools.product(range(3), repeat=len(nbrs) + 1):
                        self_prev = assignment[0]
                        nbr_prev = assignment[1:]
                        probs = _transition_probs(self_prev, nbr_prev, lam_v, self.mu)
                        for x in range(3):
                            if probs[x] > 0.0:
                                table[(x,) + assignment] = probs[x]
                    new_factors.append(Factor(scope, table))
            factors.extend(new_factors)
            if t == self.K:
                for v in self.G.nodes:
                    obs = observed[v]
                    factors.append(Factor(((v, t),), {(obs,): 1.0}))
            if t > 1:
                for v in self.order:
                    factors = _eliminate(factors, (v, t - 1))
                    widest = max(widest, max((len(f.scope) for f in factors), default=0))
        for v in self.order:
            factors = _eliminate(factors, (v, self.K))
        return widest


def brute_force_posterior(G, weight_fn, source, observed: dict, beta: float, nu: float, T: float, dt: float = 0.05) -> float:
    """Reference implementation: explicit joint over all 3^n states, applied
    forward for K steps. Only for tiny graphs (n <= ~7); used purely to
    verify :class:`ExactSIR` in tests."""
    nodes = list(G.nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    mu = 1.0 - math.exp(-nu * dt)
    K = max(1, int(round(T / dt)))
    lam = {(u, v): 1.0 - math.exp(-beta * weight_fn(u, v) * dt) for u, v in G.edges}
    lam.update({(v, u): l for (u, v), l in list(lam.items())})
    neighbors = {v: list(G.neighbors(v)) for v in nodes}

    init = [S] * n
    init[idx[source]] = I
    dist = {tuple(init): 1.0}

    for _ in range(K):
        new_dist: dict = {}
        for config, p in dist.items():
            if p == 0.0:
                continue
            per_node_next = []
            for v in nodes:
                nbrs = neighbors[v]
                prev_self = config[idx[v]]
                prev_nbrs = [config[idx[u]] for u in nbrs]
                lam_v = [lam[(u, v)] for u in nbrs]
                per_node_next.append(_transition_probs(prev_self, prev_nbrs, lam_v, mu))
            for combo in itertools.product(*[range(3)] * n):
                w = 1.0
                for i in range(n):
                    w *= per_node_next[i][combo[i]]
                    if w == 0.0:
                        break
                if w > 0.0:
                    new_dist[combo] = new_dist.get(combo, 0.0) + p * w
        dist = new_dist

    target = tuple(observed[v] for v in nodes)
    return dist.get(target, 0.0)


@register
class ExactSIRPosterior(SourceDetector):
    """Exact (non-mean-field) SIR source posterior on bounded-treewidth
    graphs; see module docstring. Falls back to restricting candidates like
    DMP does, since this is even more expensive per-candidate than DMP."""

    name = "exact_sir"
    probabilistic = True

    def __init__(self, ctx, dt: float = 0.05, max_candidates: int = 60, **params):
        super().__init__(ctx, **params)
        self.dt = dt
        self.max_candidates = max_candidates
        self._order = None

    def score(self, states):
        inf = np.flatnonzero(states != 0)
        s = np.full(self.ctx.n, -np.inf)
        if len(inf) == 1:
            s[inf[0]] = 0.0
            return s
        if self._order is None:
            self._order = elimination_order(self.ctx.G)
        cands = inf
        if len(cands) > self.max_candidates:
            from .centrality import DistanceCenter
            dc = DistanceCenter(self.ctx).score(states)
            cands = np.argsort(-dc)[: self.max_candidates]
        observed = {v: int(states[v]) for v in self.ctx.G.nodes}
        for v in cands:
            model = ExactSIR(self.ctx.G, lambda a, b: 1.0, self.ctx.beta, self.ctx.nu, self.ctx.T, dt=self.dt, order=self._order)
            s[v] = model.log_likelihood(int(v), observed)
        return s
