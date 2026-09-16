"""Simulation-based estimators that use outbreaks simulated from every node.

* ``sme``: Soft Margin Estimator (Antulov-Fantulin et al., PRL 2015).
* ``mcs``: Monte-Carlo mean-field node-state likelihood (the simulation
  baseline of Sterchi et al.), i.e. an independent-node likelihood of the
  snapshot under per-source empirical state frequencies with add-one smoothing.
"""
from __future__ import annotations

import numpy as np

from .base import SourceDetector, register


def _per_source_index(train):
    n = train.states.shape[1]
    order = np.argsort(train.sources, kind="stable")
    counts = np.bincount(train.sources, minlength=n)
    return order, counts


@register
class SoftMargin(SourceDetector):
    name = "sme"
    probabilistic = True
    needs_training_sims = True

    def __init__(self, ctx, a: float | None = None, **params):
        super().__init__(ctx, **params)
        self.a_grid = np.array([0.5 ** i for i in range(1, 16)]) if a is None else np.array([a])

    def fit(self, train=None):
        assert train is not None, "SME needs simulated outbreaks"
        self.mask = (train.states != 0)  # (n_sim, n)
        self.mask_sum = self.mask.sum(axis=1)
        self.sources = train.sources
        self.n_sim = len(train)

    def score(self, states):
        obs = (states != 0)
        inter = self.mask[:, obs].sum(axis=1)
        union = self.mask_sum + obs.sum() - inter
        jac = inter / np.maximum(union, 1)
        n = self.ctx.n
        # kernel per simulation for each a, averaged per source
        best = None
        prev_map = None
        for a in self.a_grid:  # from wide to narrow kernels
            k = np.exp(-((jac - 1.0) ** 2) / (a * a))
            per_src = np.bincount(self.sources, weights=k, minlength=n) / np.maximum(np.bincount(self.sources, minlength=n), 1)
            if per_src.sum() <= 0 or not np.isfinite(per_src).all():
                break
            p = per_src / per_src.sum()
            m = int(np.argmax(p))
            if prev_map is not None and m == prev_map and abs(p[m] - prev_p[m]) > 0.05:
                break  # estimate stopped being stable: keep previous kernel
            best = per_src
            prev_map, prev_p = m, p
            if per_src[m] < 1e-12:
                break
        if best is None:
            best = np.ones(n)
        s = np.log(np.maximum(best, 1e-300))
        return self.restrict(s, states)


@register
class MonteCarloMeanField(SourceDetector):
    name = "mcs"
    probabilistic = True
    needs_training_sims = True

    def fit(self, train=None):
        assert train is not None, "MCS needs simulated outbreaks"
        n = self.ctx.n
        counts = np.ones((n, n, 3))  # add-one smoothing, indexed [source, node, state]
        for src, st in zip(train.sources, train.states):
            counts[src, np.arange(n), st] += 1
        counts[np.arange(n), np.arange(n), 0] -= 1  # a source is never susceptible
        with np.errstate(divide="ignore"):
            self.logp = np.log(counts / counts.sum(axis=2, keepdims=True))

    def score(self, states):
        n = self.ctx.n
        s = self.logp[:, np.arange(n), states.astype(int)].sum(axis=1)
        return self.restrict(s, states)
