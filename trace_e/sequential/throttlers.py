"""Soft, evidence-gated interventions applied *while* the test is running.

A throttler returns, for the current round, a per-node factor rho in
[rho_min, 1] that scales the bad cascade's activation probability into each
exposed node (a downrank). Because the factors are chosen before the round's
outcomes from information available at that time, the e-process computed
with the throttled baseline remains an exact test supermartingale
(Theorem 1 applies verbatim). Throttling is active only while the running
evidence exceeds a soft threshold 1/alpha_soft, which bounds the probability
that a benign cascade is ever throttled by alpha_soft.

* ``hub``     - information-preserving throttling (AVID-active): throttle the
                exposed nodes with the largest harm weight h(v) until a
                fraction ``frac`` of the exposure mass is covered. Evidence
                per activation is the same for every node while harm is
                not, so this minimises the harm accrued per unit of evidence
                (Theorem 5).
* ``uniform`` - throttle every exposed node by rho_min (control: same
                evidence/harm ratio as no throttling, slower in rounds).
* ``random``  - throttle a random ``frac`` of the exposure mass (control).
* ``none``    - no soft action.
"""
from __future__ import annotations

import numpy as np

THROTTLERS: dict[str, type] = {}


def register(cls):
    THROTTLERS[cls.name] = cls
    return cls


def get_throttler(name: str, **kw):
    if name not in THROTTLERS:
        raise KeyError(f"unknown throttler '{name}'. Known: {sorted(THROTTLERS)}")
    return THROTTLERS[name](**kw)


class Throttler:
    name = "base"

    def __init__(self, rho_min: float = 0.2, frac: float = 0.5, alpha_soft: float = 0.5, seed: int = 0, **kw):
        self.rho_min = rho_min
        self.frac = frac
        self.log_soft = np.log(1.0 / alpha_soft) if alpha_soft < 1 else -np.inf
        self.rng = np.random.default_rng(seed)
        self.out_mass = None

    def reset(self):
        self.rounds_active = 0

    def gated(self, detector) -> bool:
        return getattr(detector, "log_E", getattr(detector, "statistic", 0.0)) >= self.log_soft

    def harm_weight(self, g, exposed, kappa_hat):
        """h(v) = 1 + expected number of children if v activates (under kappa_hat)."""
        if self.out_mass is None:
            self.out_mass = np.array([g.weights[g.indptr[v]: g.indptr[v + 1]].sum() for v in range(g.n)])
        return 1.0 + kappa_hat * self.out_mass[exposed]

    def exposure_mass(self, g, frontier, exposed):
        """Baseline probability of activation of each exposed node (its 'information mass')."""
        m = np.zeros(g.n)
        for u in frontier:
            lo, hi = g.indptr[u], g.indptr[u + 1]
            np.add.at(m, g.indices[lo:hi], g.weights[lo:hi])
        return np.minimum(1.0, m[exposed])

    def factors(self, ep, detector, exposed, kappa_hat) -> np.ndarray | None:
        raise NotImplementedError


@register
class NoThrottle(Throttler):
    name = "none"

    def factors(self, ep, detector, exposed, kappa_hat):
        return None


@register
class HubThrottle(Throttler):
    name = "hub"

    def factors(self, ep, detector, exposed, kappa_hat):
        if len(exposed) == 0 or not self.gated(detector):
            return None
        self.rounds_active += 1
        g = ep.g
        h = self.harm_weight(g, exposed, kappa_hat)
        mass = self.exposure_mass(g, ep.frontier, exposed)
        order = np.argsort(-h, kind="stable")
        cum = np.cumsum(mass[order])
        k = int(np.searchsorted(cum, self.frac * cum[-1], side="left")) + 1
        rho = np.ones(g.n)
        rho[exposed[order[:k]]] = self.rho_min
        return rho


@register
class UniformThrottle(Throttler):
    name = "uniform"

    def factors(self, ep, detector, exposed, kappa_hat):
        if len(exposed) == 0 or not self.gated(detector):
            return None
        self.rounds_active += 1
        rho = np.ones(ep.g.n)
        rho[exposed] = self.rho_min
        return rho


@register
class RandomThrottle(Throttler):
    name = "random"

    def factors(self, ep, detector, exposed, kappa_hat):
        if len(exposed) == 0 or not self.gated(detector):
            return None
        self.rounds_active += 1
        g = ep.g
        mass = self.exposure_mass(g, ep.frontier, exposed)
        order = self.rng.permutation(len(exposed))
        cum = np.cumsum(mass[order])
        k = int(np.searchsorted(cum, self.frac * cum[-1], side="left")) + 1
        rho = np.ones(g.n)
        rho[exposed[order[:k]]] = self.rho_min
        return rho
