"""Slot for the proposed method.

Replace the body of :class:`Proposed` (or add new registered classes) with the
algorithm under development. Anything registered here is selectable from the
runner with ``--methods proposed``. The class below is a placeholder that
combines Jordan-center ranking with a simulation-based likelihood so the
plumbing (fit on training simulations, batch scoring, probabilistic metrics)
is exercised end to end.
"""
from __future__ import annotations

import numpy as np

from .base import SourceDetector, register
from .simulation_based import MonteCarloMeanField
from .centrality import JordanCenter


@register
class Proposed(SourceDetector):
    name = "proposed"
    probabilistic = True
    needs_training_sims = True

    def fit(self, train=None):
        self.mcs = MonteCarloMeanField(self.ctx)
        self.mcs.fit(train)
        self.jordan = JordanCenter(self.ctx)

    def score(self, states):
        # placeholder: mean-field log-likelihood plus a mild eccentricity prior
        s = self.mcs.score(states)
        j = self.jordan.score(states)
        finite = np.isfinite(j)
        if finite.any():
            s[finite] += 0.5 * (j[finite] - j[finite].max())
        return s
