"""Slot for the proposed intervention method.

The placeholder below is greedy with a smaller sample pool; replace it with
the method under development and select it with ``--methods proposed``.
"""
from __future__ import annotations

from .base import register
from .greedy import GreedyBlocker


@register
class ProposedBlocker(GreedyBlocker):
    name = "proposed"

    def __init__(self, ctx, **params):
        params.setdefault("n_samples", 100)
        super().__init__(ctx, **params)
