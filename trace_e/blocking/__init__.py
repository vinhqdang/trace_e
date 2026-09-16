"""Misinformation blocking / influence limitation (Problem B).

Cascade models live in :mod:`cascade`; intervention strategies subclass
:class:`Blocker` and register by name in :data:`REGISTRY`.
"""
from .base import Blocker, BlockingContext, REGISTRY, register, get_blocker  # noqa: F401
from . import heuristics, greedy, proposed  # noqa: F401,E402
