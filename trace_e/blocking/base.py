from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import networkx as nx
import numpy as np

from ..graphs import CSRGraph, to_csr
from .cascade import set_edge_probabilities


@dataclass
class BlockingContext:
    G: nx.Graph
    g: CSRGraph  # weights = IC activation probabilities
    mode: str = "block"  # "block" (vertex removal) or "counter" (counter-seeding, Budak)
    good_delay: int = 0
    seed: int = 0
    n_jobs: int = 1
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_graph(cls, G: nx.Graph, prob_model: str = "wc", p: float = 0.1, **kw) -> "BlockingContext":
        g = set_edge_probabilities(to_csr(G), prob_model, p, seed=kw.get("seed", 0))
        return cls(G=G, g=g, **kw)

    @property
    def n(self) -> int:
        return self.g.n


class Blocker:
    """Interface: ``select(bad_seeds, budget) -> list of node ids``."""

    name = "base"

    def __init__(self, ctx: BlockingContext, **params):
        self.ctx = ctx
        self.params = params

    def prepare(self) -> None:  # one-off precomputation independent of the instance
        return None

    def select(self, bad_seeds: np.ndarray, budget: int) -> list[int]:
        """Return an ordered list of ``budget`` nodes; prefixes must be valid solutions for smaller budgets."""
        raise NotImplementedError


REGISTRY: dict[str, Type[Blocker]] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get_blocker(name: str, ctx: BlockingContext, **params) -> Blocker:
    if name not in REGISTRY:
        raise KeyError(f"unknown blocker '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](ctx, **params)
