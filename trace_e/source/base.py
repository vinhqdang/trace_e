from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import networkx as nx
import numpy as np

from ..graphs import CSRGraph, to_csr
from ..metrics import bfs_distances
from ..simulate import SimDataset


@dataclass
class Context:
    """Everything a detector may need about the graph and the epidemic model."""

    G: nx.Graph
    g: CSRGraph
    beta: float
    T: float
    nu: float = 1.0
    sigma: float = 0.0
    n_jobs: int = 1
    seed: int = 0
    extra: dict = field(default_factory=dict)
    _dist_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_graph(cls, G: nx.Graph, **kw) -> "Context":
        return cls(G=G, g=to_csr(G), **kw)

    @property
    def n(self) -> int:
        return self.g.n

    def distances(self, u: int) -> np.ndarray:
        if u not in self._dist_cache:
            if len(self._dist_cache) > 20000:
                self._dist_cache.clear()
            self._dist_cache[u] = bfs_distances(self.g, u)
        return self._dist_cache[u]


class SourceDetector:
    """Interface: ``fit`` (optional, may use simulated training data) then ``score``."""

    name: str = "base"
    probabilistic: bool = False  # True when score() returns log-probabilities
    needs_training_sims: bool = False

    def __init__(self, ctx: Context, **params):
        self.ctx = ctx
        self.params = params

    def fit(self, train: SimDataset | None = None) -> None:
        return None

    def score(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_batch(self, states: np.ndarray) -> np.ndarray:
        return np.stack([self.score(s) for s in states])

    @staticmethod
    def restrict(scores: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Exclude susceptible nodes: the source must have been infected."""
        scores = np.asarray(scores, dtype=np.float64).copy()
        scores[states == 0] = -np.inf
        return scores


REGISTRY: dict[str, Type[SourceDetector]] = {}


def register(cls: Type[SourceDetector]) -> Type[SourceDetector]:
    REGISTRY[cls.name] = cls
    return cls


def get_detector(name: str, ctx: Context, **params) -> SourceDetector:
    if name not in REGISTRY:
        raise KeyError(f"unknown source detector '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](ctx, **params)
