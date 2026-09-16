"""Topology-based single-snapshot estimators (no training data needed)."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

from ..metrics import bfs_distances
from .base import SourceDetector, register


def _infected_nodes(states: np.ndarray) -> np.ndarray:
    return np.flatnonzero(states != 0)


@register
class RandomDetector(SourceDetector):
    """Uniform random guess among infected/recovered nodes."""

    name = "random"
    probabilistic = True

    def score(self, states):
        inf = _infected_nodes(states)
        s = np.full(self.ctx.n, -np.inf)
        s[inf] = -math.log(len(inf))
        return s


@register
class DegreeCenter(SourceDetector):
    """Degree within the infected subgraph."""

    name = "degree"

    def score(self, states):
        inf = _infected_nodes(states)
        mask = states != 0
        s = np.full(self.ctx.n, -np.inf)
        g = self.ctx.g
        for u in inf:
            s[u] = mask[g.neighbors(u)].sum()
        return s


@register
class JordanCenter(SourceDetector):
    """Jordan center of the infected subgraph (Zhu & Ying, 2016): minimum eccentricity."""

    name = "jordan"

    def score(self, states):
        inf = _infected_nodes(states)
        mask = states != 0
        s = np.full(self.ctx.n, -np.inf)
        g = self.ctx.g
        for u in inf:
            d = bfs_distances(g, int(u), mask)
            reach = d[inf]
            ecc = reach[reach >= 0].max()
            unreached = (reach < 0).sum()  # infected subgraph may be disconnected at small T
            s[u] = -(ecc + 1000 * unreached)
        return s


@register
class DistanceCenter(SourceDetector):
    """Distance centrality (Comin & da Fontoura Costa, 2011): minimum total distance to infected nodes."""

    name = "distance"

    def score(self, states):
        inf = _infected_nodes(states)
        mask = states != 0
        s = np.full(self.ctx.n, -np.inf)
        g = self.ctx.g
        for u in inf:
            d = bfs_distances(g, int(u), mask)
            reach = d[inf]
            s[u] = -(reach[reach >= 0].sum() + 1000 * (reach < 0).sum())
        return s


@register
class ClosenessCenter(SourceDetector):
    """networkx closeness centrality of the infected subgraph."""

    name = "closeness"

    def score(self, states):
        inf = _infected_nodes(states)
        sub = self.ctx.G.subgraph(inf.tolist())
        c = nx.closeness_centrality(sub)
        s = np.full(self.ctx.n, -np.inf)
        for u, v in c.items():
            s[u] = v
        return s


@register
class BetweennessCenter(SourceDetector):
    """networkx betweenness centrality of the infected subgraph."""

    name = "betweenness"

    def score(self, states):
        inf = _infected_nodes(states)
        sub = self.ctx.G.subgraph(inf.tolist())
        c = nx.betweenness_centrality(sub)
        s = np.full(self.ctx.n, -np.inf)
        for u, v in c.items():
            s[u] = v
        return s


@register
class RumorCentrality(SourceDetector):
    """Rumor centrality (Shah & Zaman, 2011) on the BFS tree rooted at each candidate.

    R(v) = n! / prod_u T_u^v where T_u^v is the subtree size of u in the BFS
    spanning tree of the infected subgraph rooted at v. Returned as log R(v).
    For an SI process on a regular tree this is the ML estimator; on general
    graphs the BFS-tree heuristic from the original paper is used.
    """

    name = "rumor"

    def score(self, states):
        inf = _infected_nodes(states)
        mask = states != 0
        g = self.ctx.g
        s = np.full(self.ctx.n, -np.inf)
        n_inf = len(inf)
        log_nfact = math.lgamma(n_inf + 1)
        for v in inf:
            # BFS tree rooted at v inside the infected subgraph
            parent = {int(v): -1}
            order = [int(v)]
            i = 0
            while i < len(order):
                u = order[i]
                i += 1
                for w in g.neighbors(u):
                    w = int(w)
                    if mask[w] and w not in parent:
                        parent[w] = u
                        order.append(w)
            if len(order) < n_inf:
                s[v] = -np.inf  # cannot reach all infected nodes
                continue
            size = {u: 1 for u in order}
            for u in reversed(order[1:]):
                size[parent[u]] += size[u]
            s[v] = log_nfact - sum(math.log(size[u]) for u in order)
        return s
