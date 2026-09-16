"""Graph loading utilities.

Empirical networks are stored as whitespace-separated edge lists in
``data/networks/<name>.csv`` with integer node ids ``0..N-1``. Optional edge
weights live in ``data/networks/<name>_weights.csv`` (one integer per edge
line, same order). Synthetic graphs are generated with networkx.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "networks")

EMPIRICAL = {
    "karate": "Zachary karate club (34 nodes)",
    "iceland": "Iceland sexual-contact network (75 nodes)",
    "dolphin": "Dolphin social network (62 nodes)",
    "fraternity": "Fraternity network (58 nodes)",
    "workplace": "SocioPatterns workplace, static projection (92 nodes)",
    "highschool2013": "SocioPatterns high school 2013, static projection (327 nodes)",
    "highschool2010": "SocioPatterns high school 2010, static projection",
    "conference": "SocioPatterns conference, static projection",
    "powergrid": "Western US power grid (4941 nodes)",
    "airtraffic": "Country-level air traffic network (weighted)",
}


@dataclass
class CSRGraph:
    """Compressed adjacency used by the simulators and the fast baselines."""

    n: int
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray  # float, same length as indices

    def neighbors(self, u: int) -> np.ndarray:
        return self.indices[self.indptr[u]: self.indptr[u + 1]]

    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)


def to_csr(G: nx.Graph) -> CSRGraph:
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight="weight", dtype=np.float64, format="csr")
    A.sort_indices()
    return CSRGraph(n=n, indptr=A.indptr.astype(np.int64), indices=A.indices.astype(np.int64), weights=A.data.astype(np.float64))


def adjacency(G: nx.Graph, weighted: bool = False) -> sp.csr_matrix:
    n = G.number_of_nodes()
    return nx.to_scipy_sparse_array(G, nodelist=range(n), weight="weight" if weighted else None, dtype=np.float64, format="csr")


def load_empirical(name: str, weighted: bool = False, data_dir: str = DATA_DIR) -> nx.Graph:
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unknown network '{name}' (expected {path})")
    edges = np.loadtxt(path, dtype=np.int64, ndmin=2)
    n = int(edges.max()) + 1
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if weighted:
        wpath = os.path.join(data_dir, f"{name}_weights.csv")
        w = np.loadtxt(wpath, dtype=np.float64, ndmin=1)
        G.add_weighted_edges_from((int(u), int(v), float(x)) for (u, v), x in zip(edges, w))
    else:
        G.add_edges_from((int(u), int(v)) for u, v in edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_synthetic(kind: str, n: int, seed: int = 0, **kw) -> nx.Graph:
    rng = np.random.default_rng(seed)
    if kind == "er":
        p = kw.get("p", 4.0 / n)
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(2**31)))
    elif kind == "ba":
        G = nx.barabasi_albert_graph(n, kw.get("m", 2), seed=int(rng.integers(2**31)))
    elif kind == "ws":
        G = nx.watts_strogatz_graph(n, kw.get("k", 4), kw.get("p", 0.1), seed=int(rng.integers(2**31)))
    elif kind == "tree":
        G = nx.random_labeled_tree(n, seed=int(rng.integers(2**31))) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=int(rng.integers(2**31)))
    else:
        raise ValueError(f"unknown synthetic graph kind '{kind}'")
    return nx.convert_node_labels_to_integers(G)


def load_network(spec: str, seed: int = 0) -> nx.Graph:
    """Load a network from a spec string.

    ``"karate"`` loads an empirical network; ``"airtraffic:w"`` loads it with
    weights; ``"er:200"`` / ``"ba:500:m=3"`` / ``"ws:100:k=4:p=0.1"`` build a
    synthetic graph.
    """
    parts = spec.split(":")
    name = parts[0]
    if name in ("er", "ba", "ws", "tree"):
        n = int(parts[1])
        kw = {}
        for p in parts[2:]:
            k, v = p.split("=")
            kw[k] = float(v) if "." in v else int(v)
        return load_synthetic(name, n, seed=seed, **kw)
    weighted = len(parts) > 1 and parts[1] == "w"
    return load_empirical(name, weighted=weighted)


def graph_stats(G: nx.Graph) -> dict:
    cc = max(nx.connected_components(G), key=len)
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "avg_degree": float(np.mean([d for _, d in G.degree()])),
        "n_components": nx.number_connected_components(G),
        "lcc_size": len(cc),
    }
