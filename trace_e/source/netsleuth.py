"""NetSleuth (Prakash, Vreeken & Faloutsos, ICDM 2012 / KAIS 2014).

The single-seed ranking uses the smallest eigenvector of the infected-subgraph
Laplacian ``L_I = D_I - A_I`` where ``D_I`` counts *all* graph neighbours
(so the frontier is accounted for). The multi-seed variant greedily adds
seeds while the two-part MDL description length decreases.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import SourceDetector, register


def _laplacian_submatrix(A: sp.csr_matrix, inf: np.ndarray) -> np.ndarray:
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_I = A[inf][:, inf].toarray()
    return np.diag(deg[inf]) - A_I


def smallest_eigvec(L: np.ndarray) -> np.ndarray:
    if L.shape[0] <= 3:
        w, v = np.linalg.eigh(L)
        return np.abs(v[:, 0])
    if L.shape[0] < 200:
        w, v = np.linalg.eigh(L)
        return np.abs(v[:, 0])
    w, v = spla.eigsh(sp.csr_matrix(L), k=1, which="SA")
    return np.abs(v[:, 0])


@register
class NetSleuth(SourceDetector):
    name = "netsleuth"

    def fit(self, train=None):
        from ..graphs import adjacency
        self.A = adjacency(self.ctx.G)

    def score(self, states):
        if not hasattr(self, "A"):
            self.fit()
        inf = np.flatnonzero(states != 0)
        s = np.full(self.ctx.n, -np.inf)
        if len(inf) == 1:
            s[inf[0]] = 0.0
            return s
        L = _laplacian_submatrix(self.A, inf)
        vec = smallest_eigvec(L)
        s[inf] = vec
        return s
