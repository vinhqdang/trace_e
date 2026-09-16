"""LAZY-AG: AdvancedGreedy with lazily updated dominator trees.

AdvancedGreedy recomputes theta dominator trees after every pick, O(b theta m
alpha). Removing a blocker v from a live-edge graph (i) deletes v's dominator
subtree D(v) from the reachable set and (ii) can change the immediate
dominators of other nodes only if some of their seed-paths went through v.
LAZY-AG applies (i) exactly and ignores (ii): the subtree size of every
ancestor of v is reduced by |D(v)| and the nodes of D(v) are marked saved.
Per pick this costs O(theta * depth) for the ancestor walks plus O(theta * k)
to refresh the alive mask, all vectorised, instead of theta dominator
computations, so theta can be one to two orders of magnitude larger at the
same wall-clock. The gains it uses are lower bounds on the exact conditional
gains (ignoring (ii) can only miss savings that new dominators would
create), and an optional exact refresh every ``refresh_every`` picks bounds
the drift.
"""
from __future__ import annotations

import time

import numpy as np

from ..graphs import CSRGraph
from ..sequential.dominators import reachable_dag, dominator_subtree_sizes, _idoms_from_preds


class LazySample:
    def __init__(self, g: CSRGraph, live: np.ndarray, seeds, forbidden: np.ndarray):
        order, preds, index = reachable_dag(g, live, seeds, forbidden)
        self.nodes = np.array(order, dtype=np.int64)
        self.index = index
        self.k = len(order)
        if self.k:
            idom = _idoms_from_preds(order, preds)
            self.idom = np.array([(-1 if d is None or d == -1 else d) for d in idom], dtype=np.int64)
            self.sizes = dominator_subtree_sizes(order, preds).astype(np.int64)
        else:
            self.idom = np.zeros(0, dtype=np.int64)
            self.sizes = np.zeros(0, dtype=np.int64)
        self.alive = np.ones(self.k, dtype=bool)
        self.is_seed = np.zeros(self.k, dtype=bool)
        for s in seeds:
            if index[s] >= 0:
                self.is_seed[index[s]] = True

    def remove(self, v: int):
        pos = self.index[v]
        if pos < 0 or not self.alive[pos]:
            return
        d = int(self.sizes[pos])
        # ancestors lose v's subtree
        a = int(self.idom[pos])
        while a >= 0:
            self.sizes[a] -= d
            a = int(self.idom[a])
        # v's subtree is saved: descendants are exactly the alive nodes whose ancestor chain hits pos
        self.alive[pos] = False
        # positions are in reverse postorder: a node's dominator has a smaller position, so one forward pass suffices
        for q in range(pos + 1, self.k):
            if self.alive[q]:
                dq = self.idom[q]
                if dq >= 0 and not self.alive[dq]:
                    self.alive[q] = False


def lazy_greedy(g: CSRGraph, lives, seeds, budget: int, forbidden=None, refresh_every: int = 0, log=None):
    seeds = [int(s) for s in seeds]
    forbidden = np.zeros(g.n, dtype=bool) if forbidden is None else forbidden.copy()
    t0 = time.perf_counter()
    samples = [LazySample(g, lv, seeds, forbidden) for lv in lives]
    n = g.n
    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[seeds] = True
    B = []
    for step in range(budget):
        if refresh_every and step > 0 and step % refresh_every == 0:
            samples = [LazySample(g, lv, seeds, forbidden) for lv in lives]
        acc = np.zeros(n)
        for smp in samples:
            if smp.k:
                m = smp.alive & ~smp.is_seed
                acc[smp.nodes[m]] += smp.sizes[m]
        acc[seed_mask] = 0
        acc[forbidden] = 0
        v = int(np.argmax(acc))
        if acc[v] <= 0:
            break
        B.append(v)
        forbidden[v] = True
        for smp in samples:
            smp.remove(v)
        if log:
            log(f"step {step}: {v} gain {acc[v] / len(samples):.2f}")
    return B, {"time_s": time.perf_counter() - t0}
