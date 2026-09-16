"""Epidemic simulators.

``simulate_sir`` is a continuous-time SIR simulator on a static graph that
reproduces the semantics of the event-driven simulator used by the source
detection benchmark of Sterchi et al. (recovery rate 1, per-edge infection
rate ``beta * w``, snapshot at time ``T``). Node states are encoded as
``0 = susceptible``, ``1 = infected at T``, ``2 = recovered by T``.
"""
from __future__ import annotations

import heapq
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .graphs import CSRGraph

S, I, R = 0, 1, 2


def simulate_sir(g: CSRGraph, source: int, beta: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """One SIR outbreak from ``source`` observed at time ``T``.

    Returns an int8 array of node states. Recovery times are Exp(1); infection
    attempts along an edge with weight ``w`` occur after an Exp(beta*w) delay.
    """
    n = g.n
    inf_time = np.full(n, np.inf)
    rec_time = np.full(n, np.inf)
    infected = np.zeros(n, dtype=bool)
    inf_time[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        now, me = heapq.heappop(heap)
        if infected[me] or now > inf_time[me]:
            continue  # stale entry
        infected[me] = True
        rec = now + rng.exponential(1.0)
        rec_time[me] = rec
        lo, hi = g.indptr[me], g.indptr[me + 1]
        nbrs = g.indices[lo:hi]
        if len(nbrs) == 0:
            continue
        delays = rng.exponential(1.0, size=len(nbrs)) / (beta * g.weights[lo:hi])
        t = now + delays
        for j in range(len(nbrs)):
            you = nbrs[j]
            tj = t[j]
            if (not infected[you]) and tj < rec and tj < inf_time[you] and tj <= T:
                inf_time[you] = tj
                heapq.heappush(heap, (tj, you))
    states = np.zeros(n, dtype=np.int8)
    states[infected & (rec_time <= T)] = R
    states[infected & (rec_time > T)] = I
    return states


def _sample_beta(beta: float, sigma: float, rng: np.random.Generator) -> float:
    if sigma <= 0:
        return beta
    return float(math.exp(math.log(beta) + sigma * rng.standard_normal()))


def _simulate_block(args):
    g, sources, seeds, sims_per_node, beta, sigma, T, sample_T = args
    out = np.zeros((len(sources) * sims_per_node, g.n), dtype=np.int8)
    labels = np.zeros(len(sources) * sims_per_node, dtype=np.int32)
    betas = np.zeros(len(sources) * sims_per_node, dtype=np.float32)
    k = 0
    for s, sd in zip(sources, seeds):
        rng = np.random.default_rng(sd)  # one stream per source node: independent of n_jobs
        for _ in range(sims_per_node):
            b = _sample_beta(beta, sigma, rng)
            Tobs = float(rng.uniform(0, 4 * T)) if sample_T else T
            out[k] = simulate_sir(g, int(s), b, Tobs, rng)
            labels[k] = s
            betas[k] = b
            k += 1
    return out, labels, betas


@dataclass
class SimDataset:
    states: np.ndarray  # (n_sims, n_nodes) int8
    sources: np.ndarray  # (n_sims,) int32
    betas: np.ndarray  # (n_sims,) float32

    def __len__(self):
        return len(self.sources)

    @property
    def outbreak_sizes(self) -> np.ndarray:
        return (self.states != S).sum(axis=1)


def simulate_dataset(g: CSRGraph, beta: float, sigma: float, T: float, sims_per_node: int, seed: int,
                     sample_T: bool = False, n_jobs: int = 1, sources=None) -> SimDataset:
    """Simulate ``sims_per_node`` outbreaks from every node (or from ``sources``).

    ``sigma`` is the standard deviation of a log-normal perturbation of
    ``beta`` drawn independently per outbreak (median preserved). Results are
    deterministic given ``seed`` regardless of ``n_jobs``.
    """
    if sources is None:
        sources = np.arange(g.n)
    sources = np.asarray(sources)
    seeds = np.random.SeedSequence(seed).spawn(len(sources))
    n_blocks = max(1, min(len(sources), n_jobs * 8)) if n_jobs > 1 else 1
    idx_blocks = np.array_split(np.arange(len(sources)), n_blocks)
    tasks = [(g, sources[ib], [seeds[i] for i in ib], sims_per_node, beta, sigma, T, sample_T) for ib in idx_blocks]
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_simulate_block, tasks))
    else:
        results = [_simulate_block(t) for t in tasks]
    states = np.concatenate([r[0] for r in results])
    labels = np.concatenate([r[1] for r in results])
    betas = np.concatenate([r[2] for r in results])
    return SimDataset(states=states, sources=labels, betas=betas)


# ---------------------------------------------------------------------------
# Full-trajectory simulation (for sequential / streaming observation settings)
# ---------------------------------------------------------------------------

def simulate_sir_times(g: CSRGraph, source: int, beta: float, T_max: float, rng: np.random.Generator):
    """Run one SIR outbreak up to ``T_max`` and return ``(inf_time, rec_time)``.

    Both arrays have shape ``(n,)`` with ``inf`` for nodes never infected
    (before ``T_max``). Snapshots at any ``T <= T_max`` can be derived with
    :func:`snapshot_at`, which makes the same trajectory observable at several
    times (the setting needed by sequential detection / intervention methods).
    """
    n = g.n
    inf_time = np.full(n, np.inf)
    rec_time = np.full(n, np.inf)
    done = np.zeros(n, dtype=bool)
    inf_time[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        now, me = heapq.heappop(heap)
        if done[me] or now > inf_time[me]:
            continue
        done[me] = True
        rec = now + rng.exponential(1.0)
        rec_time[me] = rec
        lo, hi = g.indptr[me], g.indptr[me + 1]
        nbrs = g.indices[lo:hi]
        if len(nbrs) == 0:
            continue
        t = now + rng.exponential(1.0, size=len(nbrs)) / (beta * g.weights[lo:hi])
        for j in range(len(nbrs)):
            you = nbrs[j]
            tj = t[j]
            if (not done[you]) and tj < rec and tj < inf_time[you] and tj <= T_max:
                inf_time[you] = tj
                heapq.heappush(heap, (tj, you))
    return inf_time, rec_time


def snapshot_at(inf_time: np.ndarray, rec_time: np.ndarray, T: float) -> np.ndarray:
    """States (0/1/2) at time ``T`` from a trajectory returned by :func:`simulate_sir_times`."""
    states = np.zeros(len(inf_time), dtype=np.int8)
    infected = inf_time <= T
    states[infected & (rec_time <= T)] = R
    states[infected & (rec_time > T)] = I
    return states


@dataclass
class TrajectoryDataset:
    inf_times: np.ndarray  # (n_sims, n) float32, inf when never infected
    rec_times: np.ndarray  # (n_sims, n)
    sources: np.ndarray
    betas: np.ndarray
    T_max: float

    def __len__(self):
        return len(self.sources)

    def snapshots(self, T: float) -> np.ndarray:
        return np.stack([snapshot_at(a, b, T) for a, b in zip(self.inf_times, self.rec_times)])


def _simulate_traj_block(args):
    g, sources, seeds, sims_per_node, beta, sigma, T_max = args
    m = len(sources) * sims_per_node
    inf = np.full((m, g.n), np.inf, dtype=np.float32)
    rec = np.full((m, g.n), np.inf, dtype=np.float32)
    labels = np.zeros(m, dtype=np.int32)
    betas = np.zeros(m, dtype=np.float32)
    k = 0
    for s, sd in zip(sources, seeds):
        rng = np.random.default_rng(sd)
        for _ in range(sims_per_node):
            b = _sample_beta(beta, sigma, rng)
            a, r = simulate_sir_times(g, int(s), b, T_max, rng)
            inf[k], rec[k], labels[k], betas[k] = a, r, s, b
            k += 1
    return inf, rec, labels, betas


def simulate_trajectories(g: CSRGraph, beta: float, sigma: float, T_max: float, sims_per_node: int, seed: int,
                          n_jobs: int = 1, sources=None) -> TrajectoryDataset:
    """Like :func:`simulate_dataset` but keeps infection/recovery times up to ``T_max``."""
    if sources is None:
        sources = np.arange(g.n)
    sources = np.asarray(sources)
    seeds = np.random.SeedSequence(seed).spawn(len(sources))
    n_blocks = max(1, min(len(sources), n_jobs * 8)) if n_jobs > 1 else 1
    idx_blocks = np.array_split(np.arange(len(sources)), n_blocks)
    tasks = [(g, sources[ib], [seeds[i] for i in ib], sims_per_node, beta, sigma, T_max) for ib in idx_blocks]
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_simulate_traj_block, tasks))
    else:
        results = [_simulate_traj_block(t) for t in tasks]
    return TrajectoryDataset(inf_times=np.concatenate([r[0] for r in results]), rec_times=np.concatenate([r[1] for r in results]),
                             sources=np.concatenate([r[2] for r in results]), betas=np.concatenate([r[3] for r in results]), T_max=T_max)
