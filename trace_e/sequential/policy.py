"""Run one detect-and-contain episode interactively."""
from __future__ import annotations

import time

import numpy as np

from .simulator import Episode, exposure, exposed_nodes


def run_episode(ep: Episode, detector, container, budget: int, max_rounds: int = 60, g_det=None, throttler=None) -> dict:
    """Advance ``ep`` round by round, feeding the detector; after it fires, let the container act.

    ``g_det`` is the graph (with baseline probabilities) the detector believes
    in; it defaults to the true one and differs only in misspecification
    experiments. Returns per-episode metrics.
    """
    g = ep.g if g_det is None else g_det
    detector.reset()
    container.reset(budget)
    if throttler is not None:
        throttler.reset()
    t_det = 0.0
    t_con = 0.0
    alarm_round = None
    harm_at_alarm = None
    harmw_at_alarm = None
    out_mass = np.array([ep.g.weights[ep.g.indptr[v]: ep.g.indptr[v + 1]].sum() for v in range(ep.g.n)])
    kappa_hat = 1.0
    n_intervened = 0
    rounds = 0
    throttled_rounds = 0
    throttled_mass = 0.0
    while ep.alive and rounds < max_rounds:
        if detector.fired:
            t0 = time.perf_counter()
            nodes = container.act(ep, kappa_hat)
            applied = ep.intervene(nodes)
            n_intervened += len(applied)
            t_con += time.perf_counter() - t0
        inactive = ~(ep.active | ep.good | ep.blocked)
        frontier = ep.frontier
        size_before = ep.harm
        rho = None
        if throttler is not None and not detector.fired:
            exp_nodes = exposed_nodes(ep.g, frontier, inactive)
            rho = throttler.factors(ep, detector, exp_nodes, detector.kappa_hat())
            if rho is not None:
                throttled_rounds += 1
                throttled_mass += float((1.0 - rho[exp_nodes]).sum())
        new = ep.step(rho)
        rounds += 1
        if not detector.fired:
            t0 = time.perf_counter()
            exposed, log1mq = exposure(g, frontier, inactive, detector.kappas, rho)
            outcomes = np.zeros(len(exposed), dtype=np.int8)
            if len(new):
                outcomes[np.isin(exposed, new)] = 1
            fired = detector.update(log1mq, outcomes, size_before)
            t_det += time.perf_counter() - t0
            if fired:
                alarm_round = rounds
                harm_at_alarm = ep.harm
                # influence-weighted harm W_tau with h(v) = 1 + kappa * expected children (true kappa, for evaluation)
                nonseed = ep.active.copy()
                nonseed[ep.seeds] = False
                harmw_at_alarm = float((1.0 + ep.kappa * out_mass[nonseed]).sum())
                kappa_hat = detector.kappa_hat()
    return {
        "alarm_round": alarm_round,
        "harm_at_alarm": harm_at_alarm,
        "harmw_at_alarm": harmw_at_alarm,
        "harm_final": ep.harm,
        "harm_counterfactual": ep.counterfactual_harm(max_rounds),
        "n_intervened": n_intervened,
        "kappa_hat": kappa_hat if alarm_round is not None else None,
        "rounds": rounds,
        "t_detect": t_det,
        "t_contain": t_con,
        "statistic": float(detector.statistic),
        "throttled_rounds": throttled_rounds,
        "throttled_nodes": throttled_mass,
    }


def max_statistic(ep: Episode, detector, max_rounds: int = 60, g_det=None) -> float:
    """Largest statistic value over the life of an unintervened cascade (for calibration)."""
    g = ep.g if g_det is None else g_det
    detector.reset()
    m = -np.inf
    rounds = 0
    while ep.alive and rounds < max_rounds:
        inactive = ~(ep.active | ep.good | ep.blocked)
        frontier = ep.frontier
        size_before = ep.harm
        new = ep.step()
        rounds += 1
        exposed, log1mq = exposure(g, frontier, inactive, detector.kappas)
        outcomes = np.zeros(len(exposed), dtype=np.int8)
        if len(new):
            outcomes[np.isin(exposed, new)] = 1
        detector.threshold = None if detector.needs_calibration else detector.threshold
        detector.update(log1mq, outcomes, size_before)
        m = max(m, float(detector.statistic))
    return m
