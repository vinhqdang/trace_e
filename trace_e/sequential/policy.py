"""Run one detect-and-contain episode interactively."""
from __future__ import annotations

import time

import numpy as np

from .simulator import Episode, exposure


def run_episode(ep: Episode, detector, container, budget: int, max_rounds: int = 60) -> dict:
    """Advance ``ep`` round by round, feeding the detector; after it fires, let the container act.

    Returns per-episode metrics. ``ep`` must be freshly constructed or reset.
    """
    g = ep.g
    detector.reset()
    container.reset(budget)
    t_det = 0.0
    t_con = 0.0
    alarm_round = None
    harm_at_alarm = None
    kappa_hat = 1.0
    n_intervened = 0
    rounds = 0
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
        new = ep.step()
        rounds += 1
        if not detector.fired:
            t0 = time.perf_counter()
            exposed, log1mq = exposure(g, frontier, inactive, detector.kappas)
            outcomes = np.zeros(len(exposed), dtype=np.int8)
            if len(new):
                outcomes[np.isin(exposed, new)] = 1
            fired = detector.update(log1mq, outcomes, size_before)
            t_det += time.perf_counter() - t0
            if fired:
                alarm_round = rounds
                harm_at_alarm = ep.harm
                kappa_hat = detector.kappa_hat()
    return {
        "alarm_round": alarm_round,
        "harm_at_alarm": harm_at_alarm,
        "harm_final": ep.harm,
        "harm_counterfactual": ep.counterfactual_harm(),
        "n_intervened": n_intervened,
        "kappa_hat": kappa_hat if alarm_round is not None else None,
        "rounds": rounds,
        "t_detect": t_det,
        "t_contain": t_con,
        "statistic": float(detector.statistic),
    }


def max_statistic(ep: Episode, detector, max_rounds: int = 60) -> float:
    """Largest statistic value over the life of an unintervened cascade (for calibration)."""
    g = ep.g
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
