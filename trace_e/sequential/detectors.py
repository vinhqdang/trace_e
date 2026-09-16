"""Sequential detectors. Each consumes one round of exposure/outcome data.

``update(log1mq, outcomes)`` receives ``log1mq`` of shape ``(K, X)`` = log(1-q)
for the detector's kappa grid (row 0 = baseline) and the 0/1 outcomes of the
X exposed nodes, and returns True when the detector fires. ``statistic`` is
the running test statistic used for threshold calibration of the baselines.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

DETECTORS: dict[str, type] = {}


def register(cls):
    DETECTORS[cls.name] = cls
    return cls


def get_detector(name: str, **kw):
    if name not in DETECTORS:
        raise KeyError(f"unknown detector '{name}'. Known: {sorted(DETECTORS)}")
    return DETECTORS[name](**kw)


def geometric_grid(kmin: float, kmax: float, eta: float = 0.25) -> np.ndarray:
    J = int(np.ceil(np.log(kmax / kmin) / np.log1p(eta))) + 1
    return np.exp(np.linspace(np.log(kmin), np.log(kmax), J))


def round_log_lr(log1mq: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """log Lambda^kappa_t for every kappa row (row 0 is the baseline)."""
    if log1mq.shape[1] == 0:
        return np.zeros(log1mq.shape[0])
    l0 = log1mq[0]
    q0 = -np.expm1(l0)
    q = -np.expm1(log1mq)
    with np.errstate(divide="ignore", invalid="ignore"):
        la = np.where(outcomes[None, :] == 1, np.log(np.maximum(q, 1e-300)) - np.log(np.maximum(q0, 1e-300))[None, :],
                      log1mq - l0[None, :])
    return la.sum(axis=1)


class Detector:
    name = "base"
    needs_calibration = False  # threshold set from benign simulations
    kappas: np.ndarray = np.array([1.0])  # grid rows requested from exposure()

    def __init__(self, alpha: float = 0.05, **kw):
        self.alpha = alpha
        self.threshold = None
        self.reset()

    def reset(self):
        self.t = 0
        self.size = 0
        self.statistic = 0.0
        self.fired = False

    def kappa_hat(self) -> float:
        return 1.0

    def update(self, log1mq, outcomes, size_before: int) -> bool:
        raise NotImplementedError


@register
class EProcess(Detector):
    """AVID evidence process: mixture over a kappa grid of IC likelihood-ratio martingales."""

    name = "eprocess"

    def __init__(self, alpha=0.05, kappa_min=1.25, kappa_max=8.0, eta=0.25, kappa_null=1.0, **kw):
        grid = geometric_grid(kappa_min, kappa_max, eta)
        self.grid = grid
        self.kappas = np.concatenate([[kappa_null], grid])  # row 0 = (possibly inflated) null
        self.log_w = np.full(len(grid), -np.log(len(grid)))
        super().__init__(alpha=alpha)
        self.threshold = np.log(1.0 / alpha)

    def reset(self):
        super().reset()
        self.log_L = np.zeros(len(self.grid))
        self.log_E = 0.0

    def kappa_hat(self):
        post = np.exp(self.log_w + self.log_L - logsumexp(self.log_w + self.log_L))
        return float(post @ self.grid)

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        self.log_L += round_log_lr(log1mq, outcomes)[1:]
        self.log_E = float(logsumexp(self.log_w + self.log_L))
        self.statistic = self.log_E
        if self.log_E >= self.threshold:
            self.fired = True
        return self.fired


@register
class SPRT(Detector):
    """One-sided SPRT with a known alternative kappa (single-component e-process)."""

    name = "sprt"

    def __init__(self, alpha=0.05, kappa=2.0, kappa_null=1.0, **kw):
        self.kappas = np.array([kappa_null, kappa])
        super().__init__(alpha=alpha)
        self.threshold = np.log(1.0 / alpha)

    def reset(self):
        super().reset()
        self.log_L = 0.0

    def kappa_hat(self):
        return float(self.kappas[1])

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        self.log_L += float(round_log_lr(log1mq, outcomes)[1])
        self.statistic = self.log_L
        if self.log_L >= self.threshold:
            self.fired = True
        return self.fired


@register
class CUSUM(Detector):
    """CUSUM on the per-round log-likelihood ratio; threshold calibrated on benign cascades."""

    name = "cusum"
    needs_calibration = True

    def __init__(self, alpha=0.05, kappa=2.0, **kw):
        self.kappas = np.array([1.0, kappa])
        super().__init__(alpha=alpha)

    def reset(self):
        super().reset()
        self.S = 0.0
        self.max_stat = 0.0

    def kappa_hat(self):
        return float(self.kappas[1])

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        self.S = max(0.0, self.S + float(round_log_lr(log1mq, outcomes)[1]))
        self.statistic = self.S
        if self.threshold is not None and self.S >= self.threshold:
            self.fired = True
        return self.fired


@register
class SizeThreshold(Detector):
    """Fire when the cascade size exceeds a threshold calibrated on benign cascades."""

    name = "size"
    needs_calibration = True

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        self.size = size_before + int(outcomes.sum())
        self.statistic = float(self.size)
        if self.threshold is not None and self.size > self.threshold:
            self.fired = True
        return self.fired


@register
class GrowthThreshold(Detector):
    """Fire when the number of new activations in a round exceeds a calibrated threshold."""

    name = "growth"
    needs_calibration = True

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        new = int(outcomes.sum())
        self.statistic = float(new)
        if self.threshold is not None and new > self.threshold:
            self.fired = True
        return self.fired


@register
class ExcessRatio(Detector):
    """Fire when cumulative observed / expected-under-baseline activations exceeds a calibrated threshold
    (a moment-based, guarantee-free version of the likelihood ratio)."""

    name = "excess"
    needs_calibration = True

    def reset(self):
        super().reset()
        self.obs = 0.0
        self.exp = 0.0

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        self.obs += float(outcomes.sum())
        self.exp += float((-np.expm1(log1mq[0])).sum())
        self.statistic = (self.obs - self.exp) / np.sqrt(max(self.exp, 1.0))
        if self.threshold is not None and self.statistic > self.threshold:
            self.fired = True
        return self.fired


@register
class Never(Detector):
    name = "never"

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        return False


@register
class Immediate(Detector):
    """Fires on the first round for every cascade (timing oracle, no false-alarm control)."""

    name = "immediate"

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        self.fired = True
        return True


@register
class LogisticEarly(Detector):
    """Learned early-detection classifier on per-round summary features (no guarantee).

    Features: log size, log(1+new), round index, cumulative excess ratio, mean
    log-odds of baseline exposure. Trained on labelled benign/harmful cascades
    by the runner; the decision threshold on the max posterior is calibrated
    on benign cascades like the other threshold detectors.
    """

    name = "logistic"
    needs_calibration = True
    model = None  # sklearn estimator set by the runner

    def reset(self):
        super().reset()
        self.obs = 0.0
        self.exp = 0.0
        self.max_p = 0.0

    def features(self, log1mq, outcomes, size_before):
        new = float(outcomes.sum())
        self.obs += new
        self.exp += float((-np.expm1(log1mq[0])).sum())
        return np.array([np.log1p(size_before + new), np.log1p(new), self.t,
                         (self.obs - self.exp) / np.sqrt(max(self.exp, 1.0)), np.log1p(log1mq.shape[1])])

    def update(self, log1mq, outcomes, size_before):
        self.t += 1
        f = self.features(log1mq, outcomes, size_before)
        if self.model is not None:
            p = float(self.model.predict_proba(f[None])[0, 1])
            self.max_p = max(self.max_p, p)
        self.statistic = self.max_p
        if self.threshold is not None and self.max_p > self.threshold:
            self.fired = True
        return self.fired
