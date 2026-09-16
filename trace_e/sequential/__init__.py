"""Sequential detect-and-contain of harmful cascades (AVID) and its baselines.

See ``docs/METHOD.md``. Modules:

* ``simulator``  - live-edge coupled, round-by-round IC simulator with interventions
* ``detectors``  - e-process (AVID), SPRT, CUSUM, size/growth thresholds, logistic classifier
* ``containers`` - one-shot blockers and the adaptive frontier container
* ``policy``     - runs one episode interactively and records harm metrics
"""
from .detectors import DETECTORS, get_detector  # noqa: F401
from .containers import CONTAINERS, get_container  # noqa: F401
