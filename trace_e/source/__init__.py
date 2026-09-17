"""Source detection ("patient zero") methods.

Every method subclasses :class:`SourceDetector` and is registered by name in
:data:`REGISTRY` so the runner can select methods from the command line.
"""
from .base import SourceDetector, Context, REGISTRY, register, get_detector  # noqa: F401
from . import centrality, netsleuth, dmp, simulation_based, exact_sir  # noqa: F401,E402

try:  # torch is optional
    from . import gnn  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from . import proposed  # noqa: F401,E402
