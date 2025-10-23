from .sopt import sopt, spot
from .pawl import pawl
from .usot import sliced_unbalanced_ot, unbalanced_sliced_ot
from .sw import sliced_wasserstein
from .ulight_ot import ULightOT

# Optional import for uot_fm (requires JAX, optax, equinox)
try:
    from . import uot_fm
    _has_uot_fm = True
except ImportError:
    uot_fm = None
    _has_uot_fm = False

__all__ = [
    'sopt',
    'spot',
    'pawl',
    'sliced_unbalanced_ot',
    'unbalanced_sliced_ot',
    'sliced_wasserstein',
    'ULightOT',
]

if _has_uot_fm:
    __all__.append('uot_fm')

