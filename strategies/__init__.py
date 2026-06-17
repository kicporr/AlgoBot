"""Trading strategy implementations."""

from .base import BaseStrategy, Signal

# Core strategies (always available)
from .mtf_macd import MTF_MACD_Elder
from .mean_reversion import MeanReversion

# Meta-labeling (placeholder — not yet implemented)
try:
    from .meta_labeling import MetaLabeler
except ImportError:
    MetaLabeler = None


__all__ = [
    "BaseStrategy", "Signal",
    "MTF_MACD_Elder", "MeanReversion",
    "MetaLabeler",
]
