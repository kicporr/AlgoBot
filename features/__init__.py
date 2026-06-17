"""Feature engineering pipeline.

Components:
    FeatureEngine       — orchestrates live + bulk feature computation
    IndicatorCalculator — pure pandas/numpy TA indicators (no external deps)
    DerivedFeatures     — candlestick patterns, divergences, structure analysis
"""

from .engine import FeatureEngine
from .indicators import IndicatorCalculator
from .derived import DerivedFeatures

__all__ = [
    "FeatureEngine",
    "IndicatorCalculator",
    "DerivedFeatures",
]
