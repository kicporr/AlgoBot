"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import pandas as pd


class Signal(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class BaseStrategy(ABC):
    """All strategies must implement this interface."""
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def on_candle(self, candle: dict, features: pd.Series) -> Signal:
        """Called every primary timeframe candle close. Returns trading signal."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...
    
    def on_higher_tf_candle(self, candle: dict, timeframe: str):
        """Optional: called when a higher-timeframe candle closes."""
        pass
    
    def retrain(self, historical_data: pd.DataFrame):
        """Optional: retrain ML model on new data."""
        pass
