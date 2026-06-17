"""Multi-Timeframe MACD with Elder Triple Screen Filter.

Strategy logic:
    1. D1 trend filter (Elder): Only take longs when D1 MACD > D1 Signal.
       Only take shorts (if enabled) when D1 MACD < D1 Signal.
    2. 1H entry signal: MACD histogram changes sign in trend direction.
       - Bullish cross: MACD crosses above Signal, AND D1 is UP → LONG
       - Bearish cross: MACD crosses below Signal, AND D1 is DOWN → SHORT
    3. Exit: MACD histogram crosses opposite direction, OR trailing stop hit.

Based on: QuantPedia (Nov 2025) — MTF MACD strategy for 1H/4H.
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal
from features.indicators import IndicatorCalculator


class MTF_MACD_Elder(BaseStrategy):
    """Multi-Timeframe MACD with Elder's Triple Screen methodology."""

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategies", {}).get("mtf_macd_elder", {})

        # MACD parameters
        macd_cfg = cfg.get("macd", {})
        self.fast = macd_cfg.get("fast", 12)
        self.slow = macd_cfg.get("slow", 26)
        self.signal_period = macd_cfg.get("signal", 9)

        # Exit parameters
        exit_cfg = cfg.get("exit", {})
        self.trailing_stop_pct = exit_cfg.get("trailing_stop_pct", 0.03)   # 3%
        self.atr_stop_mult = exit_cfg.get("atr_stop_mult", 2.0)
        self.min_hold_bars = exit_cfg.get("min_hold_bars", 1)

        # Elder filter
        filter_cfg = cfg.get("elder_filter", {})
        self.require_bb_squeeze = filter_cfg.get("require_bb_squeeze", False)
        self.require_volume = filter_cfg.get("require_volume_confirm", True)
        self.volume_mult = filter_cfg.get("volume_mult", 1.2)
        self.allow_shorts = filter_cfg.get("allow_shorts", True)

        # Indicator calculator
        self.ic = IndicatorCalculator()

        # State
        self.d1_trend: str = "FLAT"        # "UP", "DOWN", "FLAT"
        self.d1_macd: float = 0.0
        self.d1_signal: float = 0.0
        self.in_position: bool = False
        self.position_side: str = ""       # "long" or "short"
        self.entry_price: float = 0.0
        self.entry_bar: int = 0
        self.highest_since_entry: float = 0.0   # For long trailing stop
        self.lowest_since_entry: float = float("inf")  # For short trailing stop
        self._bar_counter: int = 0

        # Rolling cache for D1 MACD calculation
        self._d1_closes: list[float] = []

    @property
    def name(self) -> str:
        return "MTF_MACD_Elder"

    # ─── D1 Trend (Higher Timeframe) ───────────────────────────

    def on_higher_tf_candle(self, candle: dict, timeframe: str):
        """Update daily trend when a new 1D candle completes."""
        if timeframe == "1d":
            self._d1_closes.append(candle["close"])
            if len(self._d1_closes) >= self.slow + self.signal_period:
                self._update_d1_trend()

    def _update_d1_trend(self):
        """Recalculate D1 MACD from cached closes."""
        closes = pd.Series(self._d1_closes)
        macd_line, signal_line, histogram = self.ic.macd(
            closes, self.fast, self.slow, self.signal_period
        )
        self.d1_macd = macd_line.iloc[-1]
        self.d1_signal = signal_line.iloc[-1]

        if len(histogram) >= 2:
            prev_hist = histogram.iloc[-2]
            curr_hist = histogram.iloc[-1]
            if curr_hist > prev_hist:
                self.d1_trend = "UP"
            elif curr_hist < prev_hist:
                self.d1_trend = "DOWN"
            else:
                self.d1_trend = "FLAT"
        else:
            self.d1_trend = "FLAT"

    def set_d1_trend_direct(self, trend: str, d1_macd: float = 0, d1_signal: float = 0):
        """Manually set D1 trend (used in backtesting when D1 data is pre-computed)."""
        self.d1_trend = trend
        self.d1_macd = d1_macd
        self.d1_signal = d1_signal

    # ─── 1H Signal Generation ──────────────────────────────────

    def on_candle(self, candle: dict, features: pd.Series) -> Signal:
        """Generate signal for a 1H candle.

        Uses pre-computed features from the FeatureEngine. The features
        Series contains MACD values from the engine's indicator calculator.

        Entry: 1H MACD crosses signal in D1 trend direction
        Exit: MACD crosses opposite direction, OR trailing stop breached
        """
        self._bar_counter += 1

        # Requires D1 trend context
        if self.d1_trend == "FLAT":
            return Signal.FLAT

        # Extract feature values
        macd = features.get("macd", 0.0)
        macd_sig = features.get("macd_signal", 0.0)
        macd_hist = features.get("macd_hist", 0.0)
        macd_cross = features.get("macd_cross", 0.0)
        close = candle["close"]

        # ── Exit Logic ──────────────────────────────────────
        if self.in_position:
            exit_signal = self._check_exit_conditions(
                macd_cross, macd_hist, close, candle
            )
            if exit_signal != Signal.FLAT:
                self.in_position = False
                self.position_side = ""
                return exit_signal

            # Update trailing stop levels
            if self.position_side == "long":
                self.highest_since_entry = max(self.highest_since_entry, close)
            elif self.position_side == "short":
                self.lowest_since_entry = min(self.lowest_since_entry, close)

            return Signal.FLAT

        # ── Entry Logic ─────────────────────────────────────
        # MACD bullish crossover (macd_cross == 1)
        if self.d1_trend == "UP" and macd_cross == 1:
            # Elder volume filter: entry candle volume > average * multiplier
            if self.require_volume:
                vol_ratio = features.get("volume_sma_ratio", 1.0)
                if vol_ratio < self.volume_mult:
                    return Signal.FLAT

            self.in_position = True
            self.position_side = "long"
            self.entry_price = close
            self.entry_bar = self._bar_counter
            self.highest_since_entry = close
            return Signal.LONG

        # MACD bearish crossover (macd_cross == -1)
        if self.allow_shorts and self.d1_trend == "DOWN" and macd_cross == -1:
            if self.require_volume:
                vol_ratio = features.get("volume_sma_ratio", 1.0)
                if vol_ratio < self.volume_mult:
                    return Signal.FLAT

            self.in_position = True
            self.position_side = "short"
            self.entry_price = close
            self.entry_bar = self._bar_counter
            self.lowest_since_entry = close
            return Signal.SHORT

        return Signal.FLAT

    def _check_exit_conditions(
        self,
        macd_cross: float,
        macd_hist: float,
        close: float,
        candle: dict,
    ) -> Signal:
        """Check if we should exit current position.

        Returns Signal.FLAT if no exit condition met, otherwise the
        opposite signal (LONG to close short, SHORT to close long).
        """
        # 1. MACD opposite cross
        bars_held = self._bar_counter - self.entry_bar
        if self.position_side == "long" and macd_cross == -1:
            if bars_held >= self.min_hold_bars:
                return Signal.SHORT  # Close long

        if self.position_side == "short" and macd_cross == 1:
            if bars_held >= self.min_hold_bars:
                return Signal.LONG  # Close short

        # 2. Trailing stop (long positions)
        if self.position_side == "long":
            if self.highest_since_entry > 0:
                stop_level = self.highest_since_entry * (1 - self.trailing_stop_pct)
                if close <= stop_level:
                    return Signal.SHORT

        # 3. Trailing stop (short positions)
        if self.position_side == "short":
            if self.lowest_since_entry < float("inf"):
                stop_level = self.lowest_since_entry * (1 + self.trailing_stop_pct)
                if close >= stop_level:
                    return Signal.LONG

        # 4. ATR-based stop
        atr = candle.get("atr_14", 0)
        if atr > 0:
            if self.position_side == "long":
                atr_stop = self.entry_price - (self.atr_stop_mult * atr)
                if close <= atr_stop:
                    return Signal.SHORT
            elif self.position_side == "short":
                atr_stop = self.entry_price + (self.atr_stop_mult * atr)
                if close >= atr_stop:
                    return Signal.LONG

        return Signal.FLAT

    # ─── Retrain (for walk-forward) ────────────────────────────

    def retrain(self, historical_data: pd.DataFrame):
        """Not applicable for rule-based strategy — included for interface compatibility."""
        pass

    def reset_state(self):
        """Reset all internal state (between backtest folds)."""
        self.d1_trend = "FLAT"
        self.d1_macd = 0.0
        self.d1_signal = 0.0
        self.in_position = False
        self.position_side = ""
        self.entry_price = 0.0
        self.entry_bar = 0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = float("inf")
        self._bar_counter = 0
        self._d1_closes = []
