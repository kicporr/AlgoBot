"""Mean Reversion Strategy (for ranging markets only).

Uses RSI extremes + Bollinger Band position as entry signals.
Only active when regime classifier detects RANGING markets.

Entry logic:
    LONG: RSI below oversold threshold AND price near lower Bollinger band
    SHORT (if enabled): RSI above overbought AND price near upper band

Exit: RSI crosses back past midpoint (50), or BB middle band retest.
"""

import pandas as pd
from .base import BaseStrategy, Signal


class MeanReversion(BaseStrategy):
    """RSI + Bollinger Bands mean reversion for range-bound markets.

    Feature names from FeatureEngine:
        rsi_14, bb_position (0=near lower, 1=near upper)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategies", {}).get("mean_reversion", {})
        rsi_cfg = cfg.get("rsi", {})
        bb_cfg = cfg.get("bollinger", {})
        self.rsi_period = rsi_cfg.get("period", 14)
        self.oversold = rsi_cfg.get("oversold", 30)
        self.overbought = rsi_cfg.get("overbought", 70)
        self.bb_period = bb_cfg.get("period", 20)
        self.bb_std = bb_cfg.get("std_dev", 2)
        self.require_both = cfg.get("require_both_signals", True)
        self.allow_shorts = cfg.get("allow_shorts", False)

        # Configurable entry/exit thresholds
        self.bb_oversold = bb_cfg.get("oversold_threshold", 0.05)
        self.bb_overbought = bb_cfg.get("overbought_threshold", 0.95)
        self.bb_exit_long = bb_cfg.get("exit_long_threshold", 0.5)
        self.bb_exit_short = bb_cfg.get("exit_short_threshold", 0.5)
        self.rsi_exit_long = rsi_cfg.get("exit_long_threshold", 50)
        self.rsi_exit_short = rsi_cfg.get("exit_short_threshold", 50)

        # State
        self.in_position = False
        self.position_side = ""

    @property
    def name(self) -> str:
        return "MeanReversion"

    def on_candle(self, candle: dict, features: dict) -> Signal:
        """Generate signal on oversold/overbought conditions.

        Uses feature names from FeatureEngine:
            rsi_14, bb_position
        """
        rsi = features.get("rsi_14", 50)
        bb_pos = features.get("bb_position", 0.5)  # 0=at lower band, 1=at upper band
        close = candle.get("close", 0)

        # ── Exit Logic ──────────────────────────────────────
        if self.in_position:
            if self.position_side == "long":
                # Exit long when RSI returns above threshold or price at BB exit threshold
                if rsi >= self.rsi_exit_long or bb_pos >= self.bb_exit_long:
                    self.in_position = False
                    self.position_side = ""
                    return Signal.SHORT  # Close long
            elif self.position_side == "short":
                if rsi <= self.rsi_exit_short or bb_pos <= self.bb_exit_short:
                    self.in_position = False
                    self.position_side = ""
                    return Signal.LONG  # Close short
            return Signal.FLAT

        # ── Entry Logic ─────────────────────────────────────
        # Long: oversold RSI and/or price near lower band
        if self.require_both:
            long_cond = (rsi <= self.oversold) and (bb_pos <= self.bb_oversold)
        else:
            long_cond = (rsi <= self.oversold) or (bb_pos <= self.bb_oversold)

        if long_cond:
            self.in_position = True
            self.position_side = "long"
            return Signal.LONG

        # Short: overbought RSI and/or price near upper band
        if self.allow_shorts:
            if self.require_both:
                short_cond = (rsi >= self.overbought) and (bb_pos >= self.bb_overbought)
            else:
                short_cond = (rsi >= self.overbought) or (bb_pos >= self.bb_overbought)

            if short_cond:
                self.in_position = True
                self.position_side = "short"
                return Signal.SHORT

        return Signal.FLAT

    def on_position_closed(self):
        """Synchronize internal state when position is closed externally."""
        self.in_position = False
        self.position_side = ""

    def retrain(self, historical_data: pd.DataFrame):
        """Not applicable for rule-based strategy."""
        pass

    def reset_state(self):
        self.in_position = False
        self.position_side = ""
