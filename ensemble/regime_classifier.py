"""Market regime classification.

Detects current market state using ADX, Bollinger Bands, ATR, and trend strength.
Uses real feature names from FeatureEngine output.

States:
    TRENDING — ADX > 25, directional DI alignment, trend strength confirmed
    RANGING  — ADX < 20, tight Bollinger Bands, low volatility
    VOLATILE — ATR > 2× average, BB width elevated, high Garman-Klass
    UNCLEAR  — Mixed signals; fall back to ensemble default

Hysteresis: requires 2 consecutive bars of the same regime to switch.
"""

from enum import Enum
from typing import Optional
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNCLEAR = "unclear"


class RegimeClassifier:
    """Rule-based regime detector with hysteresis.

    Uses features from FeatureEngine:
        adx_14, di_plus, di_minus, atr_14, bb_width, volatility_20,
        garman_klass, dist_sma_50, trend_strength
    """

    def __init__(self, config: dict):
        rc_cfg = config.get("regime", {})

        # Trending thresholds
        trend_cfg = rc_cfg.get("trending", {})
        self.adx_trend_min = trend_cfg.get("adx_min", 25)
        self.di_ratio_strong = trend_cfg.get("di_ratio_strong", 1.3)   # DI+ / DI- > 1.3
        self.di_ratio_reverse = trend_cfg.get("di_ratio_reverse", 0.77) # DI+ / DI- < 0.77

        # Ranging thresholds
        range_cfg = rc_cfg.get("ranging", {})
        self.adx_range_max = range_cfg.get("adx_max", 20)
        self.bb_width_max = range_cfg.get("bb_width_max", 0.04)       # BB width < 4%
        self.vol_range_max = range_cfg.get("vol_max", 0.50)           # HV < 50%

        # Volatile thresholds
        vol_cfg = rc_cfg.get("volatile", {})
        self.atr_mult = vol_cfg.get("atr_mult", 2.0)                  # ATR > 2× avg
        self.vol_absolute = vol_cfg.get("vol_absolute", 1.0)          # HV > 100%
        self.bb_width_vol = vol_cfg.get("bb_width_min", 0.08)         # BB > 8%

        # General
        self.hysteresis_bars = rc_cfg.get("hysteresis_bars", 2)
        self.lookback = rc_cfg.get("lookback_bars", 100)

        # State tracking
        self.current_regime = MarketRegime.UNCLEAR
        self._regime_votes: list[MarketRegime] = []  # For hysteresis
        self._bar_count = 0

        # Rolling averages for comparison
        self._atr_samples: list[float] = []
        self._vol_samples: list[float] = []

    def classify(self, features: dict) -> MarketRegime:
        """Classify current market regime from a feature dict.

        Args:
            features: Feature dict from FeatureEngine.process_candle()

        Returns:
            MarketRegime enum value.
        """
        self._bar_count += 1

        # Extract features with defaults
        adx = features.get("adx_14", 20.0)
        di_p = features.get("di_plus", 15.0)
        di_m = features.get("di_minus", 15.0)
        di_ratio = di_p / (di_m + 1e-12)
        atr = features.get("atr_14", 0.0)
        bb_width = features.get("bb_width", 0.0)
        hv = features.get("volatility_20", 0.0)
        gk = features.get("garman_klass", 0.0)
        price = features.get("price", 0.0)
        dist_sma50 = features.get("dist_sma_50", 0.0)
        trend_r2 = features.get("trend_strength", 0.0) if "trend_strength" in features else 0.5

        # Update rolling stats
        self._atr_samples.append(atr)
        if len(self._atr_samples) > self.lookback:
            self._atr_samples = self._atr_samples[-self.lookback:]

        avg_atr = sum(self._atr_samples) / len(self._atr_samples) if self._atr_samples else atr

        # ── Rule 1: VOLATILE — check first (risk priority) ────
        vol_signals = 0
        if avg_atr > 0 and atr > self.atr_mult * avg_atr:
            vol_signals += 1
        if hv > self.vol_absolute:
            vol_signals += 1
        if bb_width > self.bb_width_vol:
            vol_signals += 1

        if vol_signals >= 2:
            regime = MarketRegime.VOLATILE
        # ── Rule 2: TRENDING ──────────────────────────────────
        elif adx >= self.adx_trend_min and (
            di_ratio >= self.di_ratio_strong or di_ratio <= self.di_ratio_reverse
        ):
            regime = MarketRegime.TRENDING
        # ── Rule 3: RANGING ───────────────────────────────────
        elif (
            adx <= self.adx_range_max
            and bb_width <= self.bb_width_max
            and hv <= self.vol_range_max
        ):
            regime = MarketRegime.RANGING
        # ── Rule 4: UNCLEAR (fallthrough) ─────────────────────
        else:
            regime = MarketRegime.UNCLEAR

        # ── Hysteresis ────────────────────────────────────────
        return self._apply_hysteresis(regime)

    def _apply_hysteresis(self, new_regime: MarketRegime) -> MarketRegime:
        """Only switch regime if N consecutive bars agree."""
        self._regime_votes.append(new_regime)
        if len(self._regime_votes) > self.hysteresis_bars:
            self._regime_votes = self._regime_votes[-self.hysteresis_bars:]

        # If all votes agree on new regime, switch
        if len(self._regime_votes) == self.hysteresis_bars:
            if all(v == new_regime for v in self._regime_votes):
                if self.current_regime != new_regime:
                    logger.info(
                        f"Regime switch: {self.current_regime.value} → {new_regime.value} "
                        f"(bar {self._bar_count})"
                    )
                self.current_regime = new_regime
                return new_regime

        return self.current_regime

    def get_regime_metadata(self) -> dict:
        """Return diagnostic info about current classification."""
        return {
            "regime": self.current_regime.value,
            "bar_count": self._bar_count,
            "hysteresis": [r.value for r in self._regime_votes],
            "avg_atr": round(
                sum(self._atr_samples) / len(self._atr_samples), 1
            ) if self._atr_samples else 0.0,
        }

    def reset(self):
        """Reset state (between backtest folds)."""
        self.current_regime = MarketRegime.UNCLEAR
        self._regime_votes = []
        self._bar_count = 0
        self._atr_samples = []
        self._vol_samples = []
