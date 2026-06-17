"""Ensemble router — routes signals to the optimal strategy based on market regime.

Strategy allocation:
    TRENDING  → MTF MACD + Elder (trend-following, uses D1 Elder filter)
    RANGING   → Mean Reversion (RSI + Bollinger Band touch)
    VOLATILE  → FLAT (85% of trades die in volatility — StrategyArena 2026)
    UNCLEAR   → FLAT (sit out unclear markets)

Features:
    - Hysteresis prevents regime-hopping on every bar
    - Per-regime profit tracking to detect strategy underperformance
    - Fallback to FLAT when classified strategy is unavailable
"""

from typing import Optional
import pandas as pd

from .regime_classifier import RegimeClassifier, MarketRegime
from strategies.base import BaseStrategy, Signal


class EnsembleRouter:
    """Routes trading decisions based on market regime.

    Usage:
        router = EnsembleRouter(strategies, classifier)
        signal = router.get_signal(candle, features)
    """

    def __init__(self, strategies: dict[str, BaseStrategy], regime_classifier: RegimeClassifier):
        self.strategies = strategies
        self.classifier = regime_classifier
        self.current_regime = MarketRegime.UNCLEAR

        # Strategy performance tracking
        self._regime_trades: dict[str, list[float]] = {
            "trending": [],
            "ranging": [],
            "volatile": [],
            "unclear": [],
        }

    def get_signal(self, candle: dict, features: dict) -> Signal:
        """Detect regime and delegate to the best strategy for this market.

        Args:
            candle: Current 1H candle dict (timestamp, open, high, low, close, volume)
            features: Feature dict from FeatureEngine.process_candle()

        Returns:
            Signal.LONG, Signal.SHORT, or Signal.FLAT
        """
        # Classify current regime
        self.current_regime = self.classifier.classify(features)

        # Route based on regime
        if self.current_regime == MarketRegime.TRENDING:
            strat = self.strategies.get("mtf_macd")
            if strat:
                return strat.on_candle(candle, features)

        elif self.current_regime == MarketRegime.RANGING:
            # MeanReversion disabled: 162-combo grid sweep confirmed negative PnL
            # on all TEST configs (best: -$155, PF=0.67). Asymmetric risk profile
            # (wide ATR stop vs small mean-reversion target) makes it unprofitable.
            return Signal.FLAT

        elif self.current_regime == MarketRegime.VOLATILE:
            # Flat: too dangerous to trade
            return Signal.FLAT

        elif self.current_regime == MarketRegime.UNCLEAR:
            # Sit out unclear markets
            return Signal.FLAT

        return Signal.FLAT

    def get_active_strategy_name(self) -> str:
        """Return the name of the strategy currently being used."""
        regime = self.current_regime.value

        if regime == "trending":
            s = self.strategies.get("mtf_macd")
            return s.name if s else "none"
        elif regime == "ranging":
            s = self.strategies.get("mean_reversion")
            return s.name if s else "none"
        elif regime == "volatile":
            return "flat"
        elif regime == "unclear":
            return "flat"
        return "unknown"

    def record_trade_outcome(self, pnl: float):
        """Record trade outcome for the current regime.

        Used to track if a strategy is underperforming in its assigned regime.
        """
        regime_key = self.current_regime.value if self.current_regime else "unclear"
        self._regime_trades[regime_key].append(pnl)
        # Keep last 100 per regime
        if len(self._regime_trades[regime_key]) > 100:
            self._regime_trades[regime_key] = self._regime_trades[regime_key][-100:]

    def get_regime_stats(self) -> dict:
        """Return per-regime performance statistics."""
        stats = {}
        for regime, pnls in self._regime_trades.items():
            if pnls:
                stats[regime] = {
                    "trades": len(pnls),
                    "total_pnl": round(sum(pnls), 2),
                    "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
                    "avg_pnl": round(sum(pnls) / len(pnls), 2),
                }
            else:
                stats[regime] = {"trades": 0, "total_pnl": 0, "win_rate": 0, "avg_pnl": 0}
        return stats

    def reset(self):
        """Reset state between backtest folds."""
        self.current_regime = MarketRegime.UNCLEAR
        self.classifier.reset()
        for key in self._regime_trades:
            self._regime_trades[key] = []
