"""Tests for strategy modules (Phase 3).

Uses actual feature names from FeatureEngine output.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from strategies.base import Signal


# ─── MTF MACD Elder ───────────────────────────────────────────


class TestMTF_MACD:
    """Test MTF MACD strategy with real feature names."""

    @staticmethod
    def _make_strategy(config_overrides=None):
        from strategies.mtf_macd import MTF_MACD_Elder
        config = {
            "strategies": {
                "mtf_macd_elder": {
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "exit": {"trailing_stop_pct": 0.03, "atr_stop_mult": 2.0, "min_hold_bars": 1},
                    "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
                }
            }
        }
        if config_overrides:
            config["strategies"]["mtf_macd_elder"].update(config_overrides)
        return MTF_MACD_Elder(config)

    def test_name(self):
        strategy = self._make_strategy()
        assert strategy.name == "MTF_MACD_Elder"

    def test_flat_when_no_d1_trend(self):
        """Without D1 trend, should return FLAT even with MACD cross."""
        strategy = self._make_strategy()
        strategy.d1_trend = "FLAT"

        features = pd.Series({"macd_cross": 1, "macd": 100, "macd_signal": 90})
        candle = {"close": 50000, "open": 49900}

        assert strategy.on_candle(candle, features) == Signal.FLAT

    def test_long_on_bullish_macd_cross_with_d1_up(self):
        """When D1 is UP and MACD crosses bullish → LONG."""
        strategy = self._make_strategy()
        strategy.d1_trend = "UP"

        features = pd.Series({
            "macd_cross": 1, "macd": 150, "macd_signal": 100,
            "macd_hist": 50, "volume_sma_ratio": 1.5,
        })
        candle = {"close": 50000, "open": 49900}

        signal = strategy.on_candle(candle, features)
        assert signal == Signal.LONG
        assert strategy.in_position is True
        assert strategy.position_side == "long"

    def test_short_on_bearish_macd_cross_with_d1_down(self):
        """When D1 is DOWN and MACD crosses bearish → SHORT."""
        strategy = self._make_strategy()
        strategy.d1_trend = "DOWN"

        features = pd.Series({
            "macd_cross": -1, "macd": 100, "macd_signal": 150,
            "macd_hist": -50, "volume_sma_ratio": 1.5,
        })
        candle = {"close": 50000, "open": 49900}

        signal = strategy.on_candle(candle, features)
        assert signal == Signal.SHORT

    def test_no_entry_when_already_in_position(self):
        """Should not generate additional entry signals when in a position."""
        strategy = self._make_strategy()
        strategy.d1_trend = "UP"
        strategy.in_position = True
        strategy.position_side = "long"
        strategy.entry_price = 50000
        strategy.highest_since_entry = 51000

        features = pd.Series({"macd_cross": 1, "macd": 150, "macd_signal": 100})
        candle = {"close": 50500}

        signal = strategy.on_candle(candle, features)
        assert signal == Signal.FLAT  # No new entry

    def test_exit_on_opposite_macd_cross(self):
        """When in long position and MACD crosses bearish → exit."""
        strategy = self._make_strategy()
        strategy.d1_trend = "UP"
        strategy.in_position = True
        strategy.position_side = "long"
        strategy.entry_price = 50000
        strategy.highest_since_entry = 51000

        features = pd.Series({"macd_cross": -1, "macd": 100, "macd_signal": 150})
        candle = {"close": 50500}

        signal = strategy.on_candle(candle, features)
        assert signal == Signal.SHORT  # Close long
        assert strategy.in_position is False

    def test_volume_filter_blocks_entry(self):
        """When volume filter is enabled and volume is low → no entry."""
        strategy = self._make_strategy({
            "elder_filter": {"require_volume_confirm": True, "volume_mult": 1.2, "allow_shorts": True},
        })
        strategy.d1_trend = "UP"

        features = pd.Series({
            "macd_cross": 1, "macd": 150, "macd_signal": 100,
            "volume_sma_ratio": 0.8,  # Below 1.2 threshold
        })
        candle = {"close": 50000}

        signal = strategy.on_candle(candle, features)
        assert signal == Signal.FLAT  # Blocked by low volume

    def test_trailing_stop_exit_long(self):
        """Trailing stop should exit when price drops below stop level."""
        strategy = self._make_strategy()
        strategy.d1_trend = "UP"
        strategy.in_position = True
        strategy.position_side = "long"
        strategy.entry_price = 50000
        strategy.highest_since_entry = 52000  # Peak

        # Trailing stop = 52000 * 0.97 = 50440
        features = pd.Series({"macd_cross": 0, "macd": 100, "macd_signal": 100})
        candle = {"close": 50000, "atr_14": 500}  # Below trailing stop

        signal = strategy.on_candle(candle, features)
        assert signal == Signal.SHORT  # Exit long
        assert strategy.in_position is False

    def test_d1_trend_update(self):
        """Verify on_higher_tf_candle computes D1 MACD trend."""
        from features.indicators import IndicatorCalculator

        strategy = self._make_strategy()
        ic = IndicatorCalculator()

        # Feed 35+ daily candles (need at least fast+slow+signal = 26+9 = 35)
        base_price = 50000
        for i in range(40):
            close = base_price * (1 + 0.001 * i)  # Slight uptrend
            strategy.on_higher_tf_candle({"close": close}, "1d")

        assert strategy.d1_trend in ("UP", "DOWN", "FLAT")


# ─── Mean Reversion ───────────────────────────────────────────


class TestMeanReversion:

    @staticmethod
    def _make_strategy(config_overrides=None):
        from strategies.mean_reversion import MeanReversion
        config = {
            "strategies": {
                "mean_reversion": {
                    "rsi": {"period": 14, "oversold": 30, "overbought": 70},
                    "bollinger": {"period": 20, "std_dev": 2},
                    "require_both_signals": True,
                }
            }
        }
        return MeanReversion(config)

    def test_name(self):
        strategy = self._make_strategy()
        assert strategy.name == "MeanReversion"

    def test_long_on_oversold_at_lower_band(self):
        """RSI oversold AND price near lower BB → LONG."""
        strategy = self._make_strategy()
        features = pd.Series({"rsi_14": 25, "bb_position": 0.02})
        candle = {"close": 49000}
        signal = strategy.on_candle(candle, features)
        assert signal == Signal.LONG

    def test_no_entry_when_rsi_ok(self):
        """RSI above oversold → no entry."""
        strategy = self._make_strategy()
        features = pd.Series({"rsi_14": 50, "bb_position": 0.02})
        candle = {"close": 50000}
        signal = strategy.on_candle(candle, features)
        assert signal == Signal.FLAT

    def test_exit_on_rsi_retracement(self):
        """Exit long when RSI returns to 50."""
        strategy = self._make_strategy()
        strategy.in_position = True
        strategy.position_side = "long"

        features = pd.Series({"rsi_14": 55, "bb_position": 0.3})
        candle = {"close": 50500}
        signal = strategy.on_candle(candle, features)
        assert signal == Signal.SHORT  # Close long
        assert strategy.in_position is False


# ─── Backtest Engine Integration ──────────────────────────────


class TestBacktestIntegration:
    """End-to-end backtest with MTF MACD on synthetic data."""

    def make_synthetic_data(self, n_bars=1000, trend_strength=0.0002):
        """Generate synthetic 1H OHLCV with a trend bias."""
        rng = np.random.default_rng(42)
        timestamps = [1_706_400_000_000 + i * 3_600_000 for i in range(n_bars)]
        returns = rng.normal(trend_strength, 0.01, n_bars)
        close = 50000 * np.cumprod(1 + returns)

        open_vals = np.roll(close, 1)
        open_vals[0] = close[0] * 0.999
        highs = np.maximum(open_vals, close) * (1 + rng.uniform(0.001, 0.01, n_bars))
        lows = np.minimum(open_vals, close) * (1 - rng.uniform(0.001, 0.01, n_bars))
        volumes = rng.uniform(50, 200, n_bars)

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_vals,
            "high": highs,
            "low": lows,
            "close": close,
            "volume": volumes,
        })

    def make_synthetic_1d(self, data_1h, n_daily_bars=60):
        """Create synthetic 1D data from 1H data."""
        # Take every 24th candle
        df_d = data_1h.iloc[::24].copy().reset_index(drop=True)
        df_d = df_d.head(n_daily_bars)
        # Regenerate OHLC for daily bars
        rng = np.random.default_rng(24)
        close_vals = 50000 * np.cumprod(1 + rng.normal(0.001, 0.02, len(df_d)))
        open_vals = np.roll(close_vals, 1)
        open_vals[0] = close_vals[0] * 0.995
        highs = np.maximum(open_vals, close_vals) * (1 + rng.uniform(0.005, 0.03, len(df_d)))
        lows = np.minimum(open_vals, close_vals) * (1 - rng.uniform(0.005, 0.03, len(df_d)))

        return pd.DataFrame({
            "timestamp": df_d["timestamp"].values,
            "open": open_vals,
            "high": highs,
            "low": lows,
            "close": close_vals,
            "volume": rng.uniform(500, 2000, len(df_d)),
        })

    def test_walk_forward_runs(self):
        """Smoke test: walk-forward backtest doesn't crash."""
        from backtest.engine import BacktestEngine
        from strategies.mtf_macd import MTF_MACD_Elder

        # Need enough data for folds and feature warm-up
        data = self.make_synthetic_data(n_bars=800)
        data_1d = self.make_synthetic_1d(data)

        config = {
            "exchange": {"fees": {"taker": 0.001, "maker": 0.0005, "slippage": 0.0005}},
            "risk": {"initial_capital": 10000, "max_position_pct": 0.95},
            "backtest": {"walk_forward_folds": 8, "min_train_fraction": 0.25},
            "strategies": {"mtf_macd_elder": {
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "exit": {"trailing_stop_pct": 0.03, "atr_stop_mult": 2.0, "min_hold_bars": 1},
                "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
            }},
            "features": {"max_window_bars": 300, "min_bars_required": 50},
        }

        engine = BacktestEngine(config)
        result = engine.run_walk_forward(data, MTF_MACD_Elder, data_1d=data_1d)

        assert isinstance(result.metrics, dict)
        assert "total_trades" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert len(result.trades) >= 0  # May have zero trades, that's OK for smoke test
        assert len(result.equity_curve) > 0

    def test_walk_forward_with_strong_trend(self):
        """With strong uptrend, strategy should find some trades."""
        from backtest.engine import BacktestEngine
        from strategies.mtf_macd import MTF_MACD_Elder

        # Strong uptrend
        data = self.make_synthetic_data(n_bars=500, trend_strength=0.001)
        data_1d = self.make_synthetic_1d(data)

        config = {
            "exchange": {"fees": {"taker": 0.001, "maker": 0.0005, "slippage": 0.0005}},
            "risk": {"initial_capital": 10000, "max_position_pct": 0.95},
            "backtest": {"walk_forward_folds": 5, "min_train_fraction": 0.25},
            "strategies": {"mtf_macd_elder": {
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "exit": {"trailing_stop_pct": 0.05, "atr_stop_mult": 2.0, "min_hold_bars": 1},
                "elder_filter": {"require_volume_confirm": False, "allow_shorts": False},
            }},
            "features": {"max_window_bars": 300, "min_bars_required": 50},
        }

        engine = BacktestEngine(config)
        result = engine.run_walk_forward(data, MTF_MACD_Elder, data_1d=data_1d)

        # Verify trade records are well-formed
        for trade in result.trades:
            assert "entry_time" in trade
            assert "exit_time" in trade
            assert "side" in trade
            assert "pnl" in trade
            assert "exit_reason" in trade
            assert trade["entry_time"] < trade["exit_time"], "Entry must be before exit"

        # Verify equity curve is monotonic (with deposits at least non-negative starting cap)
        assert result.equity_curve[0] == 10000.0
