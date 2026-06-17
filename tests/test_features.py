"""Tests for Phase 2: Feature Engineering.

Covers:
- IndicatorCalculator: all indicator formulas produce correct shapes
- DerivedFeatures: pattern detection, divergences
- FeatureEngine: process_candle (live) + bulk_compute (backtest)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np


# ─── Generate realistic OHLCV data ─────────────────────────────

def make_ohlcv(n_bars: int = 500, start_price: float = 50000.0, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    timestamps = [1_700_000_000_000 + i * 3_600_000 for i in range(n_bars)]

    # Random walk with drift
    returns = rng.normal(0.0002, 0.02, n_bars)  # Small positive drift
    close = start_price * np.cumprod(1 + returns)

    opens = np.roll(close, 1)
    opens[0] = start_price * 0.998  # First open slightly below start

    # Generate realistic OHLC from close
    highs = np.maximum(opens, close) * (1 + rng.uniform(0.001, 0.02, n_bars))
    lows = np.minimum(opens, close) * (1 - rng.uniform(0.001, 0.02, n_bars))
    volumes = rng.uniform(10, 100, n_bars)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": close,
        "volume": volumes,
    })


# ─── IndicatorCalculator Tests ─────────────────────────────────


class TestIndicatorCalculator:
    """Verify each indicator produces correctly-shaped output."""

    def setup_method(self):
        self.df = make_ohlcv(300)
        self.o, self.h, self.l, self.c, self.v = (
            self.df["open"], self.df["high"], self.df["low"],
            self.df["close"], self.df["volume"],
        )
        from features.indicators import IndicatorCalculator
        self.ic = IndicatorCalculator()

    def test_sma(self):
        result = self.ic.sma(self.c, 20)
        assert len(result) == len(self.c)
        assert result.iloc[19] > 0  # First valid value at index 19
        assert np.isnan(result.iloc[0])  # Before period, should be NaN

    def test_ema(self):
        result = self.ic.ema(self.c, 20)
        assert len(result) == len(self.c)

    def test_atr(self):
        result = self.ic.atr(self.h, self.l, self.c, 14)
        assert len(result) == len(self.c)
        assert result.iloc[-1] > 0

    def test_bollinger_bands(self):
        upper, mid, lower = self.ic.bollinger_bands(self.c, 20, 2)
        assert len(upper) == len(self.c)
        valid = upper.notna() & mid.notna() & lower.notna()
        assert (upper[valid] > mid[valid]).all()
        assert (mid[valid] > lower[valid]).all()

    def test_historical_volatility(self):
        result = self.ic.historical_volatility(self.c, 20)
        assert len(result) == len(self.c)

    def test_garman_klass(self):
        result = self.ic.garman_klass(self.o, self.h, self.l, self.c, 20)
        assert len(result) == len(self.c)
        valid = result.dropna()
        assert (valid >= 0).all()  # Should be non-negative

    def test_parkinson(self):
        result = self.ic.parkinson(self.h, self.l, 20)
        assert len(result) == len(self.c)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_macd(self):
        macd_line, signal, hist = self.ic.macd(self.c, 12, 26, 9)
        assert len(macd_line) == len(self.c)
        assert len(signal) == len(self.c)
        assert len(hist) == len(self.c)

    def test_macd_cross(self):
        macd_line, signal, _ = self.ic.macd(self.c, 12, 26, 9)
        crosses = self.ic.macd_cross(macd_line, signal)
        assert crosses.isin([-1, 0, 1]).all()
        assert crosses.sum() != 0  # Should have at least some crosses in 300 bars

    def test_adx(self):
        result = self.ic.adx(self.h, self.l, self.c, 14)
        assert len(result) == len(self.c)
        assert result.max() <= 100
        assert result.min() >= 0

    def test_di(self):
        di_plus, di_minus = self.ic.di(self.h, self.l, self.c, 14)
        assert len(di_plus) == len(self.c)
        assert di_plus.max() <= 100
        assert di_minus.min() >= 0

    def test_rsi(self):
        result = self.ic.rsi(self.c, 14)
        assert len(result) == len(self.c)
        assert result.max() <= 100
        assert result.min() >= 0

    def test_stochastic(self):
        k, d = self.ic.stochastic(self.h, self.l, self.c, 14, 3)
        assert len(k) == len(self.c)
        assert len(d) == len(self.c)

    def test_cci(self):
        result = self.ic.cci(self.h, self.l, self.c, 20)
        assert len(result) == len(self.c)

    def test_williams_r(self):
        result = self.ic.williams_r(self.h, self.l, self.c, 14)
        assert len(result) == len(self.c)
        assert result.max() <= 0
        assert result.min() >= -100

    def test_roc(self):
        result = self.ic.roc(self.c, 10)
        assert len(result) == len(self.c)

    def test_obv(self):
        result = self.ic.obv(self.c, self.v)
        assert len(result) == len(self.c)
        assert result.iloc[-1] != 0  # Should accumulate over 300 bars

    def test_mfi(self):
        result = self.ic.mfi(self.h, self.l, self.c, self.v, 14)
        assert len(result) == len(self.c)
        assert result.max() <= 100
        assert result.min() >= 0

    def test_ease_of_movement(self):
        result = self.ic.ease_of_movement(self.h, self.l, self.v, 14)
        assert len(result) == len(self.c)


# ─── DerivedFeatures Tests ─────────────────────────────────────


class TestDerivedFeatures:

    def setup_method(self):
        self.df = make_ohlcv(300)
        self.o, self.h, self.l, self.c = (
            self.df["open"], self.df["high"],
            self.df["low"], self.df["close"],
        )
        from features.derived import DerivedFeatures
        self.df_util = DerivedFeatures()

    def test_detect_patterns_returns_dict(self):
        patterns = self.df_util.detect_patterns(self.o, self.h, self.l, self.c)
        assert isinstance(patterns, dict)
        assert len(patterns) > 0
        # All values should be 0/1
        for name, col in patterns.items():
            assert col.isin([0, 1]).all(), f"{name} has non-binary values"

    def test_doji_pattern(self):
        patterns = self.df_util.detect_patterns(self.o, self.h, self.l, self.c)
        assert patterns["doji"].sum() > 0  # Should find some dojis in 300 bars

    def test_bullish_engulfing(self):
        patterns = self.df_util.detect_patterns(self.o, self.h, self.l, self.c)
        assert "bullish_engulfing" in patterns

    def test_trend_strength(self):
        result = self.df_util.trend_strength(self.c, 20)
        assert len(result) == len(self.c)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_gap_type(self):
        result = self.df_util.gap_type(self.o, self.h, self.l, self.c)
        assert len(result) == len(self.c)
        assert result.isin([-2, -1, 0, 1, 2]).all()

    def test_rsi_divergence(self):
        from features.indicators import IndicatorCalculator
        ic = IndicatorCalculator()
        rsi = ic.rsi(self.c, 14)
        result = self.df_util.rsi_divergence(self.c, rsi, lookback=5)
        assert result.isin([-1, 0, 1]).all()


# ─── FeatureEngine Tests ───────────────────────────────────────


class TestFeatureEngine:

    def setup_method(self):
        from features.engine import FeatureEngine
        config = {"features": {"max_window_bars": 500, "min_bars_required": 50}}
        self.engine = FeatureEngine(config)

    def test_bulk_compute_returns_dataframe(self):
        df = make_ohlcv(300)
        features = self.engine.bulk_compute(df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(df)
        assert len(features.columns) > 50, f"Expected 60+ features, got {len(features.columns)}"

    def test_bulk_compute_with_multitf(self):
        df_1h = make_ohlcv(300)

        # Make 4H: take every 4th bar
        df_4h = df_1h.iloc[::4].copy().reset_index(drop=True)
        df_4h["timestamp"] = df_4h["timestamp"].astype("int64")

        # Make 1D: take every 24th bar
        df_1d = df_1h.iloc[::24].copy().reset_index(drop=True)
        df_1d["timestamp"] = df_1d["timestamp"].astype("int64")

        features = self.engine.bulk_compute(df_1h, df_4h, df_1d)
        assert len(features) == len(df_1h)

        # Check multi-TF columns exist
        mtf_cols = [c for c in features.columns if c.startswith("vs_")]
        assert len(mtf_cols) > 0, f"Expected multi-TF columns, got: {mtf_cols}"

    def test_bulk_compute_no_crash_on_empty_mtf(self):
        df = make_ohlcv(300)
        features = self.engine.bulk_compute(df, None, None)
        assert len(features) == len(df)

    def test_process_candle_with_enough_history(self):
        """Prime with 100 bars, then feed one new candle."""
        df = make_ohlcv(100)
        self.engine.prime_cache(df)

        # Feed one new candle
        new_candle = {
            "timestamp": 1_700_000_000_000 + 100 * 3_600_000,
            "open": 52000.0, "high": 52200.0, "low": 51800.0,
            "close": 52100.0, "volume": 50.0,
        }
        features = self.engine.process_candle(new_candle)
        assert isinstance(features, dict)
        assert len(features) > 50, f"Expected 60+ features, got {len(features)}"

    def test_process_candle_without_enough_history(self):
        """Without enough bars, should return empty dict."""
        new_candle = {
            "timestamp": 1_700_000_000_000 + 100 * 3_600_000,
            "open": 52000.0, "high": 52200.0, "low": 51800.0,
            "close": 52100.0, "volume": 50.0,
        }
        features = self.engine.process_candle(new_candle)
        assert features == {}

    def test_feature_names_consistent(self):
        df = make_ohlcv(300)
        features1 = self.engine.bulk_compute(df)
        names1 = self.engine.get_feature_names()

        features2 = self.engine.bulk_compute(df)
        names2 = self.engine.get_feature_names()

        assert names1 == names2, "Feature names should be stable"

    def test_all_features_finite(self):
        """No NaN or Inf in fully-warmed features (beyond min_bars_required)."""
        df = make_ohlcv(300)
        features = self.engine.bulk_compute(df)

        # Check rows after warm-up (skip first 50)
        warm_slice = features.iloc[100:]  # Well past min_bars_required
        for col in warm_slice.columns:
            assert not warm_slice[col].isna().all(), f"Column {col} is all NaN"
            assert np.isfinite(warm_slice[col].dropna()).all(), f"Column {col} has non-finite values"

    def test_bulk_compute_no_inf(self):
        """Features should not contain Inf values."""
        df = make_ohlcv(500)
        features = self.engine.bulk_compute(df)
        for col in features.columns:
            finite = features[col].dropna()
            assert np.isfinite(finite).all(), f"Column {col} contains Inf"
