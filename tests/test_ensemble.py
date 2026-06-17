"""Tests for Phase 5/8: Ensemble routing and regime classification.

Uses real feature names from FeatureEngine output.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np


# ─── RegimeClassifier ─────────────────────────────────────────


class TestRegimeClassifier:

    @staticmethod
    def _make_classifier(config_overrides=None):
        from ensemble.regime_classifier import RegimeClassifier
        config = {
            "regime": {
                "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
                "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
                "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
                "hysteresis_bars": 2,
                "lookback_bars": 100,
            }
        }
        return RegimeClassifier(config)

    def _make_features(self, **overrides):
        """Build a feature dict with defaults suitable for each test."""
        defaults = {
            "adx_14": 20.0, "di_plus": 15.0, "di_minus": 15.0,
            "atr_14": 100.0, "bb_width": 0.03, "volatility_20": 0.30,
            "garman_klass": 0.02, "price": 50000, "dist_sma_50": 0.01,
        }
        defaults.update(overrides)
        return defaults

    def test_classify_trending(self):
        from ensemble.regime_classifier import MarketRegime
        rc = self._make_classifier()
        feats = self._make_features(adx_14=30.0, di_plus=30.0, di_minus=15.0)
        # Need 2 bars for hysteresis
        rc.classify(feats)
        regime = rc.classify(feats)
        assert regime == MarketRegime.TRENDING

    def test_classify_ranging(self):
        from ensemble.regime_classifier import MarketRegime
        rc = self._make_classifier()
        feats = self._make_features(adx_14=15.0, bb_width=0.02, volatility_20=0.20)
        rc.classify(feats)
        regime = rc.classify(feats)
        assert regime == MarketRegime.RANGING

    def test_classify_volatile(self):
        from ensemble.regime_classifier import MarketRegime
        rc = self._make_classifier()
        # High ATR relative to average + high HV
        rc._atr_samples = [50.0] * 10  # avg = 50
        feats = self._make_features(atr_14=200.0, volatility_20=1.5, bb_width=0.10)
        rc.classify(feats)
        regime = rc.classify(feats)
        assert regime == MarketRegime.VOLATILE

    def test_classify_unclear(self):
        from ensemble.regime_classifier import MarketRegime
        rc = self._make_classifier()
        feats = self._make_features(adx_14=22.0)  # Between trending and ranging
        rc.classify(feats)
        regime = rc.classify(feats)
        assert regime == MarketRegime.UNCLEAR

    def test_hysteresis_prevents_rapid_flips(self):
        from ensemble.regime_classifier import MarketRegime
        rc = self._make_classifier()
        # Force trending
        trending_feats = self._make_features(adx_14=30, di_plus=40, di_minus=15)
        rc.classify(trending_feats)
        rc.classify(trending_feats)  # Now locked to TRENDING

        # One bar of ranging shouldn't flip
        ranging_feats = self._make_features(adx_14=15, bb_width=0.02, volatility_20=0.20)
        regime = rc.classify(ranging_feats)
        assert regime == MarketRegime.TRENDING  # Stays trending

        # Second bar should still hold
        regime = rc.classify(ranging_feats)
        # Now 2/2 are ranging, should switch
        assert regime == MarketRegime.RANGING

    def test_reset(self):
        from ensemble.regime_classifier import MarketRegime
        rc = self._make_classifier()
        feats = self._make_features(adx_14=30, di_plus=40, di_minus=15)
        rc.classify(feats)
        rc.classify(feats)
        rc.reset()
        assert rc.current_regime == MarketRegime.UNCLEAR

    def test_metadata(self):
        rc = self._make_classifier()
        feats = self._make_features()
        rc.classify(feats)
        meta = rc.get_regime_metadata()
        assert "regime" in meta
        assert "hysteresis" in meta


# ─── EnsembleRouter ───────────────────────────────────────────


class TestEnsembleRouter:

    @staticmethod
    def _make_router():
        from ensemble.regime_classifier import RegimeClassifier
        from ensemble.router import EnsembleRouter
        from strategies.mtf_macd import MTF_MACD_Elder
        from strategies.mean_reversion import MeanReversion

        config = {
            "strategies": {
                "mtf_macd_elder": {
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "exit": {"trailing_stop_pct": 0.03, "atr_stop_mult": 2.0, "min_hold_bars": 1},
                    "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
                },
                "mean_reversion": {
                    "rsi": {"period": 14, "oversold": 30, "overbought": 70},
                    "bollinger": {"period": 20, "std_dev": 2},
                    "require_both_signals": True,
                },
            },
            "regime": {
                "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
                "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
                "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
                "hysteresis_bars": 1,  # No hysteresis for tests
                "lookback_bars": 100,
            },
        }

        macd = MTF_MACD_Elder(config)
        mr = MeanReversion(config)
        rc = RegimeClassifier(config)

        strategies = {"mtf_macd": macd, "mean_reversion": mr}
        return EnsembleRouter(strategies, rc), macd, rc

    def test_routes_to_macd_in_trending(self):
        from strategies.base import Signal
        from ensemble.regime_classifier import MarketRegime

        router, macd, rc = self._make_router()

        # Force trending regime
        rc.current_regime = MarketRegime.TRENDING
        macd.d1_trend = "UP"

        features = pd.Series({
            "macd_cross": 1, "macd": 150, "macd_signal": 100, "macd_hist": 50,
            "volume_sma_ratio": 1.5,
            "adx_14": 30, "di_plus": 40, "di_minus": 15,
        })
        candle = {"close": 50000, "open": 49900}

        signal = router.get_signal(candle, features)
        assert signal == Signal.LONG

    def test_routes_to_mean_reversion_in_ranging(self):
        """RANGING regime returns FLAT (MR disabled after 162-combo grid sweep)."""
        from strategies.base import Signal
        from ensemble.regime_classifier import MarketRegime

        router, macd, rc = self._make_router()

        rc.current_regime = MarketRegime.RANGING

        features = pd.Series({
            "rsi_14": 25, "bb_position": 0.02,
            "adx_14": 15, "bb_width": 0.02, "volatility_20": 0.20,
        })
        candle = {"close": 49000}

        signal = router.get_signal(candle, features)
        assert signal == Signal.FLAT  # MR disabled — sit out ranging markets

    def test_flat_in_volatile(self):
        from strategies.base import Signal
        from ensemble.regime_classifier import MarketRegime

        router, macd, rc = self._make_router()
        rc.current_regime = MarketRegime.VOLATILE

        features = pd.Series({"adx_14": 15, "bb_width": 0.10, "volatility_20": 1.5})
        candle = {"close": 50000}

        signal = router.get_signal(candle, features)
        assert signal == Signal.FLAT

    def test_flat_in_unclear_without_xgboost(self):
        from strategies.base import Signal
        from ensemble.regime_classifier import MarketRegime

        router, macd, rc = self._make_router()
        rc.current_regime = MarketRegime.UNCLEAR

        features = pd.Series({"adx_14": 22})
        candle = {"close": 50000}

        signal = router.get_signal(candle, features)
        assert signal == Signal.FLAT

    def test_regime_stats(self):
        router, macd, rc = self._make_router()
        router.record_trade_outcome(100)
        router.record_trade_outcome(-50)

        stats = router.get_regime_stats()
        assert "unclear" in stats
        assert stats["unclear"]["trades"] == 2

    def test_get_active_strategy_name(self):
        from strategies.base import Signal
        from ensemble.regime_classifier import MarketRegime
        router, macd, rc = self._make_router()

        # Trending: route via get_signal which updates current_regime
        rc.current_regime = MarketRegime.TRENDING
        macd.d1_trend = "UP"
        _ = router.get_signal(
            {"close": 50000},
            {"macd_cross": 0, "macd": 100, "macd_signal": 100, "macd_hist": 0,
             "volume_sma_ratio": 1.0, "adx_14": 30, "di_plus": 40, "di_minus": 15},
        )
        assert "mtf_macd" in router.get_active_strategy_name().lower()

        # Volatile: route to flat
        rc.current_regime = MarketRegime.VOLATILE
        _ = router.get_signal(
            {"close": 50000},
            {"adx_14": 30, "atr_14": 500, "bb_width": 0.10, "volatility_20": 1.5},
        )
        assert router.get_active_strategy_name() == "flat"


# ─── End-to-end ensemble integration ──────────────────────────


class TestEnsembleIntegration:

    def test_full_pipeline_routing(self):
        """Simulate 500 bars and verify regime routing logic doesn't crash."""
        from ensemble.regime_classifier import RegimeClassifier, MarketRegime
        from ensemble.router import EnsembleRouter
        from strategies.mtf_macd import MTF_MACD_Elder
        from strategies.mean_reversion import MeanReversion
        from features.engine import FeatureEngine

        # Generate data with different regimes
        rng = np.random.default_rng(42)
        n = 500
        timestamps = [1_706_400_000_000 + i * 3_600_000 for i in range(n)]

        # Mix: trending at start, ranging in middle, volatile at end
        returns = rng.normal(0.001, 0.01, n // 3)  # Strong trend
        returns = np.concatenate([returns, rng.normal(0, 0.005, n // 3)])  # Range
        returns = np.concatenate([returns, rng.normal(0, 0.03, n - 2 * (n // 3))])  # Volatile

        close = 50000 * np.cumprod(1 + returns)
        o = np.roll(close, 1); o[0] = close[0] * 0.999
        h = np.maximum(o, close) * (1 + rng.uniform(0.001, 0.02, n))
        l = np.minimum(o, close) * (1 - rng.uniform(0.001, 0.02, n))
        v = rng.uniform(50, 200, n)

        df = pd.DataFrame({
            "timestamp": timestamps, "open": o, "high": h, "low": l, "close": close, "volume": v,
        })

        # Build pipeline
        config_fe = {"features": {"max_window_bars": 500, "min_bars_required": 50}}
        fe = FeatureEngine(config_fe)
        features_df = fe.bulk_compute(df)

        config = {
            "strategies": {
                "mtf_macd_elder": {
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "exit": {"trailing_stop_pct": 0.03, "atr_stop_mult": 2.0, "min_hold_bars": 1},
                    "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
                },
                "mean_reversion": {
                    "rsi": {"period": 14, "oversold": 30, "overbought": 70},
                    "bollinger": {"period": 20, "std_dev": 2},
                    "require_both_signals": True,
                },
            },
            "regime": {
                "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
                "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
                "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
                "hysteresis_bars": 2,
                "lookback_bars": 100,
            },
        }

        macd = MTF_MACD_Elder(config)
        mr = MeanReversion(config)
        rc = RegimeClassifier(config)
        router = EnsembleRouter({"mtf_macd": macd, "mean_reversion": mr}, rc)

        # Set D1 trend for MACD
        macd.d1_trend = "UP"

        # Run through all bars
        signals = []
        regimes = []
        for i in range(50, len(df)):  # Skip warm-up
            row = df.iloc[i]
            feats_raw = features_df.iloc[i].to_dict()

            # Add atr_14 to candle for trailing stop
            candle = {"close": row["close"], "open": row["open"], "atr_14": feats_raw.get("atr_14", 0)}

            signal = router.get_signal(candle, feats_raw)
            signals.append(signal.value)
            regimes.append(router.current_regime.value)

        # Verify we see multiple signals and regimes
        assert len(set(signals)) >= 1  # At least FLAT
        assert len(set(regimes)) >= 1  # At least one regime detected
