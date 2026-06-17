"""Feature engineering pipeline.

Computes 60+ features across 7 categories using pure pandas/numpy.
No external TA library required — every indicator is implemented
from documented formulas.

Categories:
    1. Price-based (8 features)  — returns, ratios, bar position
    2. Volatility (6 features)   — ATR, Bollinger, Garman-Klass, Parkinson
    3. Trend (12 features)       — MACD family, ADX, EMA slopes, SMA crosses
    4. Momentum (10 features)    — RSI, Stochastic, CCI, Williams %R, ROC
    5. Volume (7 features)       — OBV, volume ratios, MFI, EOM
    6. Pattern (15 features)     — candlestick recognition
    7. Multi-TF context (8)      — price relative to higher-TF MAs, regime signals
"""

from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from .indicators import IndicatorCalculator
from .derived import DerivedFeatures


class FeatureEngine:
    """Orchestrates feature computation for live trading and backtesting.

    Live mode:
        engine = FeatureEngine(config)
        features = engine.process_candle(candle_1h, candle_4h, candle_1d)
        # features is a dict (or pd.Series) of 60+ values

    Backtest mode:
        df_1h, df_4h, df_1d = load_data(...)
        df_features = engine.bulk_compute(df_1h, df_4h, df_1d)
        # df_features is a DataFrame with all feature columns

    Internal state:
        Maintains rolling window DataFrames per timeframe for live mode.
        Each new candle is appended; old candles are pruned to a max window.
    """

    def __init__(self, config: dict):
        feat_cfg = config.get("features", {})
        self.max_window = feat_cfg.get("max_window_bars", 500)
        self.min_bars_required = feat_cfg.get("min_bars_required", 50)

        # MACD parameters for trend features
        macd_cfg = config.get("strategies", {}).get("mtf_macd_elder", {}).get("macd", {})
        if not macd_cfg:
            macd_cfg = config.get("features", {}).get("macd", {})
        self.macd_fast = macd_cfg.get("fast", 12)
        self.macd_slow = macd_cfg.get("slow", 26)
        self.macd_signal = macd_cfg.get("signal", 9)

        self.indicators = IndicatorCalculator()
        self.derived = DerivedFeatures()

        # Rolling cache: {timeframe: DataFrame}
        self._cache: dict[str, pd.DataFrame] = {
            "1h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            "4h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            "1d": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        }

        # Track which features were computed (for consistent ordering)
        self.feature_names: list[str] = []

    # ─── Live Mode ─────────────────────────────────────────────

    def process_candle(
        self,
        candle_1h: dict,
        candle_4h: Optional[dict] = None,
        candle_1d: Optional[dict] = None,
    ) -> dict:
        """Process a single 1H candle and return the full feature vector.

        Updates internal cache. Returns empty dict if not enough history.
        """
        # Append to cache
        self._append_to_cache("1h", candle_1h)
        if candle_4h:
            self._append_to_cache("4h", candle_4h)
        if candle_1d:
            self._append_to_cache("1d", candle_1d)

        # Need minimum bars to compute meaningful features
        df = self._cache["1h"]
        if len(df) < self.min_bars_required:
            return {}

        # Compute features from the full cached DataFrame
        features_df = self._compute_all_features(df)

        # Return only the last row (current candle)
        if features_df.empty:
            return {}

        last_row = features_df.iloc[-1].to_dict()
        self.feature_names = list(last_row.keys())
        return last_row

    # ─── Backtest Mode ─────────────────────────────────────────

    def bulk_compute(
        self,
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute all features for a full historical DataFrame.

        For backtesting and model training. The returned DataFrame has the
        same index as df_1h, aligned to 1H candles.

        Args:
            df_1h: DataFrame with columns [timestamp, open, high, low, close, volume]
            df_4h: Optional 4H DataFrame for multi-TF features
            df_1d: Optional 1D DataFrame for multi-TF features

        Returns:
            DataFrame with all feature columns, same row count as df_1h.
            Rows at the beginning (before min_bars_required) will have NaN
            for rolling features.
        """
        features = self._compute_all_features(df_1h)

        # Multi-TF features
        if df_4h is not None and not df_4h.empty:
            features = self._add_multitf_features(
                features, df_1h, df_4h, "4h"
            )

        if df_1d is not None and not df_1d.empty:
            features = self._add_multitf_features(
                features, df_1h, df_1d, "1d"
            )

        self.feature_names = list(features.columns)
        return features

    # ─── Internal: Feature Assembly ────────────────────────────

    def _compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute every feature category on a DataFrame.

        Returns a DataFrame with all feature columns (index preserved).
        Original OHLCV columns are dropped — the output is features only.
        """
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]
        v = df["volume"]

        features = pd.DataFrame(index=df.index)

        # 1. Price-based
        self._add_price_features(features, o, h, l, c)

        # 2. Volatility
        self._add_volatility_features(features, o, h, l, c)

        # 3. Trend
        self._add_trend_features(features, o, h, l, c)

        # 4. Momentum
        self._add_momentum_features(features, o, h, l, c, v)

        # 5. Volume
        self._add_volume_features(features, o, h, l, c, v)

        # 6. Candlestick patterns
        self._add_pattern_features(features, o, h, l, c)

        # 7. Market structure
        self._add_structure_features(features, o, h, l, c)

        return features

    # ─── Feature Groups ────────────────────────────────────────

    def _add_price_features(self, f, o, h, l, c):
        """1. Price-based features (8)."""
        f["returns"] = c.pct_change()
        f["log_returns"] = np.log(c / c.shift(1))
        f["hl_ratio"] = (h - l) / c
        f["oc_ratio"] = (c - o) / o
        f["hl_range"] = h / l
        f["close_position"] = (c - l) / (h - l + 1e-12)  # 0=low, 1=high
        f["gap"] = (o - c.shift(1)) / c.shift(1)  # overnight/mid-candle gap
        f["price"] = c

    def _add_volatility_features(self, f, o, h, l, c):
        """2. Volatility features (6)."""
        # ATR(14)
        f["atr_14"] = self.indicators.atr(h, l, c, 14)
        f["atr_pct"] = f["atr_14"] / c * 100

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self.indicators.bollinger_bands(c, 20, 2)
        f["bb_width"] = (bb_upper - bb_lower) / bb_mid
        f["bb_position"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-12)

        # Historical volatility (annualized)
        f["volatility_20"] = self.indicators.historical_volatility(c, 20)

        # Garman-Klass volatility estimator
        f["garman_klass"] = self.indicators.garman_klass(o, h, l, c, 20)

    def _add_trend_features(self, f, o, h, l, c):
        """3. Trend features (12)."""
        # MACD family
        macd_line, macd_signal, macd_hist = self.indicators.macd(c, self.macd_fast, self.macd_slow, self.macd_signal)
        f["macd"] = macd_line
        f["macd_signal"] = macd_signal
        f["macd_hist"] = macd_hist
        f["macd_cross"] = self.indicators.macd_cross(macd_line, macd_signal)

        # ADX
        f["adx_14"] = self.indicators.adx(h, l, c, 14)
        f["di_plus"], f["di_minus"] = self.indicators.di(h, l, c, 14)
        f["di_ratio"] = f["di_plus"] / (f["di_minus"] + 1e-12)

        # EMA slopes (percentage change over 5 bars)
        ema_20 = self.indicators.ema(c, 20)
        ema_50 = self.indicators.ema(c, 50)
        f["ema_20_slope"] = (ema_20 - ema_20.shift(5)) / (ema_20.shift(5) + 1e-12)
        f["ema_50_slope"] = (ema_50 - ema_50.shift(5)) / (ema_50.shift(5) + 1e-12)

        # SMA crosses
        sma_20 = self.indicators.sma(c, 20)
        sma_50 = self.indicators.sma(c, 50)
        sma_200 = self.indicators.sma(c, 200)
        f["sma_20_50"] = (sma_20 - sma_50) / (sma_50 + 1e-12)
        f["sma_50_200"] = (sma_50 - sma_200) / (sma_200 + 1e-12)

    def _add_momentum_features(self, f, o, h, l, c, v):
        """4. Momentum features (10)."""
        f["rsi_14"] = self.indicators.rsi(c, 14)
        f["rsi_7"] = self.indicators.rsi(c, 7)

        # Stochastic
        stoch_k, stoch_d = self.indicators.stochastic(h, l, c, 14, 3)
        f["stoch_k"] = stoch_k
        f["stoch_d"] = stoch_d

        # CCI
        f["cci_20"] = self.indicators.cci(h, l, c, 20)

        # Williams %R
        f["willr_14"] = self.indicators.williams_r(h, l, c, 14)

        # Rate of Change
        f["roc_5"] = self.indicators.roc(c, 5)
        f["roc_10"] = self.indicators.roc(c, 10)
        f["roc_20"] = self.indicators.roc(c, 20)

        # Price momentum (absolute)
        f["momentum_10"] = c - c.shift(10)

    def _add_volume_features(self, f, o, h, l, c, v):
        """5. Volume features (7)."""
        f["obv"] = self.indicators.obv(c, v)
        f["volume_ratio"] = v / v.shift(1)
        f["volume_sma_ratio"] = v / self.indicators.sma(v, 20)
        f["volume_std_ratio"] = v / (v.rolling(20).std() + 1e-12)

        # MFI (Money Flow Index)
        f["mfi_14"] = self.indicators.mfi(h, l, c, v, 14)

        # Ease of Movement
        f["eom"] = self.indicators.ease_of_movement(h, l, v, 14)

        # Relative volume (vs 4-week average)
        f["rel_volume_20"] = v / v.rolling(20).mean()

    def _add_pattern_features(self, f, o, h, l, c):
        """6. Candlestick pattern features (15)."""
        patterns = self.derived.detect_patterns(o, h, l, c)

        for name, col in patterns.items():
            f[f"pattern_{name}"] = col

    def _add_structure_features(self, f, o, h, l, c):
        """7. Market structure features (8)."""
        # Distance from moving averages
        f["dist_sma_20"] = (c - self.indicators.sma(c, 20)) / (self.indicators.sma(c, 20) + 1e-12)
        f["dist_sma_50"] = (c - self.indicators.sma(c, 50)) / (self.indicators.sma(c, 50) + 1e-12)
        f["dist_sma_200"] = (c - self.indicators.sma(c, 200)) / (self.indicators.sma(c, 200) + 1e-12)

        # Rolling max/min
        f["high_20"] = h.rolling(20).max()
        f["low_20"] = l.rolling(20).min()
        f["dist_high_20"] = (f["high_20"] - c) / (f["high_20"] + 1e-12)
        f["dist_low_20"] = (c - f["low_20"]) / (f["low_20"] + 1e-12)

        # N-bar return
        f["return_5"] = c.pct_change(5)
        f["return_10"] = c.pct_change(10)
        f["return_20"] = c.pct_change(20)

    # ─── Multi-Timeframe Features ──────────────────────────────

    def _add_multitf_features(
        self,
        features: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_higher: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        """Add features comparing 1H price to higher-TF levels.

        For each 1H candle, finds the preceding higher-TF candle and
        computes relative position. Backtest-safe: only uses preceding bars.
        """
        if df_higher.empty or "close" not in df_higher.columns:
            return features

        # Get the last higher-TF candle before each 1H timestamp
        c_ht = df_higher["close"].reindex(
            df_1h.index, method="ffill"
        )
        h_ht = df_higher["high"].reindex(
            df_1h.index, method="ffill"
        )
        l_ht = df_higher["low"].reindex(
            df_1h.index, method="ffill"
        )

        c_1h = df_1h["close"]

        features[f"vs_{label}_close"] = (c_1h - c_ht) / (c_ht + 1e-12)
        features[f"vs_{label}_high"] = (c_1h - h_ht) / (h_ht + 1e-12)
        features[f"vs_{label}_low"] = (c_1h - l_ht) / (l_ht + 1e-12)

        # Simple MA of higher-TF (forward-fill to 1H grid)
        sma_20_ht = c_ht.rolling(20).mean()
        features[f"vs_{label}_sma20"] = (c_1h - sma_20_ht) / (sma_20_ht + 1e-12)

        return features

    # ─── Cache Management ──────────────────────────────────────

    def _append_to_cache(self, timeframe: str, candle: dict):
        """Add a candle to the rolling cache DataFrame."""
        df = self._cache[timeframe]
        new_row = pd.DataFrame([{
            "timestamp": candle["timestamp"],
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
        }])

        # Avoid duplicates
        if not df.empty and candle["timestamp"] in df["timestamp"].values:
            df.loc[df["timestamp"] == candle["timestamp"]] = new_row.iloc[0]
        else:
            df = pd.concat([df, new_row], ignore_index=True)

        # Prune
        if len(df) > self.max_window:
            df = df.iloc[-self.max_window:]

        self._cache[timeframe] = df

    def prime_cache(self, df_1h: pd.DataFrame, df_4h=None, df_1d=None):
        """Initialize the rolling cache with historical data."""
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        self._cache["1h"] = df_1h[required_cols].tail(self.max_window).reset_index(drop=True)
        if df_4h is not None:
            self._cache["4h"] = df_4h[required_cols].tail(self.max_window).reset_index(drop=True)
        if df_1d is not None:
            self._cache["1d"] = df_1d[required_cols].tail(self.max_window).reset_index(drop=True)

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of feature names from the last computation."""
        return self.feature_names

    def reset(self):
        """Clear all cached data."""
        for tf in self._cache:
            self._cache[tf] = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        self.feature_names = []
