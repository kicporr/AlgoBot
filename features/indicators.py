"""Technical indicator calculator — pure pandas/numpy implementations.

Zero external TA dependencies. Every indicator follows the documented formula.
All methods are static vectorized functions that take pandas Series
and return pandas Series (or tuple of Series).

Performance: all calculations use pandas rolling/ewm, which internally
uses numpy vectorized operations. Suitable for datasets up to ~1M rows.
"""

import pandas as pd
import numpy as np


class IndicatorCalculator:
    """Collection of technical indicators as static vectorized methods."""

    # ─── Moving Averages ───────────────────────────────────────

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average using pandas ewm."""
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average (linearly weighted)."""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True,
        )

    # ─── Volatility ────────────────────────────────────────────

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range with Wilder's smoothing."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing: ATR = (prev_ATR * (n-1) + TR) / n
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return atr

    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
        """Bollinger Bands. Returns (upper, middle, lower)."""
        mid = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std(ddof=0)
        upper = mid + num_std * std
        lower = mid - num_std * std
        return upper, mid, lower

    @staticmethod
    def historical_volatility(close: pd.Series, period: int = 20, trading_periods: int = 365 * 24) -> pd.Series:
        """Annualized historical volatility from log returns."""
        log_ret = np.log(close / close.shift(1))
        return log_ret.rolling(window=period, min_periods=period).std() * np.sqrt(trading_periods)

    @staticmethod
    def garman_klass(
        open_: pd.Series, high: pd.Series, low: pd.Series,
        close: pd.Series, period: int = 20
    ) -> pd.Series:
        """Garman-Klass volatility estimator.

        GK = 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        gk_sq = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        gk_sq = gk_sq.clip(lower=0)  # Prevent negative from numerical noise
        gk = np.sqrt(gk_sq.rolling(window=period, min_periods=period).mean())
        return gk

    @staticmethod
    def parkinson(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """Parkinson volatility estimator (high-low range).

        P = (1 / (4*ln(2))) * (ln(H/L))^2
        """
        log_hl = np.log(high / low)
        p_sq = (1.0 / (4.0 * np.log(2.0))) * log_hl**2
        return np.sqrt(p_sq.rolling(window=period, min_periods=period).mean())

    # ─── Trend ─────────────────────────────────────────────────

    @staticmethod
    def macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ):
        """MACD. Returns (macd_line, signal_line, histogram)."""
        ema_fast = IndicatorCalculator.ema(close, fast)
        ema_slow = IndicatorCalculator.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = IndicatorCalculator.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def macd_cross(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
        """MACD crossover signal: +1=bullish cross, -1=bearish cross, 0=none."""
        prev_diff = (macd_line - signal_line).shift(1)
        curr_diff = macd_line - signal_line

        cross = pd.Series(0, index=macd_line.index)
        cross[(prev_diff < 0) & (curr_diff > 0)] = 1   # Bullish cross
        cross[(prev_diff > 0) & (curr_diff < 0)] = -1  # Bearish cross
        return cross

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index."""
        plus_dm, minus_dm = IndicatorCalculator._directional_movement(high, low)

        tr = IndicatorCalculator._true_range(high, low, close)

        # Smooth with Wilder's method
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
        adx = dx.ewm(alpha=1 / period, min_periods=period * 2, adjust=False).mean()
        return adx

    @staticmethod
    def di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Directional Indicators. Returns (di_plus, di_minus)."""
        plus_dm, minus_dm = IndicatorCalculator._directional_movement(high, low)
        tr = IndicatorCalculator._true_range(high, low, close)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        di_plus = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
        di_minus = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
        return di_plus, di_minus

    @staticmethod
    def _directional_movement(high: pd.Series, low: pd.Series):
        """Compute +DM and -DM."""
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        mask_plus = (up_move > down_move) & (up_move > 0)
        mask_minus = (down_move > up_move) & (down_move > 0)

        plus_dm[mask_plus] = up_move[mask_plus]
        minus_dm[mask_minus] = down_move[mask_minus]

        return plus_dm, minus_dm

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ─── Momentum ──────────────────────────────────────────────

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's smoothing)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3):
        """Stochastic Oscillator. Returns (%K, %D)."""
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()

        k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
        d = k.rolling(window=d_period, min_periods=d_period).mean()
        return k, d

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        tp = (high + low + close) / 3.0
        sma_tp = IndicatorCalculator.sma(tp, period)
        mean_dev = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True,
        )
        cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-12)
        return cci

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()
        wr = -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-12)
        return wr

    @staticmethod
    def roc(close: pd.Series, period: int) -> pd.Series:
        """Rate of Change: (C - C[n]) / C[n] * 100."""
        return (close - close.shift(period)) / close.shift(period) * 100.0

    # ─── Volume ────────────────────────────────────────────────

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        obv = (volume * direction).cumsum()
        return obv

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        tp = (high + low + close) / 3.0
        raw_money_flow = tp * volume

        tp_prev = tp.shift(1)
        positive_flow = pd.Series(0.0, index=tp.index)
        negative_flow = pd.Series(0.0, index=tp.index)

        positive_flow[tp > tp_prev] = raw_money_flow[tp > tp_prev]
        negative_flow[tp < tp_prev] = raw_money_flow[tp < tp_prev]

        pos_sum = positive_flow.rolling(window=period, min_periods=period).sum()
        neg_sum = negative_flow.rolling(window=period, min_periods=period).sum()

        money_ratio = pos_sum / (neg_sum + 1e-12)
        mfi = 100.0 - (100.0 / (1.0 + money_ratio))
        return mfi

    @staticmethod
    def ease_of_movement(
        high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14
    ) -> pd.Series:
        """Ease of Movement indicator."""
        midpoint = (high + low) / 2.0
        prev_midpoint = midpoint.shift(1)
        box_ratio = volume / (high - low + 1e-12)
        emv_raw = (midpoint - prev_midpoint) / (box_ratio + 1e-12)
        eom = emv_raw.rolling(window=period, min_periods=period).mean()
        return eom
