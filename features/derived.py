"""Custom derived features — candlestick patterns, divergences, strength scores.

All pattern detection uses pure price logic — no TA-lib dependency.
Each method returns a pandas Series of 0/1 (absent/present) or continuous values.
"""

import pandas as pd
import numpy as np


class DerivedFeatures:
    """Custom derived features not available in standard TA libraries."""

    # ─── Candlestick Body Types ────────────────────────────────

    @staticmethod
    def _body(o: pd.Series, c: pd.Series) -> pd.Series:
        """Absolute body size."""
        return (c - o).abs()

    @staticmethod
    def _upper_shadow(o: pd.Series, h: pd.Series, c: pd.Series) -> pd.Series:
        """Upper shadow length."""
        body_high = pd.concat([o, c], axis=1).max(axis=1)
        return h - body_high

    @staticmethod
    def _lower_shadow(o: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        """Lower shadow length."""
        body_low = pd.concat([o, c], axis=1).min(axis=1)
        return body_low - l

    @staticmethod
    def _total_range(h: pd.Series, l: pd.Series) -> pd.Series:
        """Total candle range."""
        return h - l

    @staticmethod
    def _is_bullish(o: pd.Series, c: pd.Series) -> pd.Series:
        """Candle is bullish (close > open)."""
        return c > o

    @staticmethod
    def _is_bearish(o: pd.Series, c: pd.Series) -> pd.Series:
        """Candle is bearish (close < open)."""
        return c < o

    # ─── Pattern Detection ─────────────────────────────────────

    @staticmethod
    def detect_patterns(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> dict[str, pd.Series]:
        """Detect all candlestick patterns. Returns dict of {name: Series of 0/1}."""
        body = DerivedFeatures._body(o, c)
        us = DerivedFeatures._upper_shadow(o, h, c)
        ls = DerivedFeatures._lower_shadow(o, l, c)
        tr = DerivedFeatures._total_range(h, l)
        bull = DerivedFeatures._is_bullish(o, c)
        bear = DerivedFeatures._is_bearish(o, c)

        patterns = {}

        # 1. Doji — body < 10% of total range
        doji = (body < tr * 0.1) & (tr > 0)
        patterns["doji"] = doji.astype(int)

        # 2. Long-legged Doji — doji with both shadows > body * 3
        ll_doji = doji & (us > body * 3) & (ls > body * 3)
        patterns["doji_long_legged"] = ll_doji.astype(int)

        # 3. Hammer — bullish reversal: small body at top, long lower shadow
        hammer = (bull & (ls > body * 2) & (us < body * 0.5) & (tr > 0))
        patterns["hammer"] = hammer.astype(int)

        # 4. Shooting Star — bearish reversal: small body at bottom, long upper shadow
        shooting_star = (bear & (us > body * 2) & (ls < body * 0.5) & (tr > 0))
        patterns["shooting_star"] = shooting_star.astype(int)

        # 5. Bullish Engulfing
        prev_bear = bear.shift(1)
        prev_o = o.shift(1)
        prev_c = c.shift(1)
        bull_engulf = (
            bull & prev_bear &
            (o < prev_c) & (c > prev_o) &
            (body > DerivedFeatures._body(prev_o, prev_c))
        )
        patterns["bullish_engulfing"] = bull_engulf.astype(int)

        # 6. Bearish Engulfing
        prev_bull = bull.shift(1)
        bear_engulf = (
            bear & prev_bull &
            (o > prev_c) & (c < prev_o) &
            (body > DerivedFeatures._body(prev_o, prev_c))
        )
        patterns["bearish_engulfing"] = bear_engulf.astype(int)

        # 7. Harami — small body inside previous large body
        prev_body = DerivedFeatures._body(prev_o, prev_c)
        harami = (body < prev_body * 0.5) & (prev_body > 0)
        patterns["harami"] = harami.astype(int)

        # 8. Morning Star — 3-bar bullish reversal pattern
        # Bar1: strong bearish, Bar2: small body (doji), Bar3: strong bullish
        bar1_bearish = bear.shift(2) & (DerivedFeatures._body(o.shift(2), c.shift(2)) > tr.shift(2).median())
        bar2_small = (body.shift(1) < tr.shift(1) * 0.3)
        gap_down = o.shift(1) < c.shift(2)  # Bar2 opens below Bar1 close
        bar3_bullish = bull & (body > tr.median())
        gap_up = o > c.shift(1)  # Bar3 opens above Bar2 close
        morning_star = bar1_bearish & bar2_small & bar3_bullish & gap_down & gap_up
        patterns["morning_star"] = morning_star.astype(int)

        # 9. Evening Star — 3-bar bearish reversal pattern
        bar1_bullish = bull.shift(2) & (DerivedFeatures._body(o.shift(2), c.shift(2)) > tr.shift(2).median())
        gap_up2 = o.shift(1) > c.shift(2)
        bar3_bearish = bear & (body > tr.median())
        gap_down2 = o < c.shift(1)
        evening_star = bar1_bullish & bar2_small & bar3_bearish & gap_up2 & gap_down2
        patterns["evening_star"] = evening_star.astype(int)

        # 10. Three White Soldiers — 3 consecutive strong bullish candles
        soldiers_1 = bull.shift(2) & (body.shift(2) > tr.shift(2).median())
        soldiers_2 = bull.shift(1) & (body.shift(1) > tr.shift(1).median()) & (c.shift(1) > c.shift(2)) & (o.shift(1) > o.shift(2))
        soldiers_3 = bull & (body > tr.median()) & (c > c.shift(1)) & (o > o.shift(1))
        patterns["three_white_soldiers"] = (soldiers_1 & soldiers_2 & soldiers_3).astype(int)

        # 11. Three Black Crows — 3 consecutive strong bearish candles
        crows_1 = bear.shift(2) & (body.shift(2) > tr.shift(2).median())
        crows_2 = bear.shift(1) & (body.shift(1) > tr.shift(1).median()) & (c.shift(1) < c.shift(2)) & (o.shift(1) < o.shift(2))
        crows_3 = bear & (body > tr.median()) & (c < c.shift(1)) & (o < o.shift(1))
        patterns["three_black_crows"] = (crows_1 & crows_2 & crows_3).astype(int)

        # 12. Marubozu — very small shadows relative to body
        marubozu = (us < body * 0.1) & (ls < body * 0.1) & (body > 0)
        patterns["marubozu"] = marubozu.astype(int)

        return patterns

    # ─── RSI Divergence ────────────────────────────────────────

    @staticmethod
    def rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 5) -> pd.Series:
        """Detect RSI divergence.

        Bullish divergence: price makes lower low, RSI makes higher low.
        Bearish divergence: price makes higher high, RSI makes lower high.

        Returns: 1=bullish divergence, -1=bearish divergence, 0=none.
        """
        div = pd.Series(0, index=close.index)

        for i in range(lookback, len(close)):
            rsi_window = rsi.iloc[i - lookback : i + 1]
            price_window = close.iloc[i - lookback : i + 1]

            # Skip if window contains NaN (RSI warm-up period)
            if rsi_window.isna().any():
                continue

            # Bullish: price is at window low but RSI is higher than before
            if price_window.idxmin() == price_window.index[-1]:
                rsi_min_idx = rsi_window.idxmin()
                if rsi_min_idx != price_window.index[-1]:
                    div.iloc[i] = 1

            # Bearish: price is at window high but RSI is lower than before
            if price_window.idxmax() == price_window.index[-1]:
                rsi_max_idx = rsi_window.idxmax()
                if rsi_max_idx != price_window.index[-1]:
                    div.iloc[i] = -1

        return div

    # ─── Trend Strength ────────────────────────────────────────

    @staticmethod
    def trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
        """Measure trend strength as R² of linear regression over the window.

        Returns values 0..1 where 1 = perfect linear trend.
        """
        def r_squared(x):
            if len(x) < period:
                return np.nan
            t = np.arange(len(x))
            slope, intercept = np.polyfit(t, x, 1)
            y_pred = slope * t + intercept
            ss_res = np.sum((x - y_pred) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            if ss_tot == 0:
                return 1.0
            return 1.0 - ss_res / ss_tot

        return close.rolling(window=period, min_periods=period).apply(
            r_squared, raw=True
        )

    # ─── Support / Resistance Proximity ────────────────────────

    @staticmethod
    def pivot_levels(high: pd.Series, low: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
        """Find pivot highs and lows (local extrema).

        Returns (pivot_highs, pivot_lows) — price level at each pivot point,
        NaN elsewhere.
        """
        pivot_highs = pd.Series(np.nan, index=high.index)
        pivot_lows = pd.Series(np.nan, index=low.index)

        for i in range(window, len(high) - window):
            h_slice = high.iloc[i - window : i + window + 1]
            l_slice = low.iloc[i - window : i + window + 1]

            if high.iloc[i] == h_slice.max():
                pivot_highs.iloc[i] = high.iloc[i]
            if low.iloc[i] == l_slice.min():
                pivot_lows.iloc[i] = low.iloc[i]

        return pivot_highs, pivot_lows

    # ─── Gap Detection ─────────────────────────────────────────

    @staticmethod
    def gap_type(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        """Classify gaps.

        Returns: 2=gap up (L > prev_H), -2=gap down (H < prev_L),
                 1=partial up, -1=partial down, 0=no gap.
        """
        prev_h = h.shift(1)
        prev_l = l.shift(1)

        gap = pd.Series(0, index=o.index)
        gap[l > prev_h] = 2      # Full gap up
        gap[h < prev_l] = -2     # Full gap down
        gap[(o > prev_h) & (l <= prev_h)] = 1    # Partial gap up
        gap[(o < prev_l) & (h >= prev_l)] = -1   # Partial gap down

        return gap
