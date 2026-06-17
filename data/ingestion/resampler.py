"""OHLCV Resampler: 1m → 1H, 4H, 1D.

Two modes:
1. Incremental (live trading): feeds 1m candles one-by-one, emits higher-TF
   candles when a period completes.
2. Bulk (backtesting): resamples an entire DataFrame at once using pandas.

Critical for backtesting correctness:
- Uses OPEN-time timestamps (matching Binance convention)
- Never looks ahead — a higher-TF candle is only emitted AFTER its last
  constituent 1m candle has arrived.
- Handles gaps gracefully (exchange downtime, missing candles).
"""

from typing import Optional
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
from loguru import logger


class Timeframe(Enum):
    M1 = "1m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    
    @property
    def duration_ms(self) -> int:
        """Duration of one candle in milliseconds."""
        return {
            Timeframe.M1: 60_000,
            Timeframe.H1: 3_600_000,
            Timeframe.H4: 14_400_000,
            Timeframe.D1: 86_400_000,
        }[self]
    
    @property
    def min_bars(self) -> int:
        """Minimum number of 1m bars needed for a valid candle."""
        return {
            Timeframe.M1: 1,
            Timeframe.H1: 55,   # 55 out of 60 minutes (allow 5 min gap)
            Timeframe.H4: 220,  # 220 out of 240 minutes
            Timeframe.D1: 1380, # 1380 out of 1440 minutes (allow 60 min gap)
        }[self]


@dataclass
class CandleBuilder:
    """Accumulates 1m candles to build a higher-timeframe candle."""
    timeframe: Timeframe
    period_start: int = 0          # ms timestamp of period open
    open: float = 0.0
    high: float = float("-inf")
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    bar_count: int = 0
    first_ts: int = 0
    last_ts: int = 0
    
    def is_empty(self) -> bool:
        return self.bar_count == 0
    
    def add(self, candle: dict):
        """Add a 1m candle to this builder."""
        if self.bar_count == 0:
            self.open = candle["open"]
            if self.period_start <= 0:
                dur = self.timeframe.duration_ms
                self.period_start = (candle["timestamp"] // dur) * dur
            self.first_ts = candle["timestamp"]

        self.high = max(self.high, candle["high"])
        self.low = min(self.low, candle["low"])
        self.close = candle["close"]
        self.volume += candle["volume"]
        self.bar_count += 1
        self.last_ts = candle["timestamp"]
    
    def build(self) -> Optional[dict]:
        """Build the final candle, or None if not enough bars."""
        timeframe = self.timeframe
        if self.bar_count < timeframe.min_bars:
            logger.debug(
                f"Insufficient bars for {timeframe.value} candle: "
                f"{self.bar_count}/{timeframe.min_bars} at ts={self.period_start}"
            )
            return None
        
        return {
            "timestamp": self.period_start,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "bar_count": self.bar_count,
            "timeframe": timeframe.value,
        }
    
    def reset(self, new_period_start: int):
        """Reset builder for a new period."""
        self.period_start = new_period_start
        self.open = 0.0
        self.high = float("-inf")
        self.low = float("inf")
        self.close = 0.0
        self.volume = 0.0
        self.bar_count = 0
        self.first_ts = 0
        self.last_ts = 0


class OHLCVResampler:
    """Resamples 1m OHLCV candles to higher timeframes.
    
    Incremental mode (live trading):
        resampler = OHLCVResampler()
        for candle_1m in stream:
            new_candles = resampler.add_1m_candle(candle_1m)
            # new_candles is dict: {"1h": candle_or_None, "4h": ..., "1d": ...}
    
    Bulk mode (backtesting):
        resampler = OHLCVResampler()
        candles_1h = resampler.resample_bulk(df_1m, Timeframe.H1)
    """
    
    def __init__(self):
        self._builders: dict[Timeframe, CandleBuilder] = {
            Timeframe.H1: CandleBuilder(Timeframe.H1),
            Timeframe.H4: CandleBuilder(Timeframe.H4),
            Timeframe.D1: CandleBuilder(Timeframe.D1),
        }
        # Track which periods have been emitted (prevents double-emit)
        self._emitted_periods: dict[Timeframe, set[int]] = {
            tf: set() for tf in [Timeframe.H1, Timeframe.H4, Timeframe.D1]
        }
    
    # ─── Incremental Mode ───────────────────────────────────────
    
    def add_1m_candle(self, candle: dict) -> dict[str, Optional[dict]]:
        """Feed a single completed 1m candle. Returns any completed higher-TF candles.
        
        Returns:
            {"1h": candle_dict_or_None, "4h": ..., "1d": ...}
            All keys are always present; None means no candle completed.
        """
        ts = candle["timestamp"]
        results: dict[str, Optional[dict]] = {
            "1h": None,
            "4h": None,
            "1d": None,
        }
        
        for tf, builder in self._builders.items():
            period_start = self._floor_timestamp(ts, tf)
            
            # Check if this 1m candle starts a new period
            if not builder.is_empty() and period_start != builder.period_start:
                # Previous period complete — emit it
                built = builder.build()
                if built and builder.period_start not in self._emitted_periods[tf]:
                    self._emitted_periods[tf].add(builder.period_start)
                    results[tf.value] = built
                    self._prune_emitted(tf)
                
                # Reset for new period
                builder.reset(period_start)
            
            # Add this candle to the current period
            builder.add(candle)
        
        return results
    
    def flush(self) -> dict[str, Optional[dict]]:
        """Flush any partially-built candles. Use when stopping the bot."""
        results: dict[str, Optional[dict]] = {}
        for tf, builder in self._builders.items():
            if not builder.is_empty():
                built = builder.build()
                if built and builder.period_start not in self._emitted_periods[tf]:
                    self._emitted_periods[tf].add(builder.period_start)
                    results[tf.value] = built
                builder.reset(0)
        return results
    
    def prime(self, candles_1m: list[dict]):
        """Prime the resampler with historical 1m candles.
        
        Call this on startup to initialize the builders before live WebSocket
        data starts flowing. Prevents emitting incomplete candles.
        """
        for candle in candles_1m:
            self.add_1m_candle(candle)
    
    # ─── Bulk Mode (Backtesting) ────────────────────────────────
    
    @staticmethod
    def resample_bulk(df_1m: pd.DataFrame, target_tf: Timeframe) -> pd.DataFrame:
        """Resample a DataFrame of 1m candles to a higher timeframe.
        
        Args:
            df_1m: DataFrame with columns [timestamp, open, high, low, close, volume]
                   Timestamp must be in milliseconds.
            target_tf: Target timeframe (H1, H4, or D1).
        
        Returns:
            DataFrame with resampled candles, timestamp as index.
        """
        if df_1m.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "bar_count"])
        
        df = df_1m.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        
        freq_map = {
            Timeframe.H1: "1h",
            Timeframe.H4: "4h",
            Timeframe.D1: "1D",
        }
        
        if target_tf not in freq_map:
            raise ValueError(f"Unsupported bulk resample timeframe: {target_tf}")
        
        freq = freq_map[target_tf]
        
        resampled = df.resample(freq, closed="left", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        
        # Count bars per period
        bar_counts = df.resample(freq, closed="left", label="left").size()
        resampled["bar_count"] = bar_counts
        
        # Filter out periods with too few bars
        resampled = resampled[resampled["bar_count"] >= target_tf.min_bars]
        
        # Drop rows with NaN (incomplete periods at edges)
        resampled = resampled.dropna()
        
        # Convert index back to ms timestamps
        resampled = resampled.reset_index()
        resampled["timestamp"] = resampled["timestamp"].astype("int64") // 1_000_000  # ns → ms
        resampled["timeframe"] = target_tf.value
        
        return resampled
    
    @staticmethod
    def resample_all(df_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Resample to all supported higher timeframes at once.
        
        Returns:
            {"1h": DataFrame, "4h": DataFrame, "1d": DataFrame}
        """
        return {
            "1h": OHLCVResampler.resample_bulk(df_1m, Timeframe.H1),
            "4h": OHLCVResampler.resample_bulk(df_1m, Timeframe.H4),
            "1d": OHLCVResampler.resample_bulk(df_1m, Timeframe.D1),
        }
    
    # ─── Helpers ────────────────────────────────────────────────
    
    @staticmethod
    def _floor_timestamp(timestamp_ms: int, timeframe: Timeframe) -> int:
        """Round a timestamp down to the nearest period boundary."""
        dur = timeframe.duration_ms
        return (timestamp_ms // dur) * dur
    
    def _prune_emitted(self, tf: Timeframe):
        """Keep emitted_periods set from growing unboundedly."""
        if len(self._emitted_periods[tf]) > 5000:
            # Keep only the most recent 2500
            sorted_periods = sorted(self._emitted_periods[tf])
            self._emitted_periods[tf] = set(sorted_periods[-2500:])
    
    def reset(self):
        """Reset all state (use when reconnecting or restarting)."""
        for builder in self._builders.values():
            builder.reset(0)
        for tf in self._emitted_periods:
            self._emitted_periods[tf].clear()
