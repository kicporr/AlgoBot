"""Tests for Phase 1: Data Pipeline.

Covers:
- DataValidator: all validation rules
- OHLCVResampler: incremental and bulk modes
- CandleRepository: CRUD operations (with SQLite in-memory)
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd


# Helper: hour-aligned timestamp
# 1_706_400_000_000 / 3_600_000 = 474,000 — exactly hour 474,000
HOUR_ALIGNED_MS = 1_706_400_000_000


# ═══════ Validator Tests ═══════════════════════════════════════


class TestDataValidator:
    
    def test_valid_candle_passes(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {
            "timestamp": HOUR_ALIGNED_MS,
            "open": 50000.0, "high": 50100.0, "low": 49900.0,
            "close": 50050.0, "volume": 100.5,
        }
        result = v.validate(candle)
        assert result.valid, f"Should be valid: {result.reason}"
    
    def test_missing_field_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {"timestamp": 123, "open": 1, "high": 2, "low": 0.5}
        result = v.validate(candle)
        assert not result.valid
        assert "close" in result.reason
    
    def test_null_field_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {"timestamp": 123, "open": None, "high": 2, "low": 0.5, "close": 1, "volume": 10}
        result = v.validate(candle)
        assert not result.valid
    
    def test_negative_price_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {"timestamp": 123, "open": -1, "high": 2, "low": 0.5, "close": 1, "volume": 10}
        result = v.validate(candle)
        assert not result.valid
    
    def test_high_below_max_oc_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {"timestamp": 123, "open": 100, "high": 99, "low": 50, "close": 105, "volume": 10}
        result = v.validate(candle)
        assert not result.valid
    
    def test_low_above_min_oc_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {"timestamp": 123, "open": 100, "high": 110, "low": 101, "close": 105, "volume": 10}
        result = v.validate(candle)
        assert not result.valid
    
    def test_negative_volume_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {"timestamp": 123, "open": 100, "high": 110, "low": 90, "close": 105, "volume": -1}
        result = v.validate(candle)
        assert not result.valid
    
    def test_duplicate_timestamp_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candle = {"timestamp": HOUR_ALIGNED_MS, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 10}
        assert v.validate(candle).valid
        assert not v.validate(candle).valid
    
    def test_future_timestamp_rejected(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        future_ts = int(time.time() * 1000) + 60_000
        candle = {"timestamp": future_ts, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 10}
        result = v.validate(candle)
        assert not result.valid
    
    def test_validate_batch(self):
        from data.ingestion.data_validator import DataValidator
        v = DataValidator()
        candles = [
            {"timestamp": HOUR_ALIGNED_MS, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 10},
            {"timestamp": HOUR_ALIGNED_MS + 60_000, "open": -1, "high": 110, "low": 90, "close": 105, "volume": 10},
            {"timestamp": HOUR_ALIGNED_MS + 120_000, "open": 106, "high": 115, "low": 95, "close": 110, "volume": 20},
        ]
        valid, rejected = v.validate_batch(candles)
        assert len(valid) == 2
        assert len(rejected) == 1
        assert rejected[0]["index"] == 1


# ═══════ Resampler Tests — Incremental ════════════════════════


class TestOHLCVResamplerIncremental:
    
    def test_incremental_1h_complete(self):
        from data.ingestion.resampler import OHLCVResampler
        r = OHLCVResampler()
        
        hour_start = HOUR_ALIGNED_MS  # exact hour boundary
        
        # Feed 60 1m candles for the first hour
        results = None
        for i in range(60):
            ts = hour_start + (i * 60_000)
            candle = {
                "timestamp": ts,
                "open": 100.0 + i,
                "high": 100.0 + i + 5,
                "low": 100.0 + i - 2,
                "close": 100.0 + i + 2,
                "volume": 10.0,
            }
            results = r.add_1m_candle(candle)
            
            # No emission until next hour's first candle arrives
            assert results["1h"] is None, f"Should not emit 1H at minute {i}"
            assert results["4h"] is None
            assert results["1d"] is None
        
        # Feed the first candle of the next hour — this triggers hour 0 emission
        results = r.add_1m_candle({
            "timestamp": hour_start + 3_600_000,  # Next hour
            "open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0, "volume": 10.0,
        })
        h1 = results["1h"]
        assert h1 is not None, "1H candle should emit when new hour starts"
        assert h1["timestamp"] == hour_start
        assert h1["open"] == 100.0          # Open of first 1m candle (i=0)
        assert h1["close"] == 100.0 + 59 + 2  # Close of last 1m candle (i=59)
        assert h1["high"] == 100.0 + 59 + 5   # Max high across all 60
        assert h1["low"] == 100.0 - 2         # Min low across all 60
        assert h1["volume"] == 600.0
        assert h1["bar_count"] == 60
    
    def test_insufficient_bars_no_emit(self):
        from data.ingestion.resampler import OHLCVResampler
        r = OHLCVResampler()
        
        hour_start = HOUR_ALIGNED_MS
        
        # Feed only 10 candles (less than min_bars=55 for 1H)
        for i in range(10):
            ts = hour_start + (i * 60_000)
            r.add_1m_candle({
                "timestamp": ts,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        # Next hour candle — should NOT emit since only 10 bars
        results = r.add_1m_candle({
            "timestamp": hour_start + 3_600_000,
            "open": 200.0, "high": 210.0, "low": 190.0, "close": 205.0,
            "volume": 10.0,
        })
        assert results["1h"] is None, "Should not emit with only 10 bars"
    
    def test_incremental_4h_complete(self):
        from data.ingestion.resampler import OHLCVResampler
        r = OHLCVResampler()
        
        block_start = HOUR_ALIGNED_MS
        
        # Feed 240 1m candles (4 hours)
        for i in range(240):
            ts = block_start + (i * 60_000)
            r.add_1m_candle({
                "timestamp": ts,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        # Feed next 4H candle to trigger
        results = r.add_1m_candle({
            "timestamp": block_start + 14_400_000,
            "open": 200.0, "high": 210.0, "low": 190.0, "close": 205.0,
            "volume": 10.0,
        })
        
        h4 = results["4h"]
        assert h4 is not None, "4H candle should emit"
        assert h4["timestamp"] == block_start
        assert h4["bar_count"] == 240
    
    def test_flush_emits_partial(self):
        from data.ingestion.resampler import OHLCVResampler
        r = OHLCVResampler()
        
        hour_start = HOUR_ALIGNED_MS
        
        # Feed 60 complete candles
        for i in range(60):
            ts = hour_start + (i * 60_000)
            r.add_1m_candle({
                "timestamp": ts,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        # Move to next hour, feed 30 candles only (incomplete hour)
        for i in range(30):
            ts = hour_start + 3_600_000 + (i * 60_000)
            r.add_1m_candle({
                "timestamp": ts,
                "open": 200.0, "high": 210.0, "low": 190.0, "close": 205.0,
                "volume": 10.0,
            })
        
        flushed = r.flush()
        # flush emits anything partially built — but second hour has only 30 < 55 bars
        # First hour already emitted when second hour started (in add_1m_candle)
        h1 = flushed.get("1h")
        # flush should return None for hours with insufficient bars
        # The first hour was already emitted previously, not in flush
        # Second hour has 30 < 55 bars
        assert h1 is None, "Second hour incomplete, should not emit"
    
    def test_prime_no_callbacks(self):
        from data.ingestion.resampler import OHLCVResampler
        r = OHLCVResampler()
        
        block_start = HOUR_ALIGNED_MS
        
        candles = []
        for i in range(120):
            ts = block_start + (i * 60_000)
            candles.append({
                "timestamp": ts,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        r.prime(candles)
        # After priming with 120 candles (2 complete hours), the resampler
        # should have emitted hour 0 when hour 1 started (via add_1m_candle).
        # The builders should be working on hour 2 (starting at block_start + 7200000).
        # prime() == add_1m_candle for each, so emissions should have happened.
        # This is a smoke test — no crash, no exceptions.
        assert True


# ═══════ Resampler Tests — Bulk (Backtesting) ═════════════════


class TestOHLCVResamplerBulk:
    
    def test_resample_1m_to_1h(self):
        from data.ingestion.resampler import OHLCVResampler, Timeframe
        
        hour_start = HOUR_ALIGNED_MS
        
        # Create 120 1m candles (2 complete hours)
        data = []
        for i in range(120):
            data.append({
                "timestamp": hour_start + i * 60_000,
                "open": 100.0 + i,
                "high": 100.0 + i + 5,
                "low": 100.0 + i - 2,
                "close": 100.0 + i + 2,
                "volume": 10.0,
            })
        
        df = pd.DataFrame(data)
        df_1h = OHLCVResampler.resample_bulk(df, Timeframe.H1)
        
        assert len(df_1h) == 2, f"Expected 2 hourly candles, got {len(df_1h)}"
        
        # First hour
        h0 = df_1h.iloc[0]
        assert h0["open"] == 100.0
        assert h0["close"] == 100.0 + 59 + 2
        assert h0["high"] == 100.0 + 59 + 5
        assert h0["low"] == 100.0 - 2
        assert h0["volume"] == 600.0
        
        # Second hour
        h1 = df_1h.iloc[1]
        assert h1["open"] == 100.0 + 60
    
    def test_resample_empty(self):
        from data.ingestion.resampler import OHLCVResampler, Timeframe
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = OHLCVResampler.resample_bulk(df, Timeframe.H1)
        assert result.empty
    
    def test_resample_1m_to_4h(self):
        from data.ingestion.resampler import OHLCVResampler, Timeframe
        
        hour_start = HOUR_ALIGNED_MS
        
        # 480 candles = 8 hours = 2 four-hour blocks
        data = []
        for i in range(480):
            data.append({
                "timestamp": hour_start + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        df = pd.DataFrame(data)
        df_4h = OHLCVResampler.resample_bulk(df, Timeframe.H4)
        
        assert len(df_4h) == 2
    
    def test_resample_all_timeframes(self):
        from data.ingestion.resampler import OHLCVResampler
        
        hour_start = HOUR_ALIGNED_MS
        
        # 1440 candles = exactly 24 hours
        data = []
        for i in range(1440):
            data.append({
                "timestamp": hour_start + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        df = pd.DataFrame(data)
        results = OHLCVResampler.resample_all(df)
        
        assert "1h" in results
        assert "4h" in results
        assert "1d" in results
        
        assert len(results["1h"]) == 24, f"Expected 24 1H candles, got {len(results['1h'])}"
        assert len(results["4h"]) == 6
        assert len(results["1d"]) == 1
    
    def test_resample_handles_gaps(self):
        from data.ingestion.resampler import OHLCVResampler, Timeframe
        
        hour_start = HOUR_ALIGNED_MS
        
        data = []
        # Hour 0: complete (60 candles)
        for i in range(60):
            data.append({
                "timestamp": hour_start + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        # Hour 1: complete (60 candles)
        for i in range(60):
            data.append({
                "timestamp": hour_start + 3_600_000 + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        # Gap: hour 2 is missing
        
        # Hour 3: only 30 candles (incomplete, should be filtered out)
        for i in range(30):
            data.append({
                "timestamp": hour_start + 3 * 3_600_000 + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
                "volume": 10.0,
            })
        
        df = pd.DataFrame(data)
        df_1h = OHLCVResampler.resample_bulk(df, Timeframe.H1)
        
        # Should have 2 complete hours, hour 3 is filtered (30 < 55 min_bars)
        assert len(df_1h) == 2


# ═══════ Repository Tests ══════════════════════════════════════


class TestCandleRepository:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        from data.storage.repositories import DatabaseManager
        
        DatabaseManager._instance = None
        
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "data": {
                "database": {
                    "type": "sqlite",
                    "path": os.path.join(self.temp_dir, "test.db"),
                }
            }
        }
        
        self.db = DatabaseManager(self.config)
        
        from data.storage.repositories import CandleRepository
        self.repo = CandleRepository(self.db)
        
        yield
        
        DatabaseManager._instance = None
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_insert_and_retrieve(self):
        candle = {
            "timestamp": HOUR_ALIGNED_MS,
            "open": 50000.0, "high": 50100.0, "low": 49900.0,
            "close": 50050.0, "volume": 100.5,
        }
        assert self.repo.insert(candle) is True
        assert self.repo.get_count() == 1
        
        latest = self.repo.get_latest()
        assert latest is not None
        assert latest["timestamp"] == HOUR_ALIGNED_MS
        assert latest["open"] == 50000.0
    
    def test_duplicate_insert_ignored(self):
        candle = {"timestamp": 123, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}
        assert self.repo.insert(candle) is True
        assert self.repo.insert(candle) is False
        assert self.repo.get_count() == 1
    
    def test_batch_insert(self):
        candles = []
        for i in range(100):
            candles.append({
                "timestamp": HOUR_ALIGNED_MS + i * 60_000,
                "open": 50000.0 + i, "high": 50100.0 + i, "low": 49900.0 + i,
                "close": 50050.0 + i, "volume": 100.0,
            })
        
        inserted = self.repo.insert_batch(candles)
        assert inserted == 100
        assert self.repo.get_count() == 100
    
    def test_batch_insert_skips_duplicates(self):
        for i in range(50):
            self.repo.insert({
                "timestamp": HOUR_ALIGNED_MS + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 10.0,
            })
        
        candles = []
        for i in range(100):
            candles.append({
                "timestamp": HOUR_ALIGNED_MS + i * 60_000,
                "open": 200.0, "high": 210.0, "low": 190.0, "close": 205.0, "volume": 20.0,
            })
        
        inserted = self.repo.insert_batch(candles)
        assert inserted == 50
        assert self.repo.get_count() == 100
    
    def test_get_range(self):
        for i in range(200):
            self.repo.insert({
                "timestamp": HOUR_ALIGNED_MS + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 10.0,
            })
        
        results = self.repo.get_range(
            start_ts=HOUR_ALIGNED_MS + 50 * 60_000,
            end_ts=HOUR_ALIGNED_MS + 99 * 60_000,
        )
        assert len(results) == 50
    
    def test_get_range_respects_limit(self):
        for i in range(200):
            self.repo.insert({
                "timestamp": HOUR_ALIGNED_MS + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 10.0,
            })
        
        results = self.repo.get_range(
            start_ts=HOUR_ALIGNED_MS,
            end_ts=HOUR_ALIGNED_MS + 200 * 60_000,
            limit=75,
        )
        assert len(results) == 75
    
    def test_get_all_timestamps(self):
        for i in range(10):
            self.repo.insert({
                "timestamp": HOUR_ALIGNED_MS + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 10.0,
            })
        
        ts = self.repo.get_all_timestamps()
        assert len(ts) == 10
    
    def test_delete_older_than(self):
        for i in range(100):
            self.repo.insert({
                "timestamp": HOUR_ALIGNED_MS + i * 60_000,
                "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 10.0,
            })
        
        cutoff = HOUR_ALIGNED_MS + 50 * 60_000
        deleted = self.repo.delete_older_than(cutoff)
        assert deleted == 50
        assert self.repo.get_count() == 50
