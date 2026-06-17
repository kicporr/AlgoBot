"""Tests for execution modules, specifically PositionTracker exit conditions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from execution.position_tracker import PositionTracker, Position


@pytest.fixture
def base_config():
    return {
        "risk": {
            "per_trade": {
                "stop_loss_pct": 3.0,
                "take_profit_pct": 8.0,
                "trailing_stop_activation": 3.0,
                "trailing_stop_distance": 1.5,
                "volatility_stop_mult": 2.0,
                "max_duration_hours": 48
            }
        }
    }


class TestPositionTracker:
    
    def test_long_entry_and_tp(self, base_config):
        tracker = PositionTracker(base_config)
        # atr_pct = 2.0 -> vol_factor = 1.0
        # sl_mult = 3.0, tp_mult = 2.5
        # risk = sl_mult * atr = 3.0 * 100 = 300
        # sl = 50000 - 300 = 49700
        # tp = 50000 + 2.5 * 300 = 50750
        # trail = 0.025 + 0.01 = 0.035 (3.5%)
        pos = tracker.enter(
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            atr=100.0,
            atr_pct=2.0,
            timestamp=1781494800000
        )
        
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.stop_loss == 49700.0
        assert pos.take_profit == 50750.0
        assert pos.trailing_distance == 0.035
        assert pos.highest_price == 50000.0
        
        # Test update under TP hit
        candle = {
            "open": 50000.0,
            "high": 50800.0,
            "low": 49800.0,
            "close": 50500.0,
            "timestamp": 1781494860000
        }
        res = tracker.update(candle)
        assert res == "take_profit"
        
    def test_long_sl_and_trailing(self, base_config):
        tracker = PositionTracker(base_config)
        tracker.enter(
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            atr=100.0,
            atr_pct=2.0,
            timestamp=1781494800000
        )
        
        # Test ATR stop loss hit
        candle_sl = {
            "open": 50000.0,
            "high": 50100.0,
            "low": 49650.0,
            "close": 49680.0,
            "timestamp": 1781494860000
        }
        res = tracker.update(candle_sl)
        assert res == "atr_stop"
        
        # Test trailing stop trigger
        tracker.exit()
        tracker.enter(
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            atr=100.0,
            atr_pct=2.0,
            timestamp=1781494800000
        )
        tracker.position.take_profit = 999999.0  # Prevent take profit hit
    
        # Raise highest price seen to 52000
        # New trail stop level = 52000 * (1 - 0.035) = 50180
        candle_high = {
            "open": 50000.0,
            "high": 52000.0,
            "low": 50000.0,
            "close": 51500.0,
            "timestamp": 1781494860000
        }
        res = tracker.update(candle_high)
        assert res == "hold"
        assert tracker.position.highest_price == 52000.0
        
        # Next candle low hits 50150 (below trail stop 50180)
        candle_trail_hit = {
            "open": 51500.0,
            "high": 51600.0,
            "low": 50150.0,
            "close": 50200.0,
            "timestamp": 1781494920000
        }
        res = tracker.update(candle_trail_hit)
        assert res == "trailing_stop"
        
    def test_short_entry_and_tp(self, base_config):
        tracker = PositionTracker(base_config)
        # atr_pct = 2.0 -> vol_factor = 1.0
        # sl_mult = 3.0, tp_mult = 2.5
        # risk = sl_mult * atr = 3.0 * 100 = 300
        # sl = 50000 + 300 = 50300
        # tp = 50000 - 2.5 * 300 = 49250
        # trail = 0.025 + 0.01 = 0.035 (3.5%)
        pos = tracker.enter(
            side="short",
            entry_price=50000.0,
            quantity=0.1,
            atr=100.0,
            atr_pct=2.0,
            timestamp=1781494800000
        )
        
        assert pos.side == "short"
        assert pos.entry_price == 50000.0
        assert pos.stop_loss == 50300.0
        assert pos.take_profit == 49250.0
        assert pos.trailing_distance == 0.035
        assert pos.lowest_price == 50000.0
        
        # Test update under TP hit (short: low goes below 49250)
        candle = {
            "open": 50000.0,
            "high": 50100.0,
            "low": 49200.0,
            "close": 49300.0,
            "timestamp": 1781494860000
        }
        res = tracker.update(candle)
        assert res == "take_profit"
        
    def test_short_sl_and_trailing(self, base_config):
        tracker = PositionTracker(base_config)
        tracker.enter(
            side="short",
            entry_price=50000.0,
            quantity=0.1,
            atr=100.0,
            atr_pct=2.0,
            timestamp=1781494800000
        )
        
        # Test ATR stop loss hit (short: high goes above 50300)
        candle_sl = {
            "open": 50000.0,
            "high": 50350.0,
            "low": 49900.0,
            "close": 50280.0,
            "timestamp": 1781494860000
        }
        res = tracker.update(candle_sl)
        assert res == "atr_stop"
        
        # Test trailing stop trigger (short: lowest price 48000, trail stop = 48000 * 1.035 = 49680)
        tracker.exit()
        tracker.enter(
            side="short",
            entry_price=50000.0,
            quantity=0.1,
            atr=100.0,
            atr_pct=2.0,
            timestamp=1781494800000
        )
        tracker.position.take_profit = 0.0  # Prevent take profit hit
    
        candle_low = {
            "open": 50000.0,
            "high": 50000.0,
            "low": 48000.0,
            "close": 48500.0,
            "timestamp": 1781494860000
        }
        res = tracker.update(candle_low)
        assert res == "hold"
        assert tracker.position.lowest_price == 48000.0
        
        # Next candle high hits 49700 (above trail stop 49680)
        candle_trail_hit = {
            "open": 48500.0,
            "high": 49700.0,
            "low": 48400.0,
            "close": 49600.0,
            "timestamp": 1781494920000
        }
        res = tracker.update(candle_trail_hit)
        assert res == "trailing_stop"
        
    def test_timeout_exit(self, base_config):
        tracker = PositionTracker(base_config)
        tracker.enter(
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            atr=100.0,
            atr_pct=2.0,
            timestamp=1781494800000
        )
        
        # Loop for 47 updates
        for i in range(47):
            candle = {
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50000.0,
                "timestamp": 1781494860000 + i * 60000
            }
            res = tracker.update(candle)
            assert res == "hold"
            
        # The 48th update should trigger timeout
        candle_final = {
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50000.0,
            "timestamp": 1781494860000 + 47 * 60000
        }
        res = tracker.update(candle_final)
        assert res == "time_exit"
