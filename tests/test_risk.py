"""Tests for Phase 4: Risk Layer Integration."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import numpy as np
import pandas as pd


# ─── KellyPositionSizer ───────────────────────────────────────


class TestKellyPositionSizer:

    def _make_sizer(self, overrides=None):
        from risk.position_sizer import KellyPositionSizer
        config = {
            "risk": {
                "position_sizing": {
                    "kelly": {
                        "default_win_rate": 0.45,
                        "default_avg_win_pct": 2.0,
                        "default_avg_loss_pct": 1.5,
                        "fraction": 0.5,
                        "max_kelly_pct": 0.25,
                    },
                    "max_risk_per_trade_pct": 2.0,
                    "max_position_size_btc": 0.1,
                    "max_total_exposure_pct": 50,
                    "min_position_size_btc": 0.0001,
                    "volatility": {"enabled": False, "window": 50},
                }
            }
        }
        if overrides:
            config["risk"]["position_sizing"].update(overrides)
        return KellyPositionSizer(config)

    def test_calculates_positive_size(self):
        sizer = self._make_sizer()
        size = sizer.calculate(capital=10000, btc_price=50000)
        assert size > 0
        assert size <= 0.1  # max_position_btc

    def test_zero_when_no_capital(self):
        sizer = self._make_sizer()
        assert sizer.calculate(capital=0, btc_price=50000) == 0

    def test_zero_when_no_price(self):
        sizer = self._make_sizer()
        assert sizer.calculate(capital=10000, btc_price=0) == 0

    def test_respects_max_exposure(self):
        sizer = self._make_sizer()
        size = sizer.calculate(capital=10000, btc_price=50000)
        max_exposure_btc = (10000 * 0.50) / 50000  # 50% of 10k / 50k = 0.1
        assert size <= max_exposure_btc + 1e-8

    def test_volatility_adjustment_reduces_size(self):
        from risk.position_sizer import KellyPositionSizer
        config = {
            "risk": {
                "position_sizing": {
                    "kelly": {"default_win_rate": 0.5, "default_avg_win_pct": 3.0,
                              "default_avg_loss_pct": 2.0, "fraction": 0.5, "max_kelly_pct": 0.25},
                    "max_risk_per_trade_pct": 5.0,
                    "max_position_size_btc": 0.5,
                    "max_total_exposure_pct": 50,
                    "min_position_size_btc": 0.0001,
                    "volatility": {"enabled": True, "window": 50},
                }
            }
        }
        sizer = KellyPositionSizer(config)

        size_normal = sizer.calculate(
            capital=10000, btc_price=50000, current_atr=100, avg_atr=100
        )
        size_elevated = sizer.calculate(
            capital=10000, btc_price=50000, current_atr=300, avg_atr=100
        )
        assert size_elevated < size_normal, "High vol should reduce position size"

    def test_minimum_position_filter(self):
        sizer = self._make_sizer({"min_position_size_btc": 0.01})
        size = sizer.calculate(capital=10, btc_price=50000)
        assert size == 0  # Too small

    def test_returns_metrics(self):
        sizer = self._make_sizer()
        sizer.calculate(capital=10000, btc_price=50000)
        metrics = sizer.get_last_metrics()
        assert "kelly_pct" in metrics
        assert "position_btc" in metrics

    def test_fixed_fraction_streak_sizing(self):
        from risk.position_sizer import KellyPositionSizer
        config = {
            "risk": {
                "max_position_pct": 0.20,
                "position_sizing": {
                    "method": "fixed_fraction",
                    "max_risk_per_trade_pct": 2.0,
                    "max_position_size_btc": 0.5,
                    "max_total_exposure_pct": 50,
                    "min_position_size_btc": 0.0001,
                    "volatility": {"enabled": False, "window": 50},
                }
            }
        }
        sizer = KellyPositionSizer(config)

        # Baseline: no losses, no wins
        size_base = sizer.calculate(capital=10000, btc_price=50000, consecutive_losses=0, consecutive_wins=0)
        # Expected: 20% of 10000 = 2000 USDT -> 2000 / 50000 = 0.04 BTC
        assert abs(size_base - 0.04) < 1e-6

        # After 2 losses: should be halved (10%)
        size_losses = sizer.calculate(capital=10000, btc_price=50000, consecutive_losses=2, consecutive_wins=0)
        # Expected: 10% of 10000 = 1000 USDT -> 1000 / 50000 = 0.02 BTC
        assert abs(size_losses - 0.02) < 1e-6

        # After 3 wins: should be scaled 1.5x (30%)
        size_wins = sizer.calculate(capital=10000, btc_price=50000, consecutive_losses=0, consecutive_wins=3)
        # Expected: 30% of 10000 = 3000 USDT -> 3000 / 50000 = 0.06 BTC
        assert abs(size_wins - 0.06) < 1e-6



# ─── CircuitBreaker ───────────────────────────────────────────


class TestCircuitBreaker:

    def _make_breaker(self, overrides=None):
        from risk.circuit_breaker import CircuitBreaker
        config = {
            "risk": {
                "circuit_breaker": {
                    "max_drawdown_pct": 20,
                    "daily_loss_limit_pct": 5,
                    "weekly_loss_limit_pct": 10,
                    "consecutive_loss_halt": 5,
                    "consecutive_loss_warn": 3,
                    "volatility_circuit_mult": 5.0,
                    "doom_loop": {"max_daily_trades": 20, "max_hourly_trades": 5},
                }
            }
        }
        if overrides:
            config["risk"]["circuit_breaker"].update(overrides)
        return CircuitBreaker(config)

    def test_normal_state(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        state, reason = cb.check(equity=10000, balance=10000, recent_trades_pnl=[])
        assert state == BreakerState.NORMAL
        assert reason is None

    def test_drawdown_halt(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.peak_equity = 10000
        cb.daily_pnl = 0
        cb.last_reset_day = int(time.time()) // 86400
        state, reason = cb.check(
            equity=7500, balance=10000, recent_trades_pnl=[], current_atr=100, avg_atr=200
        )
        assert state == BreakerState.HALTED
        assert "drawdown" in reason.lower()

    def test_consecutive_loss_warn(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.peak_equity = 10000
        cb.last_reset_day = int(time.time()) // 86400
        state, reason = cb.check(
            equity=10000, balance=10000, recent_trades_pnl=[-100, -50, -75],
            current_atr=100, avg_atr=200,
        )
        assert state == BreakerState.WARNING

    def test_consecutive_loss_halt(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.peak_equity = 10000
        cb.last_reset_day = int(time.time()) // 86400
        state, _ = cb.check(
            equity=10000, balance=10000,
            recent_trades_pnl=[-100, -50, -75, -200, -150],
        )
        assert state == BreakerState.HALTED

    def test_consecutive_loss_requires_manual_reset(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.peak_equity = 10000
        cb.last_reset_day = int(time.time()) // 86400
        
        # Trigger consecutive losses halt
        state, _ = cb.check(
            equity=10000, balance=10000,
            recent_trades_pnl=[-100, -50, -75, -200, -150],
        )
        assert state == BreakerState.HALTED
        assert cb.manual_halted is True
        
        # Test that reset_daily doesn't clear the halt state
        cb.reset_daily()
        assert cb.manual_halted is True
        assert cb.state == BreakerState.HALTED
        
        # Check that manual reset clears it
        cb.reset_manual_halt()
        assert cb.manual_halted is False
        assert cb.state == BreakerState.NORMAL

    def test_emergency_stop_requires_manual_reset(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.emergency_stop("Test panic")
        assert cb.manual_halted is True
        
        # Test that reset_daily doesn't clear it
        cb.reset_daily()
        assert cb.manual_halted is True
        assert cb.state == BreakerState.HALTED
        
        # Test that manual reset clears it
        cb.reset_manual_halt()
        assert cb.manual_halted is False
        assert cb.state == BreakerState.NORMAL
    def test_daily_loss_halt(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.peak_equity = 10000
        cb.daily_pnl = -600  # 6% of 10k > 5% limit
        cb.last_reset_day = int(time.time()) // 86400  # Prevent daily reset
        state, _ = cb.check(equity=10000, balance=10000, recent_trades_pnl=[])
        assert state == BreakerState.HALTED

    def test_volatility_spike_halt(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.peak_equity = 10000
        cb.daily_pnl = 0
        cb.weekly_pnl = 0
        state, reason = cb.check(
            equity=10000, balance=10000, recent_trades_pnl=[],
            current_atr=600, avg_atr=100,
        )
        assert state == BreakerState.HALTED
        assert "volatility" in reason.lower()

    def test_emergency_stop(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        state, reason = cb.emergency_stop("Test panic")
        assert state == BreakerState.HALTED
        assert "EMERGENCY" in reason

    def test_daily_reset(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.daily_pnl = -600
        cb.state = BreakerState.HALTED
        cb.reset_daily()
        assert cb.daily_pnl == 0.0

    def test_warning_clears_after_one_bar(self):
        from risk.circuit_breaker import BreakerState
        cb = self._make_breaker()
        cb.peak_equity = 10000
        # First bar: warning
        state1, _ = cb.check(equity=10000, balance=10000, recent_trades_pnl=[-100, -50, -75])
        assert state1 == BreakerState.WARNING
        # Second bar: should clear (no new consecutive losses)
        state2, _ = cb.check(equity=10000, balance=10000, recent_trades_pnl=[])
        assert state2 == BreakerState.NORMAL

    def test_record_trade_tracks_pnl(self):
        cb = self._make_breaker()
        cb.record_trade(50.0, is_closing=True)
        assert cb.daily_pnl == 50.0
        assert cb.daily_trade_count == 1

    def test_snapshot(self):
        cb = self._make_breaker()
        snap = cb.get_snapshot()
        assert "state" in snap
        assert "daily_pnl" in snap
        assert "trading_enabled" in snap


# ─── RiskMonitor ──────────────────────────────────────────────


class TestRiskMonitor:

    def _make_monitor(self):
        from risk.risk_monitor import RiskMonitor
        config = {
            "risk": {
                "initial_capital": 10000,
                "monitoring": {
                    "snapshot_interval_seconds": 3600,
                    "alerts": {"drawdown_warn_pct": 10, "consecutive_loss_warn": 3},
                },
            }
        }
        return RiskMonitor(config)

    def test_initial_capital(self):
        rm = self._make_monitor()
        assert rm.equity == 10000

    def test_update_tracks_equity(self):
        rm = self._make_monitor()
        rm.update(equity=10500, balance=9800, btc_price=50000, position_size=0.05, position_entry=49500)
        assert rm.equity == 10500
        assert rm.position_size_btc == 0.05

    def test_drawdown_calculation(self):
        rm = self._make_monitor()
        rm.update(equity=11000, balance=11000)  # New peak
        rm.update(equity=9900, balance=9900)    # -10%
        assert rm.current_drawdown_pct > 0.09
        assert rm.max_drawdown_pct >= rm.current_drawdown_pct

    def test_record_trade(self):
        rm = self._make_monitor()
        rm.record_trade(200.0, is_win=True)
        assert rm.trade_count == 1
        assert rm.win_count == 1
        assert rm.total_pnl == 200.0
        assert rm.consecutive_losses == 0

    def test_record_trade_updates_consecutive_losses(self):
        rm = self._make_monitor()
        rm.record_trade(-100, is_win=False)
        rm.record_trade(-50, is_win=False)
        assert rm.consecutive_losses == 2

    def test_snapshot(self):
        rm = self._make_monitor()
        rm.update(equity=10500, balance=10500, btc_price=50000)
        snap = rm.snapshot()
        assert "equity" in snap
        assert "total_return_pct" in snap
        assert "current_drawdown_pct" in snap
        assert "exposure_pct" in snap
        assert snap["total_return_pct"] > 0

    def test_snapshot_with_position(self):
        rm = self._make_monitor()
        rm.update(equity=10000, balance=5000, btc_price=50000, position_size=0.1, position_entry=49000)
        snap = rm.snapshot()
        assert snap["exposure_pct"] > 0
        assert snap["unrealized_pnl"] == pytest.approx(0.1 * (50000 - 49000), rel=0.01)

    def test_daily_weekly_reset(self):
        rm = self._make_monitor()
        rm.record_trade(500, is_win=True)
        rm._last_day = int(time.time()) // 86400 - 1  # Force reset
        rm.update(equity=10500, balance=10500)
        assert rm.daily_pnl == 0.0  # Reset

    def test_metrics_dashboard(self):
        rm = self._make_monitor()
        rm.update(equity=10500, balance=10500)
        d = rm.get_metrics_dashboard()
        assert "equity" in d
        assert "win_rate" in d
