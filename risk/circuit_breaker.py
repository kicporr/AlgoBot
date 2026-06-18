"""Circuit breaker — tiered emergency stops for the trading bot.

State machine: NORMAL → WARNING → HALTED

WARNING: Skip next signal, resume automatically after one bar.
HALTED: Stop all trading, requires manual intervention or daily reset.

Triggers (6 total):
    1. Max drawdown (>20% from peak equity)        → HALTED
    2. Daily loss limit (>5% of capital)             → HALTED
    3. Weekly loss limit (>10% of capital)           → HALTED
    4. Consecutive losses (3)                        → WARNING
    5. Consecutive losses (5)                        → HALTED
    6. Volatility spike (ATR > 5× rolling average)   → HALTED
"""

import time
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Tuple


class BreakerState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    HALTED = "halted"


class CircuitBreaker:
    """Multi-level circuit breaker with daily/weekly loss tracking."""

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        cb_cfg = risk_cfg.get("circuit_breaker", {})

        self.max_drawdown_pct = cb_cfg.get("max_drawdown_pct", 20) / 100
        self.daily_limit_pct = cb_cfg.get("daily_loss_limit_pct", 5) / 100
        self.weekly_limit_pct = cb_cfg.get("weekly_loss_limit_pct", 10) / 100
        self.consecutive_halt = cb_cfg.get("consecutive_loss_halt", 5)
        self.consecutive_warn = cb_cfg.get("consecutive_loss_warn", 3)
        self.vol_mult = cb_cfg.get("volatility_circuit_mult", 5.0)

        # Loss reference: "peak" (peak equity) or "initial" (starting capital)
        self.loss_reference = cb_cfg.get("loss_reference", "peak")
        self.initial_capital = risk_cfg.get("initial_capital", 10000.0)

        # Doom loop detection
        doom_cfg = cb_cfg.get("doom_loop", {})
        self.max_daily_trades = doom_cfg.get("max_daily_trades", 20)
        self.max_hourly_trades = doom_cfg.get("max_hourly_trades", 5)
        self.daily_trade_count = 0
        self.hourly_trade_count = 0
        self.last_hour = -1

        # State
        self.state = BreakerState.NORMAL
        self.peak_equity: Optional[float] = None
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_reset_day: int = 0
        self.last_reset_week: int = 0
        self.pause_until_ts: float = 0.0  # Unix timestamp
        self.total_halts: int = 0
        self.last_halt_reason: str = ""
        self.trading_enabled: bool = True
        self.manual_halted: bool = False
        self.halt_reason: str = ""
        self.halt_timestamp: Optional[datetime] = None

    def check(
        self,
        equity: float,
        balance: float,
        recent_trades_pnl: list[float],
        current_atr: float = 0.0,
        avg_atr: float = 0.0,
    ) -> Tuple[BreakerState, Optional[str]]:
        """Run all circuit breaker checks. Returns (state, reason).

        Call once per candle before generating signals.
        """
        now = int(time.time())

        # ── Check Manual Halt ─────────────────────────────────
        if self.manual_halted:
            return BreakerState.HALTED, f"Manual halt active: {self.last_halt_reason} (requires manual reset)"

        # ── Daily / Weekly Reset ──────────────────────────────
        today = now // 86400
        if today != self.last_reset_day:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.last_reset_day = today

        this_week = now // 604800
        if this_week != self.last_reset_week:
            self.weekly_pnl = 0.0
            self.last_reset_week = this_week

        # ── Hourly reset ──────────────────────────────────────
        this_hour = now // 3600
        if this_hour != self.last_hour:
            self.hourly_trade_count = 0
            self.last_hour = this_hour

        # ── Clear warning after one bar ───────────────────────
        if self.state == BreakerState.WARNING:
            self.state = BreakerState.NORMAL

        # ── Check pause-until ─────────────────────────────────
        if self.pause_until_ts > 0 and now < self.pause_until_ts:
            return BreakerState.HALTED, f"Paused until {datetime.fromtimestamp(self.pause_until_ts)}"

        # ── Peak Equity Tracking ──────────────────────────────
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        # ── 1. Max Drawdown ───────────────────────────────────
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd >= self.max_drawdown_pct:
            return self._halt(f"Max drawdown exceeded: {dd:.1%}")

        # ── 2. Daily Loss Limit ───────────────────────────────
        loss_ref = self.peak_equity if self.loss_reference == "peak" and self.peak_equity else self.initial_capital
        if self.daily_pnl < 0 and abs(self.daily_pnl) / loss_ref >= self.daily_limit_pct:
            return self._halt(f"Daily loss limit: ${self.daily_pnl:.0f} ({abs(self.daily_pnl)/loss_ref*100:.1f}% of {self.loss_reference})")

        # ── 3. Weekly Loss Limit ──────────────────────────────
        if self.weekly_pnl < 0 and abs(self.weekly_pnl) / loss_ref >= self.weekly_limit_pct:
            return self._halt(f"Weekly loss limit: ${self.weekly_pnl:.0f} ({abs(self.weekly_pnl)/loss_ref*100:.1f}% of {self.loss_reference})")

        # ── 4. Consecutive Losses ─────────────────────────────
        self._count_consecutive(recent_trades_pnl)
        if self.consecutive_losses >= self.consecutive_halt:
            self.manual_halted = True
            last_pnls = recent_trades_pnl[-self.consecutive_losses:] if len(recent_trades_pnl) >= self.consecutive_losses else recent_trades_pnl
            last_pnls_str = ", ".join([f"${p:.2f}" for p in last_pnls])
            self.halt_reason = f"{self.consecutive_losses} consecutive losses | Last PnLs: [{last_pnls_str}]"
            self.halt_timestamp = datetime.now()
            return self._halt(self.halt_reason)
        if self.consecutive_losses >= self.consecutive_warn:
            self.state = BreakerState.WARNING
            return (BreakerState.WARNING, f"Warning: {self.consecutive_losses} consecutive losses")

        # ── 5. Volatility Spike ───────────────────────────────
        if avg_atr > 0 and current_atr > 0:
            if current_atr > self.vol_mult * avg_atr:
                return self._halt(
                    f"Volatility spike: ATR={current_atr:.1f} vs avg={avg_atr:.1f}"
                )

        # ── 6. Doom Loop: Too Many Trades ─────────────────────
        if self.daily_trade_count >= self.max_daily_trades:
            return self._halt(f"Daily trade limit: {self.daily_trade_count}")
        if self.hourly_trade_count >= self.max_hourly_trades:
            return self._halt(f"Hourly trade limit: {self.hourly_trade_count}")

        self.state = BreakerState.NORMAL
        self.trading_enabled = True
        return (BreakerState.NORMAL, None)

    def record_trade(self, pnl: float, is_closing: bool = False):
        """Record a completed trade's PnL for cumulative tracking."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl

        if pnl < 0 and is_closing:
            pass  # consecutive_losses counted in check() from the list

        if is_closing:
            self.daily_trade_count += 1
            self.hourly_trade_count += 1

    def _count_consecutive(self, pnl_list: list[float]):
        """Count consecutive losing trades from the most recent."""
        self.consecutive_losses = 0
        for pnl in reversed(pnl_list):
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                break

    def _halt(self, reason: str) -> Tuple[BreakerState, str]:
        """Transition to HALTED state."""
        self.state = BreakerState.HALTED
        self.trading_enabled = False
        self.total_halts += 1
        self.last_halt_reason = reason
        return (BreakerState.HALTED, reason)

    def reset_daily(self):
        """Reset daily counters — call at midnight UTC."""
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        if self.state == BreakerState.HALTED and not self.manual_halted:
            self.state = BreakerState.NORMAL
            self.trading_enabled = True
            self.pause_until_ts = 0.0

    def emergency_stop(self, reason: str = "Manual") -> Tuple[BreakerState, str]:
        """Immediate halt — for emergency use only."""
        self.manual_halted = True
        return self._halt(f"EMERGENCY: {reason}")

    def reset_manual_halt(self):
        """Manually resume trading after consecutive losses or manual halt."""
        self.manual_halted = False
        self.state = BreakerState.NORMAL
        self.trading_enabled = True
        self.consecutive_losses = 0
        self.pause_until_ts = 0.0
        self.last_halt_reason = ""

    def get_snapshot(self) -> dict:
        """Return current state for monitoring/Grafana."""
        return {
            "state": self.state.value,
            "trading_enabled": self.trading_enabled,
            "peak_equity": self.peak_equity,
            "daily_pnl": round(self.daily_pnl, 2),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": self.daily_trade_count,
            "total_halts": self.total_halts,
            "last_halt_reason": self.last_halt_reason,
            "manual_halted": self.manual_halted,
        }
