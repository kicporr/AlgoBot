"""Real-time risk monitoring — equity tracking, drawdown calculation, exposure.

Generates periodic snapshots for Grafana dashboards and Telegram alerts.
"""

import time
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from loguru import logger


class RiskMonitor:
    """Tracks portfolio risk metrics in real-time.

    Usage:
        monitor = RiskMonitor(config)
        monitor.set_initial_capital(10000)
        # Each candle:
        monitor.update(equity=10100, balance=9900, btc_price=50000, position_size=0.02)
        snapshot = monitor.snapshot()
    """

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        mon_cfg = risk_cfg.get("monitoring", {})

        # Capital
        self.initial_capital = risk_cfg.get("initial_capital", 10000.0)

        # Current state
        self.equity = self.initial_capital
        self.balance = self.initial_capital
        self.peak_equity = self.initial_capital
        self.position_size_btc = 0.0
        self.position_entry_price = 0.0
        self.btc_price = 0.0

        # PnL tracking
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.unrealized_pnl = 0.0

        # Drawdown
        self.current_drawdown_pct = 0.0
        self.max_drawdown_pct = 0.0

        # Trade history (in-memory, for quick queries)
        self.trade_count = 0
        self.win_count = 0
        self.consecutive_losses = 0
        self.last_trade_pnl = 0.0

        # Rolling metrics
        self._recent_pnls: list[float] = []  # Last 50 trades

        # Periodic reset
        self._last_day = int(time.time()) // 86400
        self._last_week = int(time.time()) // 604800
        self._snapshot_interval = mon_cfg.get("snapshot_interval_seconds", 3600)
        self._last_snapshot_ts = 0.0

        # Alert thresholds
        alert_cfg = mon_cfg.get("alerts", {})
        self.alert_drawdown_pct = alert_cfg.get("drawdown_warn_pct", 10) / 100
        self.alert_consecutive = alert_cfg.get("consecutive_loss_warn", 3)

    def set_initial_capital(self, capital: float):
        """Set initial capital (call once at startup)."""
        self.initial_capital = capital
        self.equity = capital
        self.balance = capital
        self.peak_equity = capital

    def update(
        self,
        equity: float,
        balance: float,
        btc_price: float = 0.0,
        position_size: float = 0.0,
        position_entry: float = 0.0,
    ):
        """Update all metrics from current portfolio state.

        Call once per candle with latest values from the exchange.
        """
        self.equity = equity
        self.balance = balance
        self.btc_price = btc_price
        self.position_size_btc = position_size
        self.position_entry_price = position_entry

        # Track peak equity
        self.peak_equity = max(self.peak_equity, equity)

        # Drawdown
        self.current_drawdown_pct = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.max_drawdown_pct = max(self.max_drawdown_pct, self.current_drawdown_pct)

        # Unrealized PnL
        if position_size > 0 and btc_price > 0 and position_entry > 0:
            self.unrealized_pnl = position_size * (btc_price - position_entry)
        else:
            self.unrealized_pnl = 0.0

        # Daily/Weekly reset
        now_ts = int(time.time())
        today = now_ts // 86400
        this_week = now_ts // 604800

        if today != self._last_day:
            self.daily_pnl = 0.0
            self._last_day = today

        if this_week != self._last_week:
            self.weekly_pnl = 0.0
            self._last_week = this_week

    def record_trade(self, pnl: float, is_win: bool):
        """Record a completed trade."""
        self.trade_count += 1
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.last_trade_pnl = pnl

        if is_win:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        self._recent_pnls.append(pnl)
        if len(self._recent_pnls) > 50:
            self._recent_pnls = self._recent_pnls[-50:]

        # Alert on consecutive losses
        if self.consecutive_losses >= self.alert_consecutive:
            logger.warning(f"⚠️ {self.consecutive_losses} consecutive losses")

        # Alert on drawdown
        if self.current_drawdown_pct >= self.alert_drawdown_pct:
            logger.warning(f"⚠️ Drawdown: {self.current_drawdown_pct:.1%}")

    def snapshot(self) -> dict:
        """Return comprehensive risk snapshot for logging/Grafana."""
        now = int(time.time())
        self._last_snapshot_ts = now

        return {
            "timestamp": now,
            "datetime": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            # Capital
            "initial_capital": round(self.initial_capital, 2),
            "equity": round(self.equity, 2),
            "balance": round(self.balance, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            # Returns
            "total_return_pct": round(
                ((self.equity - self.initial_capital) / self.initial_capital) * 100, 2
            ),
            "daily_pnl": round(self.daily_pnl, 2),
            "weekly_pnl": round(self.weekly_pnl, 2),
            # Drawdown
            "current_drawdown_pct": round(self.current_drawdown_pct * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "peak_equity": round(self.peak_equity, 2),
            # Position
            "position_size_btc": self.position_size_btc,
            "position_value_usd": round(self.position_size_btc * self.btc_price, 2),
            "exposure_pct": round(
                (self.position_size_btc * self.btc_price / self.equity * 100)
                if self.equity > 0 else 0, 2
            ),
            # Trades
            "trade_count": self.trade_count,
            "win_rate": round(self.win_count / self.trade_count * 100, 2) if self.trade_count > 0 else 0.0,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_pnl": round(self.last_trade_pnl, 2),
            # Rolling
            "recent_pnl_total": round(sum(self._recent_pnls), 2),
            "recent_trade_count": len(self._recent_pnls),
        }

    def get_alert_messages(self) -> list[str]:
        """Generate alert messages for Telegram if thresholds are breached."""
        alerts = []

        if self.current_drawdown_pct >= self.alert_drawdown_pct:
            alerts.append(f"⚠️ Drawdown: {self.current_drawdown_pct:.1%}")

        if self.consecutive_losses >= self.alert_consecutive:
            alerts.append(f"⚠️ {self.consecutive_losses} consecutive losses")

        return alerts

    def get_metrics_dashboard(self) -> dict:
        """Minimal metrics for quick status checks."""
        return {
            "equity": round(self.equity, 2),
            "dd_pct": round(self.current_drawdown_pct * 100, 2),
            "trades": self.trade_count,
            "win_rate": round(self.win_count / self.trade_count * 100, 1) if self.trade_count > 0 else 0.0,
            "pnl_daily": round(self.daily_pnl, 2),
        }
