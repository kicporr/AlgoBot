"""Telegram bot for alerts, trade notifications, and daily summaries.

Uses the Telegram Bot API directly via HTTP (no heavy framework) since
the bot only sends notifications — it doesn't receive user commands.

Features:
    - Trade entry/exit alerts with PnL
    - Circuit breaker warnings
    - Hourly risk snapshots
    - Daily performance summaries
    - Error alerts
"""

import json
import threading
from typing import Optional
import requests
from loguru import logger


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlerter:
    """Sends formatted alerts to a Telegram chat via Bot API.

    Usage:
        alerter = TelegramAlerter(config)
        alerter.trade_open("long", 50000, 0.02, "mtf_macd", "trending")
        alerter.daily_summary(stats_dict)
    """

    def __init__(self, config: dict):
        tg_cfg = config.get("telegram", config)
        self.token = config.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = config.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)

        # Throttle alerts to avoid spam
        alert_cfg = tg_cfg.get("alerts", {}) if isinstance(tg_cfg, dict) else {}
        self.min_interval = alert_cfg.get("min_interval_seconds", 5)
        self._last_send = 0.0
        self._lock = threading.Lock()

        # Micro-queue: accumulate dropped alerts, send as summary when interval allows
        self._dropped_count = 0
        self._dropped_types: dict[str, int] = {}  # e.g. {"trade_close": 3, "risk_snapshot": 1}

        if self.enabled:
            logger.info(f"Telegram bot enabled (chat={self.chat_id})")
        else:
            logger.info("Telegram alerts disabled (no token or chat_id)")

    def alert(self, message: str, force: bool = False, category: str = ""):
        """Send a plain text alert with rate limiting.

        Args:
            message: Formatted message to send.
            force: If True, bypass rate limit and send immediately.
            category: Optional label (e.g. 'trade_close', 'risk') for drop tracking.
        """
        if not self.enabled:
            logger.info(f"[TG] {message}")
            return

        import time
        with self._lock:
            now = time.monotonic()
            if not force and (now - self._last_send) < self.min_interval:
                self._dropped_count += 1
                if category:
                    self._dropped_types[category] = self._dropped_types.get(category, 0) + 1
                logger.debug(f"[TG] Rate-limited drop (total dropped: {self._dropped_count}): {message[:80]}...")
                return

            # Flush any pending dropped-count summary before sending new message
            pending_summary = self._flush_dropped_summary()
            self._last_send = now

        if pending_summary:
            self._send(pending_summary)
        self._send(message)

    def _flush_dropped_summary(self) -> str:
        """Build a one-line summary of dropped alerts and reset counters."""
        if self._dropped_count == 0:
            return ""
        summary = f"📦 {self._dropped_count} alert(s) suppressed (rate limit)"
        if self._dropped_types:
            detail = ", ".join(f"{k}:{v}" for k, v in sorted(self._dropped_types.items()))
            summary += f" [{detail}]"
        self._dropped_count = 0
        self._dropped_types.clear()
        return summary

    def trade_open(
        self, side: str, price: float, size: float,
        strategy: str, regime: str, equity: float = 0,
    ):
        """Notify about a new trade entry."""
        emoji = "📈" if side == "long" else "📉"
        msg = (
            f"{emoji} *{side.upper()}* opened\n"
            f"Price: `${price:,.2f}`\n"
            f"Size: `{size:.6f} BTC`\n"
            f"Strategy: `{strategy}`\n"
            f"Regime: `{regime}`\n"
        )
        if equity > 0:
            msg += f"Equity: `${equity:,.2f}`"
        self.alert(msg)

    def trade_close(
        self, side: str, pnl: float, pnl_pct: float,
        reason: str, equity: float = 0,
    ):
        """Notify about a trade exit."""
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} *{side.upper()}* closed\n"
            f"PnL: `${pnl:+,.2f}` ({pnl_pct:+.2f}%)\n"
            f"Reason: `{reason}`\n"
        )
        if equity > 0:
            msg += f"Equity: `${equity:,.2f}`"
        self.alert(msg)

    def risk_snapshot(self, snap: dict):
        """Send risk metrics snapshot."""
        msg = (
            f"📊 *Risk Snapshot*\n"
            f"Equity: `${snap.get('equity', 0):,.2f}`\n"
            f"Return: `{snap.get('total_return_pct', 0)}%`\n"
            f"Drawdown: `{snap.get('current_drawdown_pct', 0)}%`\n"
            f"Trades: `{snap.get('trade_count', 0)}`\n"
            f"Win rate: `{snap.get('win_rate', 0)}%`"
        )
        self.alert(msg)

    def circuit_breaker(self, reason: str):
        """Alert when circuit breaker trips."""
        self.alert(f"⛔ *CIRCUIT BREAKER*: {reason}", force=True)

    def circuit_warning(self, reason: str):
        """Alert on circuit breaker warning."""
        self.alert(f"⚠️ *Warning*: {reason}")

    def daily_summary(self, stats: dict):
        """Send end-of-day performance summary."""
        msg = (
            f"📊 *Daily Summary*\n"
            f"PnL: `${stats.get('daily_pnl', 0):+,.2f}`\n"
            f"Trades: `{stats.get('trade_count', 0)}`\n"
            f"Win rate: `{stats.get('win_rate', 0):.1f}%`\n"
            f"Drawdown: `{stats.get('current_drawdown_pct', 0):.1f}%`\n"
            f"Equity: `${stats.get('equity', 0):,.2f}`"
        )
        self.alert(msg, force=True)

    def error(self, message: str):
        """Send error alert."""
        self.alert(f"❌ *Error*: {message[:300]}", force=True)

    def startup(self, mode: str, capital: float, exchange: str):
        """Notify that the bot has started."""
        self.alert(
            f"🚀 *bocik started*\n"
            f"Mode: `{mode}`\n"
            f"Capital: `${capital:,.2f}`\n"
            f"Exchange: `{exchange}`",
            force=True,
        )

    def shutdown(self, reason: str = "Manual stop"):
        """Notify that the bot has stopped."""
        self.alert(f"⏹️ *bocik stopped*: {reason}", force=True)

    def send_photo(self, photo_bytes: bytes, caption: str = ""):
        """Send a photo with a caption to the Telegram channel using Bot API."""
        if not self.enabled:
            logger.info(f"[TG Photo] (caption: {caption})")
            return

        url = TELEGRAM_API.format(token=self.token, method="sendPhoto")
        files = {"photo": ("chart.png", photo_bytes, "image/png")}
        data = {
            "chat_id": self.chat_id,
            "caption": caption,
            "parse_mode": "Markdown",
        }

        try:
            resp = requests.post(url, data=data, files=files, timeout=15)
            res_json = resp.json()
            if not res_json.get("ok"):
                logger.warning(f"Telegram send photo failed: {res_json.get('description', 'unknown')}")
        except requests.RequestException as e:
            logger.warning(f"Telegram send photo error: {e}")
        except Exception as e:
            logger.error(f"Telegram send photo unexpected error: {e}")

    def _send(self, message: str):
        """Send a message via Telegram Bot API with Markdown formatting."""
        url = TELEGRAM_API.format(token=self.token, method="sendMessage")

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            data = resp.json()
            if not data.get("ok"):
                logger.warning(f"Telegram send failed: {data.get('description', 'unknown')}")
        except requests.RequestException as e:
            logger.warning(f"Telegram send error: {e}")
        except Exception as e:
            logger.error(f"Telegram unexpected error: {e}")
