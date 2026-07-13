#!/usr/bin/env python
"""Watchdog — monitors bocik health and alerts via Telegram on downtime.

Usage:
    python scripts/watchdog.py                          # one-shot check
    python scripts/watchdog.py --loop 60                # check every 60 seconds
    python scripts/watchdog.py --url http://1.2.3.4:8088  # remote bot

Configuration (environment variables or config/.env):
    TELEGRAM_BOT_TOKEN    — Telegram bot token
    TELEGRAM_CHAT_ID      — Telegram chat ID for alerts
    HEALTH_URL            — URL to check (default: http://127.0.0.1:8088/api/health)
    CHECK_INTERVAL        — seconds between checks in loop mode (default: 60)

Exit codes:
    0 — healthy
    1 — unhealthy (bot down or error)
"""

import os, sys, time, json
import urllib.request
import urllib.error
from pathlib import Path

# Load .env if present
ENV_PATH = Path(__file__).parent.parent / "config" / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                if key.strip() not in os.environ:
                    os.environ[key.strip()] = val.strip()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
HEALTH_URL = os.environ.get("HEALTH_URL", "http://127.0.0.1:8088/api/health")

# State file to track transitions (prevents repeated "down" alerts)
STATE_FILE = Path(__file__).parent.parent / "data" / ".watchdog_state"


def send_telegram(message: str):
    """Send a message via Telegram Bot API. Non-blocking, best-effort."""
    if not TOKEN or not CHAT_ID:
        print(f"[watchdog] Telegram not configured. Message: {message}")
        return
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = json.dumps({
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[watchdog] Telegram send failed: {e}")


def read_state() -> str:
    """Read last known state: 'up', 'down', or None."""
    try:
        return STATE_FILE.read_text().strip()
    except FileNotFoundError:
        return None


def write_state(state: str):
    """Write current state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(state)


def check_health() -> bool:
    """Ping the health endpoint. Returns True if healthy."""
    try:
        req = urllib.request.Request(HEALTH_URL)
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        return data.get("status") == "ok"
    except Exception as e:
        print(f"[watchdog] Health check failed: {e}")
        return False


def run_once():
    """Single health check with alerting on state transitions."""
    healthy = check_health()
    prev = read_state()
    current = "up" if healthy else "down"

    if prev is None:
        write_state(current)
        if not healthy:
            send_telegram("🚨 *bocik — PIERWSZY ALERT*\nBot nie odpowiada na health check!\nSprawdź `orchestrator.py` lub połączenie.")
        return healthy

    if prev == "up" and current == "down":
        write_state(current)
        send_telegram("🔴 *bocik — NIEDOSTĘPNY*\nBot przestał odpowiadać na health check.\nSprawdź logi i połączenie z serwerem.")
        return False

    if prev == "down" and current == "up":
        write_state(current)
        send_telegram("🟢 *bocik — ODZYSKANY*\nBot znowu odpowiada. Połączenie przywrócone.")
        return True

    # No change — silent
    write_state(current)
    return healthy


def run_loop(interval: int = 60):
    """Continuous monitoring loop."""
    print(f"[watchdog] Starting loop mode. Checking {HEALTH_URL} every {interval}s")
    while True:
        healthy = run_once()
        status = "UP" if healthy else "DOWN"
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {status}")
        time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="bocik health watchdog")
    parser.add_argument("--loop", type=int, default=0, help="Run continuously, check every N seconds")
    parser.add_argument("--url", type=str, default=None, help="Health URL override")
    args = parser.parse_args()

    if args.url:
        HEALTH_URL = args.url

    if args.loop > 0:
        run_loop(args.loop)
    else:
        ok = run_once()
        sys.exit(0 if ok else 1)
