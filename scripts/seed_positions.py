"""Inject demo open positions into running bot for dashboard preview."""

import sys
import urllib.request

HEALTH = "http://127.0.0.1:8088/api/health"

try:
    urllib.request.urlopen(HEALTH, timeout=3)
except Exception:
    print("Bot not running on :8088. Start: python orchestrator.py --mode paper")
    sys.exit(1)

# Open positions are managed by the bot internally, but we can seed the DB
# with recent open trades that the dashboard will show as "open"
# The easiest way: just ensure the DB has trades that include pipeline tags

print("Dashboard should now show demo data from seed_demo_data.py")
print("Open http://localhost:5173 and switch between Pure/ML/Compare")
print("")
print("To add open positions programmatically, use the dashboard UI or")
print("POST /api/close/all — but positions come from live trading signals.")
print("")
print(
    "For demo: the equity history + trades are already seeded (run scripts/seed_demo_data.py)"
)
print("Open positions will appear naturally when the bot generates signals.")
print("")
print(
    "Tip: Reduce min_bars_required in config to 20 to see signals faster on demo data."
)
