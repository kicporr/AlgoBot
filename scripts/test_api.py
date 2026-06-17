"""Quick Bitget API connection test"""
import sys, os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / "config" / ".env")

api_key = os.getenv("BITGET_API_KEY", "")
secret = os.getenv("BITGET_SECRET_KEY", "")
passphrase = os.getenv("BITGET_PASSPHRASE", "")

print(f"API Key set: {bool(api_key)}")
if api_key:
    print(f"  Prefix: {api_key[:8]}...")
print(f"Secret set:  {bool(secret)}")
print(f"Passphrase:  {bool(passphrase)}")

if not all([api_key, secret, passphrase]):
    print("\nERROR: Missing credentials in config/.env")
    sys.exit(1)

import ccxt
print("\nConnecting to Bitget...")

ex = ccxt.bitget({
    "apiKey": api_key,
    "secret": secret,
    "password": passphrase,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})

# Test 1: Server time
try:
    server_time = ex.fetch_time()
    from datetime import datetime, timezone
    print(f"[OK] Server time: {datetime.fromtimestamp(server_time/1000, tz=timezone.utc)}")
except Exception as e:
    print(f"[FAIL] Server time: {e}")

# Test 2: Markets
try:
    ex.load_markets()
    btc_market = ex.market("BTC/USDT")
    limits = btc_market.get("limits", {}).get("amount", {})
    print(f"[OK] BTC/USDT loaded, min order: {limits.get('min', '?')}")
except Exception as e:
    print(f"[FAIL] Markets: {e}")

# Test 3: Balance
try:
    balance = ex.fetch_balance()
    total = balance.get("total", {})
    btc = total.get("BTC", 0)
    usdt = total.get("USDT", 0)
    print(f"[OK] Balance: {btc} BTC, {usdt} USDT")
except Exception as e:
    print(f"[FAIL] Balance: {e}")

# Test 4: Ticker
try:
    ticker = ex.fetch_ticker("BTC/USDT")
    print(f"[OK] BTC/USDT: ${ticker.get('last', 0):,.2f}")
except Exception as e:
    print(f"[FAIL] Ticker: {e}")

print("\n=== API test complete ===")
