"""Fetch full BTC history — 2020 to 2026 (optimized)"""
from pathlib import Path
import sys, time
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.ingestion.rest_client import BitgetRESTClient

config = {"exchange":{"name":"bitget","symbols":["BTC/USDT"],"rate_limit":{"max_requests_per_second":8}}}
client = BitgetRESTClient(config)

from datetime import datetime, timezone
print(f"Fetching BTC/USDT 1H from 2020-01-01 to now...")
t0 = time.perf_counter()

# Extended patience: 700 batches × 0.15s rate = ~105s + overhead
candles = client.fetch_since(timeframe="1h", since_date="2020-01-01")

elapsed = time.perf_counter() - t0
if candles is not None and not candles.empty:
    n = len(candles)
    sd = datetime.fromtimestamp(candles['timestamp'].iloc[0]/1000, tz=timezone.utc)
    ed = datetime.fromtimestamp(candles['timestamp'].iloc[-1]/1000, tz=timezone.utc)
    print(f"Done: {n:,} bars in {elapsed:.0f}s")
    print(f"Range: {sd.date()} -> {ed.date()}")
    print(f"Years: {(ed-sd).days/365:.1f}")
    
    candles.to_parquet(PROJECT_ROOT / "data" / "cache" / "btc_1h_2020_2026.parquet")
    print(f"Saved to cache")
else:
    print(f"FAILED: empty dataframe")
