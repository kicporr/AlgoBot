"""Fetch full BTC/USDT history from Bitget (2018-2026)"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import time, pandas as pd
from data.ingestion.rest_client import BitgetRESTClient

config = {
    "exchange": {"name":"bitget","symbols":["BTC/USDT"],
                 "rate_limit":{"max_requests_per_second":5}},
}
client = BitgetRESTClient(config)

# Try fetching from different start years to see what's available
for start_year in [2018, 2019, 2020, 2021, 2022]:
    print(f"\nTrying since {start_year}-01-01...")
    t0 = time.perf_counter()
    candles = client.fetch_ohlcv_range(
        timeframe="1h", start_ms=int(pd.Timestamp(f"{start_year}-01-01").timestamp()*1000),
        end_ms=int(time.time()*1000),
    )
    elapsed = time.perf_counter() - t0
    if candles:
        from datetime import datetime, timezone
        first = datetime.fromtimestamp(candles[0]['timestamp']/1000, tz=timezone.utc)
        last = datetime.fromtimestamp(candles[-1]['timestamp']/1000, tz=timezone.utc)
        print(f"  Got {len(candles):,} candles in {elapsed:.0f}s: {first.date()} -> {last.date()}")
    else:
        print(f"  No data available from {start_year}")
        
    # Only fetch once to test, then stop if we get good data
    if candles and len(candles) > 1000:
        break
