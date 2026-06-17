"""REST API client for Bitget spot market data and account operations.

Uses ccxt for unified exchange API access.
Bitget requires: API key + secret + PASSPHRASE (3-field auth, not 2 like Binance).

Rate limit: 6000 requests/min/IP, 20 req/s for spot endpoints.
"""

import time
from typing import Optional
from datetime import datetime, timezone
import pandas as pd
import ccxt
from loguru import logger


CCXT_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class BitgetRESTClient:
    """Fetches historical OHLCV data and account info from Bitget via ccxt.

    Usage:
        client = BitgetRESTClient(config)
        candles = client.fetch_ohlcv("BTC/USDT", "1m", since_ms=..., limit=1000)
        df = client.fetch_ohlcv_dataframe("BTC/USDT", "1h", days_back=30)
    """

    def __init__(self, config: dict):
        ex_cfg = config.get("exchange", {})
        self.symbol = ex_cfg.get("symbols", ["BTC/USDT"])[0]

        rate_cfg = ex_cfg.get("rate_limit", {})
        self.max_rps = rate_cfg.get("max_requests_per_second", 10)
        self._last_request_time = 0.0
        self._min_interval = 1.0 / self.max_rps

        # Bitget requires 3-field auth
        api_key = config.get("BITGET_API_KEY", "")
        secret = config.get("BITGET_SECRET_KEY", "")
        passphrase = config.get("BITGET_PASSPHRASE", "")

        ex_type = ex_cfg.get("type", "spot")
        self.exchange = ccxt.bitget({
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,  # ccxt uses 'password' for passphrase
            "enableRateLimit": True,
            "options": {"defaultType": ex_type},
        })

        # Verify connection
        logger.info(
            f"Bitget REST client initialized: {self.exchange.name} "
            f"(authenticated: {bool(api_key)})"
        )

    # ─── Rate Limiting ──────────────────────────────────────────

    def _rate_limit(self):
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    # ─── OHLCV Fetching ────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "1m",
        since_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch raw OHLCV candles from Bitget.

        Returns list of dicts: {timestamp, open, high, low, close, volume}
        """
        symbol = symbol or self.symbol
        self._rate_limit()

        try:
            raw = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=limit
            )
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            raise

        candles = []
        for row in raw:
            candles.append({
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })

        return candles

    def fetch_ohlcv_dataframe(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "1m",
        since_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV as pandas DataFrame."""
        candles = self.fetch_ohlcv(symbol, timeframe, since_ms, limit)
        if not candles:
            return pd.DataFrame(columns=CCXT_OHLCV_COLUMNS)
        df = pd.DataFrame(candles)
        df["timestamp"] = df["timestamp"].astype("int64")
        return df

    # ─── Paginated Fetching ─────────────────────────────────────

    def fetch_ohlcv_range(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "1m",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> list[dict]:
        """Fetch all candles in a time range, paginating as needed."""
        symbol = symbol or self.symbol

        if end_ms is None:
            end_ms = int(time.time() * 1000)

        if start_ms is not None and start_ms > end_ms:
            return []

        all_candles = []
        current_since = start_ms

        while True:
            # Bitget: limit=100 gives proper pagination; 200+ ignores 'since'
            batch = self.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe,
                since_ms=current_since, limit=100,
            )

            if not batch:
                break

            batch = [c for c in batch if c["timestamp"] <= end_ms]
            all_candles.extend(batch)

            # Continue while we got candles and haven't passed end_ms
            last_ts = batch[-1]["timestamp"] if batch else 0
            if len(batch) == 0 or last_ts >= end_ms:
                break

            current_since = batch[-1]["timestamp"] + 1

            if len(all_candles) > 100_000:
                logger.warning("100k candle limit reached — stopping")
                break

        # Deduplicate and sort
        seen = set()
        unique = []
        for c in all_candles:
            if c["timestamp"] not in seen:
                seen.add(c["timestamp"])
                unique.append(c)

        unique.sort(key=lambda x: x["timestamp"])

        if unique:
            logger.info(
                f"Fetched {len(unique)} {timeframe} candles "
                f"({datetime.fromtimestamp(unique[0]['timestamp']/1000, tz=timezone.utc)}"
                f" → {datetime.fromtimestamp(unique[-1]['timestamp']/1000, tz=timezone.utc)})"
            )

        return unique

    # ─── Convenience Methods ────────────────────────────────────

    def fetch_recent(self, timeframe: str = "1m", hours: int = 24) -> list[dict]:
        """Fetch recent N hours of candle data."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (hours * 3_600_000)
        return self.fetch_ohlcv_range(timeframe=timeframe, start_ms=start_ms, end_ms=now_ms)

    def fetch_days(self, timeframe: str = "1h", days: int = 30, symbol: Optional[str] = None) -> pd.DataFrame:
        """Fetch N days as DataFrame."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 86_400_000)
        candles = self.fetch_ohlcv_range(symbol=symbol, timeframe=timeframe, start_ms=start_ms, end_ms=now_ms)
        if not candles:
            return pd.DataFrame(columns=CCXT_OHLCV_COLUMNS)
        df = pd.DataFrame(candles)
        df["timestamp"] = df["timestamp"].astype("int64")
        return df

    def fetch_since(self, timeframe: str = "1h", since_date: str = "2018-12-01") -> pd.DataFrame:
        """Fetch all data since a given date."""
        since_dt = datetime.strptime(since_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(since_dt.timestamp() * 1000)
        now_ms = int(time.time() * 1000)
        candles = self.fetch_ohlcv_range(timeframe=timeframe, start_ms=start_ms, end_ms=now_ms)
        if not candles:
            return pd.DataFrame(columns=CCXT_OHLCV_COLUMNS)
        df = pd.DataFrame(candles)
        df["timestamp"] = df["timestamp"].astype("int64")
        return df

    # ─── Backfill ───────────────────────────────────────────────

    def backfill_gaps(
        self, existing_timestamps: set[int],
        timeframe: str = "1m", lookback_hours: int = 24,
    ) -> list[dict]:
        """Find and fetch missing candles."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (lookback_hours * 3_600_000)

        all_candles = self.fetch_ohlcv_range(
            timeframe=timeframe, start_ms=start_ms, end_ms=now_ms,
        )

        missing = [c for c in all_candles if c["timestamp"] not in existing_timestamps]
        if missing:
            logger.info(f"Backfill: {len(missing)} missing {timeframe} candles")
        return missing

    # ─── Account Info ───────────────────────────────────────────

    def fetch_balance(self) -> dict:
        """Fetch account balance."""
        self._rate_limit()
        try:
            return self.exchange.fetch_balance()
        except ccxt.AuthenticationError:
            logger.warning("Authentication failed — check API key/secret/passphrase")
            return {"free": {}, "used": {}, "total": {}}

    def fetch_ticker(self, symbol: Optional[str] = None) -> dict:
        """Fetch current ticker."""
        symbol = symbol or self.symbol
        self._rate_limit()
        return self.exchange.fetch_ticker(symbol)

    # ─── Health Check ───────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if API is reachable."""
        try:
            self._rate_limit()
            self.exchange.fetch_time()
            return True
        except Exception:
            return False

    def get_server_time(self) -> int:
        """Get server time in milliseconds."""
        self._rate_limit()
        return self.exchange.fetch_time()
