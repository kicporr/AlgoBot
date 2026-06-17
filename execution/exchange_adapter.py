"""ccxt-based exchange adapter for Bitget spot trading.

Bitget requires three-field auth: API key + secret + passphrase.
Uses ccxt.bitget() which handles HMAC-SHA256 signing automatically.
"""

import ccxt
from typing import Optional
from loguru import logger


class ExchangeAdapter:
    """Wrapper around ccxt for Bitget spot API calls.

    Usage:
        adapter = ExchangeAdapter(config)
        adapter.fetch_balance()
        adapter.create_limit_buy_order("BTC/USDT", 0.001, 50000)
    """

    def __init__(self, config: dict):
        ex_cfg = config.get("exchange", {})
        self.name = ex_cfg.get("name", "bitget")
        self.symbol = ex_cfg.get("symbols", ["BTC/USDT"])[0]

        # Bitget 3-field auth
        api_key = config.get("BITGET_API_KEY", "")
        secret = config.get("BITGET_SECRET_KEY", "")
        passphrase = config.get("BITGET_PASSPHRASE", "")

        ex_type = ex_cfg.get("type", "spot")
        self.exchange = ccxt.bitget({
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,  # ccxt uses 'password' for the passphrase
            "enableRateLimit": True,
            "options": {"defaultType": ex_type},
        })

        # Load markets to validate symbol
        try:
            self.exchange.load_markets()
            logger.info(f"Bitget exchange adapter initialized (authenticated: {bool(api_key)})")
        except ccxt.AuthenticationError:
            logger.error(
                "Bitget auth failed — check BITGET_API_KEY, BITGET_SECRET_KEY, and BITGET_PASSPHRASE"
            )
            raise
        except Exception as e:
            logger.warning(f"Bitget market load warning (continuing): {e}")

    def fetch_ohlcv(
        self, symbol: Optional[str] = None, timeframe: str = "1h",
        since: Optional[int] = None, limit: int = 500,
    ):
        """Fetch historical OHLCV candles from Bitget.

        Returns list of [timestamp, open, high, low, close, volume].
        """
        symbol = symbol or self.symbol
        return self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    def fetch_order_book(self, symbol: Optional[str] = None, limit: int = 20):
        """Fetch current order book."""
        symbol = symbol or self.symbol
        return self.exchange.fetch_order_book(symbol, limit)

    def fetch_balance(self) -> dict:
        """Fetch account balance. Returns ccxt unified balance dict."""
        try:
            return self.exchange.fetch_balance()
        except ccxt.AuthenticationError:
            logger.error("Auth failed — cannot fetch balance")
            return {"free": {}, "used": {}, "total": {}}

    def fetch_ticker(self, symbol: Optional[str] = None) -> dict:
        """Fetch current ticker."""
        symbol = symbol or self.symbol
        return self.exchange.fetch_ticker(symbol)

    def create_limit_buy_order(
        self, symbol: Optional[str] = None, amount: float = 0, price: float = 0,
    ):
        """Place a limit buy order."""
        symbol = symbol or self.symbol
        return self.exchange.create_limit_buy_order(symbol, amount, price)

    def create_limit_sell_order(
        self, symbol: Optional[str] = None, amount: float = 0, price: float = 0,
    ):
        """Place a limit sell order."""
        symbol = symbol or self.symbol
        return self.exchange.create_limit_sell_order(symbol, amount, price)

    def create_market_buy_order(
        self, symbol: Optional[str] = None, amount: float = 0,
    ):
        """Place a market buy order."""
        symbol = symbol or self.symbol
        return self.exchange.create_market_buy_order(symbol, amount)

    def create_market_sell_order(
        self, symbol: Optional[str] = None, amount: float = 0,
    ):
        """Place a market sell order."""
        symbol = symbol or self.symbol
        return self.exchange.create_market_sell_order(symbol, amount)

    def cancel_order(self, order_id: str, symbol: Optional[str] = None):
        """Cancel an open order."""
        symbol = symbol or self.symbol
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: Optional[str] = None):
        """Fetch all open orders."""
        symbol = symbol or self.symbol
        return self.exchange.fetch_open_orders(symbol)

    def fetch_closed_orders(self, symbol: Optional[str] = None, limit: int = 50):
        """Fetch recently closed orders."""
        symbol = symbol or self.symbol
        return self.exchange.fetch_closed_orders(symbol, limit=limit)

    def fetch_order(self, order_id: str, symbol: Optional[str] = None):
        """Fetch details of a single order by id."""
        symbol = symbol or self.symbol
        return self.exchange.fetch_order(order_id, symbol)

    def is_connected(self) -> bool:
        """Check exchange connectivity."""
        try:
            self.exchange.fetch_time()
            return True
        except Exception:
            return False

    def get_server_time(self) -> int:
        """Get exchange server time in ms."""
        return self.exchange.fetch_time()
