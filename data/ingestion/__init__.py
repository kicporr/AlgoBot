"""Data ingestion package.

WebSocket client requires `websocket-client` package.
REST client requires `ccxt` package.
Core components (validator, resampler) have no external dependencies.
"""

# Core components — always available
from .data_validator import DataValidator, ValidationResult
from .resampler import OHLCVResampler, Timeframe, CandleBuilder

# REST client — requires ccxt
try:
    from .rest_client import BitgetRESTClient
    _HAS_REST = True
except ImportError:
    _HAS_REST = False
    BitgetRESTClient = None

# WebSocket client — requires websocket-client
try:
    from .ws_client import BitgetWSClient, WSState
    _HAS_WS = True
except ImportError:
    _HAS_WS = False
    BitgetWSClient = None
    WSState = None


def has_websocket() -> bool:
    return _HAS_WS


def has_rest() -> bool:
    return _HAS_REST


__all__ = [
    "DataValidator", "ValidationResult",
    "OHLCVResampler", "Timeframe", "CandleBuilder",
    "BitgetRESTClient", "BitgetWSClient", "WSState",
    "has_websocket", "has_rest",
]
