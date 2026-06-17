"""Data layer — ingestion, storage, validation."""

from .ingestion import (
    DataValidator, ValidationResult, OHLCVResampler, Timeframe, CandleBuilder,
    has_websocket, has_rest,
)

try:
    from .ingestion import BitgetRESTClient
except ImportError:
    BitgetRESTClient = None

try:
    from .ingestion import BitgetWSClient, WSState
except ImportError:
    BitgetWSClient = None
    WSState = None

from .storage import DatabaseManager, CandleRepository, TradeRepository, SignalRepository

__all__ = [
    "DataValidator", "ValidationResult", "OHLCVResampler", "Timeframe", "CandleBuilder",
    "BitgetRESTClient", "BitgetWSClient", "WSState",
    "DatabaseManager", "CandleRepository", "TradeRepository", "SignalRepository",
    "has_websocket", "has_rest",
]
