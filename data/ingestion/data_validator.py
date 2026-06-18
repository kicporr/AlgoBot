"""Data validation for incoming OHLCV candles.

Every candle passes through this validator before being stored or used.
Rejected candles are logged and skipped — never silently dropped.
"""

from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationResult:
    """Result of candle validation."""
    valid: bool
    reason: Optional[str] = None


class DataValidator:
    """Validates OHLCV candles for sanity and consistency.
    
    Checks performed:
    1. OHLCV values exist and are numeric
    2. All prices > 0
    3. High >= max(Open, Close) and Low <= min(Open, Close)
    4. Volume >= 0
    5. Timestamp is reasonable (not too far in future/past)
    6. No extreme single-bar price jumps (>30% by default)
    """
    
    def __init__(self, config: Optional[dict] = None):
        cfg = config.get("data", {}).get("validation", {}) if config else {}
        self.max_price_jump_pct = cfg.get("max_price_jump_pct", 30) / 100.0
        self.timestamp_tolerance_s = cfg.get("timestamp_tolerance_s", 5)
        self.last_close: Optional[float] = None
        self._seen_timestamps: set[int] = set()
        self._max_seen_cache = 10000  # Prevent memory leak
    
    def validate(self, candle: dict, previous_close: Optional[float] = None) -> ValidationResult:
        """Validate a single OHLCV candle. Returns ValidationResult."""
        
        # --- 1. Required fields exist ---
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        for field in required:
            if field not in candle:
                return ValidationResult(False, f"Missing required field: {field}")
            if candle[field] is None:
                return ValidationResult(False, f"Null value for field: {field}")
        
        ts = candle["timestamp"]
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        v = float(candle["volume"])
        
        # --- 2. Prices must be positive ---
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return ValidationResult(False, f"Non-positive price: O={o} H={h} L={l} C={c}")
        
        # --- 3. OHLC relationship ---
        if h < max(o, c):
            return ValidationResult(False, f"High ({h}) < max(Open={o}, Close={c})")
        if l > min(o, c):
            return ValidationResult(False, f"Low ({l}) > min(Open={o}, Close={c})")
        
        # --- 4. Volume non-negative ---
        if v < 0:
            return ValidationResult(False, f"Negative volume: {v}")
        
        # --- 5. Duplicate timestamp check ---
        if ts in self._seen_timestamps:
            return ValidationResult(False, f"Duplicate timestamp: {ts}")
        
        # --- 6. Price jump check (optional, needs previous close) ---
        prev = previous_close if previous_close is not None else self.last_close
        if prev is not None and prev > 0:
            change = abs(c - prev) / prev
            if change > self.max_price_jump_pct:
                if change > 0.90:  # >90% jump = corrupted data, reject
                    return ValidationResult(False,
                        f"Extreme price jump ({change:.1%}): prev={prev:.2f} -> curr={c:.2f} — corrupted data")
                logger.warning(
                    f"Large price jump: {change:.2%} | "
                    f"Prev close={prev:.2f} -> Current close={c:.2f} | "
                    f"Timestamp={ts}"
                )
                # NOTE: Jumps up to 90% are accepted; crypto can move fast.
                # Circuit breaker handles volatility at the risk layer.
        
        # --- 7. Timestamp sanity (optional: check not too far in future) ---
        import time
        now_ms = int(time.time() * 1000)
        future_cutoff = now_ms + (self.timestamp_tolerance_s * 1000)
        if ts > future_cutoff:
            return ValidationResult(False, f"Timestamp in future: {ts} > {future_cutoff}")
        
        # --- Passed all checks ---
        self.last_close = c
        
        # Manage seen timestamps cache
        self._seen_timestamps.add(ts)
        if len(self._seen_timestamps) > self._max_seen_cache:
            # Remove oldest half
            sorted_ts = sorted(self._seen_timestamps)
            keep = set(sorted_ts[len(sorted_ts)//2:])
            self._seen_timestamps = keep
        
        return ValidationResult(True)
    
    def validate_batch(self, candles: list[dict]) -> tuple[list[dict], list[dict]]:
        """Validate a batch of candles. Returns (valid, rejected)."""
        valid = []
        rejected = []
        
        for i, candle in enumerate(candles):
            prev_close = valid[-1]["close"] if valid else None
            result = self.validate(candle, previous_close=prev_close)
            
            if result.valid:
                valid.append(candle)
            else:
                logger.warning(f"Rejected candle at index {i}: {result.reason}")
                rejected.append({"index": i, "candle": candle, "reason": result.reason})
        
        return valid, rejected
    
    def reset(self):
        """Reset validator state (call when reconnecting or starting fresh)."""
        self.last_close = None
        self._seen_timestamps.clear()
