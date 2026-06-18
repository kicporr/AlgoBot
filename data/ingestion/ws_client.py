"""Bitget WebSocket client for real-time spot kline streams.

Protocol differences from Binance:
    - Subscribe via JSON: {"op":"subscribe","args":[{"instType":"SPOT","channel":"candle1m","instId":"BTCUSDT"}]}
    - Data is flat arrays [ts, o, h, l, c, vol_base, vol_quote] — NOT nested dicts
    - No 'x' (close) flag — candle completion detected by timestamp change
    - Pushes 'snapshot' first, then 'update' events
    - Ping/pong via WebSocket ping frames (not JSON)

Architecture:
    Bitget WS → on_message → queue → consumer thread → validate → store → resample → callbacks
    (Same consumer pattern as Binance client; only parse/connect logic differs)
"""

import json
import time
import threading
from queue import Queue, Empty
from typing import Callable, Optional
from enum import Enum

import websocket
from loguru import logger

from .data_validator import DataValidator
from .resampler import OHLCVResampler

try:
    from .rest_client import BitgetRESTClient
    _HAS_REST = True
except ImportError:
    _HAS_REST = False


class WSState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"


class BitgetWSClient:
    """Real-time Bitget spot kline WebSocket client.

    Usage:
        client = BitgetWSClient(config)
        client.on_candle("1h", lambda c: print(f"1H: {c['close']}"))
        client.start()
        client.stop()
    """

    WS_URL = "wss://ws.bitget.com/v2/ws/public"

    def __init__(self, config: dict, rest_client: Optional["BitgetRESTClient"] = None):
        self.config = config
        ex_cfg = config.get("exchange", {})
        self.symbol = ex_cfg.get("symbols", ["BTC/USDT"])[0]
        self.symbol_id = self.symbol.split(":")[0].replace("/", "")  # BTCUSDT

        # Reconnection config
        self.reconnect_min_delay = 1.0
        self.reconnect_max_delay = 30.0
        self.reconnect_multiplier = 2.0
        self._reconnect_delay = self.reconnect_min_delay

        # Components
        self.validator = DataValidator(config)
        self.resampler = OHLCVResampler()

        # REST client for backfilling — use shared if provided
        if rest_client is not None:
            self.rest_client = rest_client
        elif _HAS_REST:
            try:
                from .rest_client import BitgetRESTClient
                self.rest_client = BitgetRESTClient(config)
            except Exception as e:
                logger.warning(f"REST client unavailable: {e}")
                self.rest_client = None
        else:
            self.rest_client = None

        # Internal queue
        self._queue: Queue = Queue(maxsize=5000)

        # Registered callbacks
        self._callbacks: dict[str, list[Callable]] = {
            "1m": [], "1h": [], "4h": [], "1d": [],
        }

        # State
        self._state = WSState.DISCONNECTED
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Candle completion tracking: {timeframe: last_seen_timestamp}
        self._last_seen: dict[str, int] = {"1m": 0, "1h": 0, "4h": 0, "1d": 0}

        # Stats
        self.candles_received = 0
        self.candles_processed = 0
        self.reconnect_count = 0
        self.last_candle_ts = 0

        # Real-time price tracking
        self.last_price = 0.0
        self.price_24h_ago = 0.0

    # ─── Public API ─────────────────────────────────────────────

    def on_candle(self, timeframe: str, callback: Callable):
        if timeframe not in self._callbacks:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        self._callbacks[timeframe].append(callback)

    def start(self, backfill_hours: int = 24):
        if self._state == WSState.CONNECTED:
            logger.warning("WebSocket already connected")
            return

        self._stop_event.clear()

        # Backfill to prime the resampler
        if self.rest_client and backfill_hours > 0:
            self._backfill_prime(backfill_hours)

        # Start consumer thread
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, name="bocik-ws-consumer", daemon=True,
        )
        self._consumer_thread.start()

        # Start WebSocket connection
        self._ws_thread = threading.Thread(
            target=self._connect_ws, name="bocik-ws-connector", daemon=True,
        )
        self._ws_thread.start()

        logger.info(f"Bitget WS client started (symbol={self.symbol_id})")

    def stop(self):
        logger.info("Stopping Bitget WS client...")
        self._stop_event.set()

        if self._ws:
            self._ws.close()

        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)

        flushed = self.resampler.flush()
        for tf, candle in flushed.items():
            if candle:
                self._fire_callbacks(tf, candle)

        self._state = WSState.STOPPED
        logger.info(f"Bitget WS stopped. Rx:{self.candles_received} Proc:{self.candles_processed}")

    @property
    def state(self) -> WSState:
        return self._state

    def is_running(self) -> bool:
        return self._state in (WSState.CONNECTED, WSState.RECONNECTING)

    # ─── WebSocket Connection ───────────────────────────────────

    def _connect_ws(self):
        """Connect to Bitget WebSocket with auto-reconnect."""
        while not self._stop_event.is_set():
            self._state = WSState.CONNECTING
            logger.info(f"Connecting to {self.WS_URL}...")

            self._ws = websocket.WebSocketApp(
                self.WS_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            try:
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WS run_forever exception: {e}")

            if self._stop_event.is_set():
                break

            self._state = WSState.RECONNECTING
            self.reconnect_count += 1
            logger.warning(
                f"WS disconnected. Reconnecting in {self._reconnect_delay:.1f}s "
                f"(attempt #{self.reconnect_count})"
            )

            self._stop_event.wait(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * self.reconnect_multiplier,
                self.reconnect_max_delay,
            )


    def _on_open(self, ws):
        self._state = WSState.CONNECTED
        self._reconnect_delay = self.reconnect_min_delay

        ex_cfg = self.config.get("exchange", {})
        inst_type = ex_cfg.get("ws_inst_type", "SPOT")

        # Subscribe to 1m candles
        sub_msg = {
            "op": "subscribe",
            "args": [{
                "instType": inst_type,
                "channel": "candle1m",
                "instId": self.symbol_id,
            }],
        }
        ws.send(json.dumps(sub_msg))
        logger.info(f"✓ Bitget WS connected, subscribed to {self.symbol_id} (instType: {inst_type}) candle1m")

        # Backfill missed candles after reconnect
        if self.rest_client and self.last_candle_ts > 0 and self.reconnect_count > 0:
            self._backfill_missed()

    def _on_message(self, ws, message: str):
        """Parse Bitget WS message."""
        try:
            data = json.loads(message)

            # Only process data pushes (skip subscribe confirmations)
            if "data" not in data or "arg" not in data:
                return

            arg = data["arg"]
            channel = arg.get("channel", "")

            # We only care about candle channels
            if not channel.startswith("candle"):
                return

            # Skip snapshots (initial state) — process, but don't emit as new candles
            is_snapshot = data.get("action") == "snapshot"

            # Parse timeframe from channel name
            tf_map = {
                "candle1m": "1m", "candle5m": "5m", "candle15m": "15m",
                "candle1H": "1h", "candle4H": "4h", "candle1D": "1d",
                "candle1Hutc": "1h", "candle4Hutc": "4h", "candle1Dutc": "1d",
            }
            tf = tf_map.get(channel)
            if tf is None:
                return

            # Parse candle array
            for row in data["data"]:
                candle = {
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),  # Base volume
                }

                self.last_price = candle["close"]

                if is_snapshot:
                    # Feed into resampler silently — don't emit yet
                    self.resampler.add_1m_candle(candle)
                    self.last_candle_ts = max(self.last_candle_ts, candle["timestamp"])
                    self._last_seen[tf] = candle["timestamp"]
                    continue

                # Candle completion detection
                prev_ts = self._last_seen.get(tf, 0)
                if candle["timestamp"] == prev_ts and tf == "1m":
                    continue

                self._last_seen[tf] = candle["timestamp"]

                # Enqueue for consumer
                try:
                    self._queue.put_nowait(candle)
                    self.candles_received += 1
                except Exception:
                    logger.warning("Queue full — dropping candle")
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(candle)
                    except Exception:
                        pass

        except json.JSONDecodeError:
            logger.debug(f"Non-JSON WS message: {message[:100]}")
        except Exception as e:
            logger.error(f"WS message error: {e}", exc_info=True)

    def _on_error(self, ws, error):
        logger.error(f"WS error: {error}")

    def _on_close(self, ws, code, msg):
        self._state = WSState.DISCONNECTED
        logger.warning(f"WS closed: code={code} msg={msg}")

    # ─── Consumer Thread ────────────────────────────────────────

    def _consumer_loop(self):
        logger.info("Consumer thread started")
        while not self._stop_event.is_set():
            try:
                candle = self._queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                self._process_candle(candle)
                self.candles_processed += 1
            except Exception as e:
                logger.error(f"Consumer error: {e}", exc_info=True)
            finally:
                self._queue.task_done()

        self._drain_queue()
        logger.info("Consumer thread stopped")

    def _process_candle(self, candle: dict):
        """Validate → fire 1m → resample → fire higher TF."""
        result = self.validator.validate(candle)
        if not result.valid:
            logger.warning(f"Rejected candle ts={candle['timestamp']}: {result.reason}")
            return

        self._fire_callbacks("1m", candle)
        new_candles = self.resampler.add_1m_candle(candle)

        for tf in ["1h", "4h", "1d"]:
            if tf in new_candles and new_candles[tf] is not None:
                self._fire_callbacks(tf, new_candles[tf])

        self.last_candle_ts = candle["timestamp"]

    def _drain_queue(self):
        drained = 0
        while not self._queue.empty():
            try:
                self._process_candle(self._queue.get_nowait())
                self._queue.task_done()
                drained += 1
            except Empty:
                break
            except Exception:
                pass
        if drained:
            logger.info(f"Drained {drained} events from queue")

    # ─── Callbacks ──────────────────────────────────────────────

    def _fire_callbacks(self, timeframe: str, candle: dict):
        for cb in self._callbacks.get(timeframe, []):
            try:
                cb(candle)
            except Exception as e:
                logger.error(f"Callback error ({timeframe}): {e}", exc_info=True)

    # ─── Backfilling ────────────────────────────────────────────

    def _backfill_prime(self, hours: int):
        if not self.rest_client:
            return
        logger.info(f"Priming resampler with {hours}h of historical data...")
        try:
            candles = self.rest_client.fetch_recent(timeframe="1m", hours=hours)
            if candles:
                valid, rejected = self.validator.validate_batch(candles)
                logger.info(f"Primed: {len(valid)} valid, {len(rejected)} rejected")
                if valid:
                    self.price_24h_ago = valid[0]["close"]
                    self.last_price = valid[-1]["close"]
                for c in valid:
                    self.resampler.add_1m_candle(c)
                    self.last_candle_ts = max(self.last_candle_ts, c["timestamp"])
        except Exception as e:
            logger.warning(f"Backfill prime failed: {e}")

    def _backfill_missed(self):
        if not self.rest_client or self.last_candle_ts == 0:
            return
        logger.info(f"Backfilling missed candles since ts={self.last_candle_ts}")
        try:
            now_ms = int(time.time() * 1000)
            start_ms = self.last_candle_ts + 60_000
            if start_ms >= now_ms:
                logger.info("No missed candles to backfill (already up to date)")
                return
            missed = self.rest_client.fetch_ohlcv_range(
                timeframe="1m",
                start_ms=start_ms,
                end_ms=now_ms,
            )
            if missed:
                valid, _ = self.validator.validate_batch(missed)
                logger.info(f"Backfilled {len(valid)} missed candles")
                for c in valid:
                    try:
                        self._queue.put_nowait(c)
                    except Exception:
                        logger.warning("Queue full during backfill — dropping candle")
                    self.last_candle_ts = max(self.last_candle_ts, c["timestamp"])
        except Exception as e:
            logger.warning(f"Backfill missed failed: {e}")

    # ─── Stats ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "state": self._state.value,
            "candles_received": self.candles_received,
            "candles_processed": self.candles_processed,
            "reconnect_count": self.reconnect_count,
            "last_candle_ts": self.last_candle_ts,
            "queue_size": self._queue.qsize(),
        }
