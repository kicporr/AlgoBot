# bocik Dashboard API Reference

Base URL: `http://localhost:8088`

## Authentication

Protected endpoints (`POST`) require either:
- `X-Dashboard-Token` header with the configured token, OR
- Localhost access (`127.0.0.1`, `::1`)

The token is read from the `DASHBOARD_TOKEN` environment variable or `config/.env`.

---

## GET Endpoints

### `GET /api/health`

Lightweight health check. No database queries.

**Query**: none

**Response** `200`:
```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "exchange": "bitget",
  "pipelines": {
    "pure": {"state": "running", "balance": 10150.0, "circuit_breaker": "normal"},
    "ml":   {"state": "running", "balance": 10000.0, "circuit_breaker": "normal"}
  }
}
```

Errors: `{"status": "error"}` or `{"status": "error", "error": "Bot not initialized"}`

---

### `GET /api/status`

Full bot status for a single pipeline.

**Query**: `?pipeline=pure|ml` (default: `pure`)

**Response** `200`:
```json
{
  "pipeline": "pure",
  "bot_name": "bocik",
  "version": "0.1.0",
  "mode": "PAPER",
  "running": true,
  "exchange": "bitget",
  "equity": 10150.0,
  "balance": 10000.0,
  "initial_capital": 10000.0,
  "meta_labeler_enabled": false,
  "active_position": { "symbol": "BTC/USDT:USDT", "side": "LONG", "size": 0.1, "entry_price": 50000, "ts": 1700000000000 },
  "active_positions": [ ... ],
  "btc_price": 67500.0,
  "btc_change_24h": 2.1,
  "tickers": { "BTC/USDT:USDT": { "last_price": 67500, "price_24h": 66100, "change_24h_pct": 2.1 }, ... },
  "circuit_breaker": { "state": "NORMAL", "reason": "", "manual_halted": false },
  "stats": { "total_trades": 45, "win_rate": 62.2, "max_drawdown": 3.2 },
  "proximity": { ... },
  "proximities": { ... },
  "regime": { "regime": "trending", "bar_count": 120, ... },
  "regime_diagnostics": { ... },
  "config": { ... }
}
```

---

### `GET /api/compare`

Side-by-side comparison of Pure vs ML pipelines.

**Query**: none

**Response** `200`:
```json
{
  "pure": {
    "name": "pure",
    "balance": 10150.0,
    "equity": 10150.0,
    "initial_capital": 10000.0,
    "return_pct": 1.5,
    "active_positions": 1,
    "unrealized_pnl": 150.0,
    "trade_count": 45,
    "win_rate": 62.2,
    "drawdown_pct": 0,
    "circuit_breaker": "NORMAL",
    "meta_labeler_enabled": false,
    "meta_labeler_trained": false
  },
  "ml": { ... }
}
```

---

### `GET /api/trades`

Last 50 closed trades for a pipeline.

**Query**: `?pipeline=pure|ml` (default: all)

**Response** `200`:
```json
[
  {
    "entry_time": 1700000000000,
    "exit_time": 1700000144000,
    "side": "long",
    "entry_price": 50000.0,
    "exit_price": 51000.0,
    "quantity": 0.1,
    "pnl": 100.0,
    "pnl_pct": 2.0,
    "strategy": "mtf_macd:BTC/USDT:USDT",
    "exit_reason": "take_profit",
    "theoretical_entry_price": 49995.0,
    "theoretical_exit_price": 51005.0,
    "pipeline": "pure"
  }
]
```

---

### `GET /api/trades/all`

All closed trades (no limit, ascending by exit time).

**Query**: `?pipeline=pure|ml` (optional filter)

**Response**: Same shape as `/api/trades`, array of all trades.

---

### `GET /api/analytics`

Full performance metrics (Sharpe, Sortino, Calmar, etc.).

**Query**: `?pipeline=pure|ml` (optional)

**Response** `200`:
```json
{
  "total_trades": 342,
  "win_rate": 62.1,
  "total_pnl": 5200.0,
  "total_return_pct": 52.0,
  "annualized_return_pct": 28.3,
  "sharpe_ratio": 1.82,
  "sortino_ratio": 2.45,
  "calmar_ratio": 1.10,
  "max_drawdown_pct": 8.5,
  "profit_factor": 2.1,
  "recovery_factor": 1.47,
  "win_count": 212,
  "loss_count": 130,
  "win_loss_count_ratio": 1.63,
  "avg_win": 45.2,
  "avg_loss": -22.1,
  "win_loss_ratio": 2.05,
  "avg_duration_seconds": 43200,
  "formatted_duration": "12h 0m",
  "max_win_trade": { ... },
  "max_loss_trade": { ... },
  "slippage_summary": {
    "global_avg_entry_slip": 0.32,
    "global_avg_exit_slip": 0.18,
    "global_avg_total_slip": 0.50,
    "global_tracked_count": 300,
    "by_symbol": [ ... ]
  }
}
```

---

### `GET /api/signals`

Last 50 recorded signals (executed and rejected).

**Query**: `?pipeline=pure|ml&strategy=mtf_macd&executed=true`

**Response** `200`:
```json
[
  {
    "id": 1,
    "timestamp": 1700000000000,
    "strategy": "mtf_macd:BTC/USDT:USDT",
    "signal": "long",
    "confidence": 0.82,
    "regime": "trending",
    "executed": true,
    "reject_reason": null,
    "pipeline": "ml"
  }
]
```

---

### `GET /api/logs`

Last 100 log lines from the current log file.

**Query**: none (client-side filtering)

**Response** `200`:
```json
[
  {"timestamp": "14:32:01.234", "level": "INFO", "message": "Signal received: BTCUSDT bullish"},
  {"timestamp": "14:32:00.123", "level": "WARNING", "message": "Order fill slippage 0.18%"}
]
```

---

### `GET /api/risk/snapshot`

Full risk dashboard data.

**Query**: none

**Response** `200`:
```json
{
  "risk": { "trade_count": 87, "win_rate": 62.1, "current_drawdown_pct": 3.2, ... },
  "breaker": { "state": "NORMAL", "daily_pnl": 120.0, ... },
  "equity_history": [ { "timestamp": 1700000000000, "equity": 10150, "drawdown_pct": 0 }, ... ],
  "correlation": { "symbols": ["BTC","ETH","SOL","XRP","LTC"], "matrix": { ... } }
}
```

---

### `GET /api/candles`

OHLCV candle data from the feature engine cache (1H, 4H, or 1D).

**Query**: `?symbol=BTC/USDT:USDT&timeframe=1h&limit=50`

**Response** `200`:
```json
[
  {"timestamp": 1700000000000, "open": 50000, "high": 50200, "low": 49800, "close": 50100, "volume": 120.5}
]
```

---

### `GET /api/orders`

Open/pending orders for a pipeline.

**Query**: `?pipeline=pure|ml` (optional)

**Response** `200`:
```json
{
  "orders": [
    {"id": "paper_BTC_long", "symbol": "BTC/USDT:USDT", "side": "LONG", "type": "limit", "price": 50000, "amount": 0.1, "filled": 0.1, "remaining": 0, "status": "filled", "timestamp": 1700000000000}
  ]
}
```

---

### `GET /api/events`

Recent trading events (trades + signals merged, sorted by time desc).

**Query**: `?pipeline=pure|ml` (optional)

**Response** `200`:
```json
[
  {"time": 1700000144000, "type": "trade", "icon": "✅", "msg": "BTC LONG closed (take_profit)", "detail": "PnL: +$100.00", "pnl": 100},
  {"time": 1700000000000, "type": "signal", "icon": "🔵", "msg": "BTC signal LONG — executed", "detail": "Confidence: 0.82", "pnl": 0}
]
```

---

### `GET /api/export/csv`

Download trade history as CSV.

**Query**: `?pipeline=pure|ml` (optional)

**Response**: `Content-Type: text/csv` with `Content-Disposition: attachment`

---

### `GET /api/export/json`

Download trade history as JSON.

**Query**: `?pipeline=pure|ml` (optional)

**Response**: `Content-Type: application/json` with `Content-Disposition: attachment`

---

## POST Endpoints

All POST endpoints require authentication.

### `POST /api/reset`

Reset the circuit breaker (clear halted state, resume trading).

**Body**: none

**Response** `200`:
```json
{"status": "ok", "message": "Circuit breaker reset successfully"}
```

---

### `POST /api/emergency`

Emergency stop: halt all trading immediately.

**Body**: none

**Response** `200`:
```json
{"status": "ok", "message": "Emergency stop triggered"}
```

---

### `POST /api/close`

Close a specific position.

**Query**: `?symbol=BTC/USDT:USDT&side=long`

**Body**: none

**Response** `200`:
```json
{"status": "ok", "closed": "BTC/USDT:USDT long", ...}
```

---

### `POST /api/close/all`

Close ALL open positions across all symbols.

**Body**: none

**Response** `200`:
```json
{"status": "ok", "closed": 3, "total_pnl": 45.20}
```

---

### `POST /api/settings`

Update bot configuration. Writes to `config/settings.yaml` and `.env`, then reinitializes all modules.

**Body**:
```json
{
  "mode": "paper",
  "risk": {
    "max_position_pct": 25,
    "initial_capital": 15000
  },
  "strategies": {
    "mtf_macd_elder": {
      "macd": {"fast": 10, "slow": 20, "signal": 9}
    }
  },
  "meta_labeling": {
    "min_confidence": 0.60
  },
  "telegram": {
    "bot_token": "...",
    "chat_id": "..."
  }
}
```

**Response** `200`:
```json
{"status": "ok", "message": "Settings updated successfully"}
```

**Response** `400`:
```json
{"status": "error", "message": "Invalid request payload"}
```

---

## Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid request payload |
| 404 | Unknown endpoint |
| 500 | Server error (bot not initialized, config missing, etc.) |

## Rate Limits

Dashboard API has no rate limiting. It runs on the same process as the trading bot. Avoid polling faster than 1 request/second on low-resource systems.

## Static Files

Any path not matching `/api/*` serves static files from the `dashboard/` directory. The built React app (`dashboard-react`) is served from here in production.
