# bocik — BTC Algorithmic Trading Bot
## Full Documentation & Implementation Status
### June 14, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Implemented Components](#3-implemented-components)
4. [Strategy Layer](#4-strategy-layer)
5. [Backtesting Pipeline](#5-backtesting-pipeline)
6. [Bitget Integration](#6-bitget-integration)
7. [Risk Management](#7-risk-management)
8. [Test Results & Performance](#8-test-results--performance)
9. [Key Design Decisions](#9-key-design-decisions)
10. [File Map](#10-file-map)
11. [How to Run](#11-how-to-run)
12. [Known Issues & Limitations](#12-known-issues--limitations)
13. [Future Plans](#13-future-plans)

---

## 1. Project Overview

**Purpose**: Algorithmic trading bot for BTC on the 1H/4H timeframes using XGBoost machine learning with cost-aware execution filtering.

**Exchange**: Bitget (spot market) — migrated from Binance after Poland futures ban.

**Primary Research Basis**: Ślepaczuk (2026), arXiv 2606.00060 — XGBoost with cost-aware filter on 70,000 hourly BTC observations achieving 65%+ annualized returns, Sharpe >1.0.

**Status**: Core pipeline complete and validated. 117 unit tests pass. Backtest runs on 6.5 years of real BTC data (56,563 bars). Currently at -4.5% on held-out test set (2024-2026 bear market) with clear regime-change limitation.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR (orchestrator.py)            │
│     Trading loop: features → signal → risk → execute       │
└───┬──────────────┬──────────────┬──────────────┬──────────┘
    │              │              │              │
┌───▼──────┐ ┌─────▼──────┐ ┌────▼────┐ ┌──────▼──────────┐
│   DATA   │ │  FEATURES  │ │  RISK   │ │   EXECUTION     │
│  LAYER   │ │   LAYER    │ │  LAYER  │ │    LAYER        │
│          │ │            │ │         │ │                 │
│ • WS feed│ │ • 67 feats │ │ • Kelly │ │ • ccxt adapter  │
│ • REST   │ │ • 19 ind.  │ │ • CB(6) │ │ • Order mgr     │
│ • Valid  │ │ • 12 pat.  │ │ • Equi- │ │ • Position trk  │
│ • Resamp │ │ • Multi-TF │ │   ty trk│ │                 │
│ • DB/SQL │ │            │ │         │ │                 │
└──────────┘ └─────┬──────┘ └────────┘ └─────────────────┘
                   │
          ┌────────▼────────┐
          │  STRATEGY LAYER │
          │                 │
          │ • XGBoost (ML)  │
          │ • MTF MACD      │
          │ • Mean Revert   │
          │ • Ensemble Rtr  │
          │ • Regime Class  │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  MONITORING     │
          │                 │
          │ • Telegram Bot  │
          │ • Loguru Logs   │
          │ • Grafana (todo)│
          └─────────────────┘
```

### Data Flow (per candle):
1. **Bitget WS** → 1m kline pushed
2. **WS Client** → validates, enqueues
3. **Consumer thread** → validates → fires 1m callback → resamples
4. **Resampler** → 1m → 1H/4H/1D → fires callbacks
5. **Orchestrator** `_on_1h_candle`:
   - FeatureEngine computes 67 features
   - RiskMonitor updates equity/drawdown/exposure
   - CircuitBreaker checks 6 triggers
   - EnsembleRouter selects strategy by regime
   - Strategy generates Signal (LONG/SHORT/FLAT)
   - KellyPositionSizer calculates position size
   - Execute (paper or live via ccxt)
6. **Risk snapshot** logged every hour

---

## 3. Implemented Components

### Data Pipeline (Phase 1) ✅
| Component | File | Description |
|---|---|---|
| DataValidator | `data/ingestion/data_validator.py` | 7 validation rules: OHLC sanity, positive prices, dedup, future detection |
| OHLCVResampler | `data/ingestion/resampler.py` | 1m→1H/4H/1D via incremental (live) and bulk (backtest) modes |
| BitgetRESTClient | `data/ingestion/rest_client.py` | ccxt-based: paginated fetch, backfill gap detection, DataFrame export |
| BitgetWSClient | `data/ingestion/ws_client.py` | Thread-safe WS: parse→queue→validate→resample→callback. Auto-reconnect with backfill |
| DatabaseManager | `data/storage/repositories.py` | SQLite/PostgreSQL: WAL mode, bulk inserts, timestamp dedup |

### Feature Engineering (Phase 2) ✅
| Component | File | Description |
|---|---|---|
| FeatureEngine | `features/engine.py` | 67 features across 7 categories. Live mode (process_candle) + bulk mode (bulk_compute) |
| IndicatorCalculator | `features/indicators.py` | 19 pure pandas/numpy indicators: SMA, EMA, ATR, MACD, ADX, RSI, Stochastic, CCI, OBV, MFI, EOM |
| DerivedFeatures | `features/derived.py` | 12 candlestick patterns (doji, engulfing, harami, soldiers/crows), RSI divergence, trend strength, pivot levels |

**Feature Categories**:
1. Price-based (8): returns, log_returns, hl_ratio, close_position, gap
2. Volatility (6): ATR, Bollinger width, historical vol, Garman-Klass, Parkinson
3. Trend (12): MACD family, ADX, DI+/-, EMA slopes, SMA crosses
4. Momentum (10): RSI 7/14, Stochastic K/D, CCI, Williams %R, ROC 5/10/20
5. Volume (7): OBV, MFI 14, volume ratios, Ease of Movement
6. Pattern (15): doji, hammer, engulfing, morning/evening star, marubozu
7. Market Structure (8): distance from SMA 20/50/200, rolling max/min, multi-bar returns

### Strategy Layer (Phase 3+5+8) ✅
| Component | File | Description |
|---|---|---|
| XGBoostCostAware | `strategies/xgb_cost_aware.py` | 4H horizon prediction, cost filter λ=4.0, trend alignment |
| MTF_MACD_Elder | `strategies/mtf_macd.py` | D1 MACD trend filter, 1H MACD crossover entry, multi-exit |
| MeanReversion | `strategies/mean_reversion.py` | RSI oversold + BB lower-band touch for ranging markets |
| RegimeClassifier | `ensemble/regime_classifier.py` | 4 regimes: TRENDING/RANGING/VOLATILE/UNCLEAR via ADX+BB+ATR |
| EnsembleRouter | `ensemble/router.py` | TRENDING→MACD, RANGING→MR, VOLATILE→FLAT, UNCLEAR→XGBoost |

### Risk Management (Phase 4) ✅
| Component | File | Description |
|---|---|---|
| KellyPositionSizer | `risk/position_sizer.py` | Half-Kelly with volatility adjustment (shrinks position when ATR spikes) |
| CircuitBreaker | `risk/circuit_breaker.py` | 6 triggers: drawdown, daily/weekly loss, consecutive losses (3/5), vol spike, doom loop |
| RiskMonitor | `risk/risk_monitor.py` | Equity tracking, drawdown, daily/weekly PnL, Grafana snapshots |

### Backtesting (Phase 3) ✅
| Component | File | Description |
|---|---|---|
| BacktestEngine | `backtest/engine.py` | Walk-forward CV, TP/SL/time exits, realistic fees, equity curves |
| Metrics | `backtest/metrics.py` | Sharpe, Sortino, Calmar, profit factor, fold stability report |

### ML Pipeline (Phase 7) ✅
| Component | File | Description |
|---|---|---|
| WalkForwardTrainer | `ml/trainer.py` | Walk-forward XGBoost training, feature importance, model save/load |
| ModelRegistry | `ml/model_registry.py` | Versioned model storage with metadata |

### Monitoring (Phase 9) ✅
| Component | File | Description |
|---|---|---|
| TelegramAlerter | `monitoring/telegram_bot.py` | HTTP Bot API: trade alerts, risk snapshots, circuit breaker warnings |
| Logger | `monitoring/logger.py` | loguru with rotation, per-module levels |

### Execution ✅
| Component | File | Description |
|---|---|---|
| ExchangeAdapter | `execution/exchange_adapter.py` | ccxt Bitget wrapper: orders, balance, order book |
| OrderManager | `execution/order_manager.py` | Order lifecycle management |
| PositionTracker | `execution/position_tracker.py` | SL/TP/trailing stop tracking |

---

## 4. Strategy Layer — Detailed Logic

### XGBoost Cost-Aware (Primary)
```
Signal generation:
  1. Model predicts 4H return direction probability (0..1)
  2. Cost threshold = λ × (transaction_cost_bps / 10000)
  3. LONG:  prob >= 0.5 + threshold AND price > SMA50
  4. SHORT: prob <= 0.5 - threshold AND price < SMA50
  5. FLAT:  otherwise

Retraining:
  - Triggered every 500 candles by orchestrator
  - Uses expanding window of all past feature data
  - Target: binary up/down over next 4 bars
  - Features: all 67 FeatureEngine columns (excludes OHLCV)

Parameters:
  λ (cost multiplier): 4.0
  Transaction cost: 30 bps (0.3% round-trip estimate)
  Confidence threshold: 0.55
  Model: XGBoost Classifier (200 estimators, max_depth=4, lr=0.03)
  Regularization: alpha=2.0, lambda=3.0, subsample=0.7
```

### MTF MACD + Elder Filter (Trending markets)
```
Signal generation:
  1. D1 MACD > D1 Signal → trend is UP
  2. 1H MACD crosses above Signal AND D1 UP → LONG
  3. 1H MACD crosses below Signal AND D1 DOWN → SHORT

Exit conditions (priority order):
  1. MACD opposite cross → exit
  2. Take-profit (2× risk) → exit
  3. Trailing stop (3%) → exit
  4. ATR stop (2× ATR) → exit
  5. Time exit (24h max hold) → exit

Elder filter: optional volume confirmation (1.2× average)
```

### Mean Reversion (Ranging markets)
```
Entry:
  LONG:  RSI <= 30 AND price near lower Bollinger Band (bb_position <= 0.05)
  SHORT: RSI >= 70 AND price near upper Bollinger Band (bb_position >= 0.95)

Exit:
  LONG:  RSI >= 50 OR bb_position >= 0.5
  SHORT: RSI <= 50 OR bb_position <= 0.5
```

---

## 5. Backtesting Pipeline

### Walk-Forward Cross-Validation
```
1. Split data into 60% train / 20% validation / 20% test (chronological)
2. For each fold in train set:
   a. Train XGBoost on all data before fold boundary
   b. Simulate trading on fold's unseen data
   c. Record trades, PnL, metrics
3. Evaluate best configuration on validation set
4. Final evaluation on untouched test set

Anti-lookahead measures:
  - Signal generated on bar N, executed on bar N+1 open
  - SL/TP checked within candle using high/low
  - All features computed from price data available at that point only
```

### Exit Priority (per candle, in position):
```
1. Take-profit hit    (highest priority — lock in profits)
2. Signal reversal    (strategy-issued opposite signal)
3. Trailing stop      (3% from peak)
4. ATR stop           (2× ATR from entry)
5. Time exit          (24h max hold)
```

### Fee Model:
- Taker fee: 0.1% per side (2 trades = 0.2%)
- Slippage: 0.05% per side (2 trades = 0.1%)
- Total round-trip cost: 0.3%

---

## 6. Bitget Integration

### Key Differences from Binance:
| Feature | Binance | Bitget |
|---|---|---|
| Auth | API key + secret | API key + secret + **passphrase** |
| WS URL | `wss://stream.binance.com:9443/ws` | `wss://ws.bitget.com/v2/ws/public` |
| WS subscribe | URL path (`/btcusdt@kline_1m`) | JSON: `{"op":"subscribe","args":[...]}` |
| Kline format | Nested dict with `k.x` close flag | Flat array `[ts,o,h,l,c,vol]` |
| 1H candle limit | 1,000 per request | 200 per request |
| Pagination quirk | Works with limit=1000 | Limit>100 ignores `since` param |
| Testnet | ✅ Available | ❌ Not available |
| Fees | 0.1% maker/taker | 0.1% (0.08% with BGB) |
| ccxt class | `ccxt.binance()` | `ccxt.bitget()` |

### Bitget Pagination Workaround:
```
Problem: Bitget API ignores `since` parameter when limit > 100.
Fix: Use limit=100 and paginate manually:
  - Fetch 100 candles from since date
  - Set next since = last_candle_timestamp + 1
  - Repeat until reaching end date
  - Deduplicate and sort
```

---

## 7. Risk Management

### Kelly Position Sizing:
```
f* = (p × b - q) / b
where: p=win rate, q=loss rate, b=avg_win/avg_loss
Position = f* × fraction × volatility_adj × capital / price

Volatility adjustment: min(1.0, avg_ATR / current_ATR)
  - When volatility is 2× normal, position reduced by 50%
  - Floor: 25% of original size
```

### Circuit Breaker (6 triggers):
| Trigger | Threshold | Action |
|---|---|---|
| Max drawdown | >20% from peak equity | HALTED |
| Daily loss | >5% of capital | HALTED |
| Weekly loss | >10% of capital | HALTED |
| Consecutive losses | 3 | WARNING (skip 1 bar) |
| Consecutive losses | 5 | HALTED |
| Volatility spike | ATR >5× rolling avg | HALTED |
| Doom loop | >5 trades/hour, >20/day | HALTED |

---

## 8. Test Results & Performance

### Unit Tests: 117/117 pass

### Backtest Results — 6.5 years BTC (2020-2026):
```
═════════════════════════════════════════════════════════════
            Period        Trades   WR     PnL      Sharpe
  TRAIN     2020-2024     355    51.8%  +$18,889  +21.15
  VAL       2024           82    40.2%  +$1,639    +8.00
  TEST      2024-2026      50    38.0%  -$449      -4.24
═════════════════════════════════════════════════════════════
```

### Performance Benchmarks:
| Data Size | Feature Compute | Full Backtest |
|---|---|---|
| 2,000 bars (3mo) | 0.09s | 0.38s |
| 10,000 bars (1yr) | 0.19s | 1.63s |
| 50,000 bars (5.7yr) | 0.72s | 7.85s |
| 56,563 bars (6.5yr) | ~0.8s | 77s (test split) |

### XGBoost Training Time:
- ~5-10 seconds per fold on 5,000-10,000 samples
- Train split (34k bars, 15 folds): ~5 minutes

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| Pure pandas/numpy indicators | No TA-lib C-compilation issues on Windows |
| 4H horizon prediction | 1H noise too high — 4H shows learnable patterns |
| Cost-aware filter (λ=4.0) | Prevents fee destruction — 483 trades → 50 on test |
| Long+short with trend filter | 50/50 up/down market needs both directions |
| TP/SL/trail/time exit chain | Multiple exit types prevent single-point failures |
| Walk-forward only | Prevents look-ahead bias — gold standard for financial ML |
| 60/20/20 split | Proper ML practice: train, validate, test on unseen data |
| Half-Kelly + vol adjustment | Reduces drawdown vs. full Kelly |
| Limit orders only | Captures maker fees (lower cost) |
| Paper trading mandatory | Minimum before any live capital |

---

## 10. File Map

```
bocik/
├── orchestrator.py              # Main trading loop
├── run_backtest.py              # CLI: fetch + backtest + report
├── requirements.txt             # Python dependencies
├── Dockerfile / docker-compose.yml
├── config/
│   ├── settings.yaml            # All parameters
│   ├── .env                     # API keys (gitignored)
│   └── .env.example
├── data/
│   ├── ingestion/
│   │   ├── ws_client.py         # Bitget WebSocket
│   │   ├── rest_client.py       # Bitget REST + pagination
│   │   ├── data_validator.py    # 7 validation rules
│   │   └── resampler.py         # 1m→1H/4H/1D
│   ├── storage/
│   │   ├── models.py            # SQLAlchemy ORM
│   │   └── repositories.py      # CRUD operations
│   └── cache/                   # Parquet data files
├── features/
│   ├── engine.py                # 67 features via FeatureEngine
│   ├── indicators.py            # 19 pure-pandas indicators
│   └── derived.py               # 12 patterns + structure
├── strategies/
│   ├── base.py                  # BaseStrategy, Signal enum
│   ├── xgb_cost_aware.py        # XGBoost + cost filter
│   ├── mtf_macd.py              # MTF MACD + Elder
│   └── mean_reversion.py        # RSI + BB touch
├── ensemble/
│   ├── regime_classifier.py     # 4-regime detector
│   └── router.py                # Strategy router
├── risk/
│   ├── position_sizer.py        # Half-Kelly + vol adj
│   ├── circuit_breaker.py       # 6-trigger safety
│   └── risk_monitor.py          # Equity + drawdown tracking
├── backtest/
│   ├── engine.py                # Walk-forward simulator
│   └── metrics.py               # Full metrics suite
├── ml/
│   ├── trainer.py               # Walk-forward XGBoost
│   └── model_registry.py        # Versioned models
├── execution/
│   ├── exchange_adapter.py      # ccxt Bitget wrapper
│   ├── order_manager.py         # Order lifecycle
│   └── position_tracker.py      # SL/TP tracker
├── monitoring/
│   ├── telegram_bot.py          # Trade alerts
│   └── logger.py                # loguru config
└── tests/
    ├── test_data_pipeline.py    # 28 tests
    ├── test_features.py         # 33 tests
    ├── test_strategies.py       # 15 tests
    ├── test_risk.py             # 27 tests
    └── test_ensemble.py         # 14 tests
```

---

## 11. How to Run

### Setup:
```bash
pip install -r requirements.txt
# Edit config/.env with your Bitget keys:
#   BITGET_API_KEY=***
#   BITGET_SECRET_KEY=***
#   BITGET_PASSPHRASE=your_passphrase
```

### Verify API:
```bash
python test_api.py
# Should show: [OK] Server time, [OK] Markets, [OK] Balance, [OK] Ticker
```

### Backtest on recent data:
```bash
python run_backtest.py --days 90
python run_backtest.py --since 2020-01-01 --trailing-stop 0.05 --shorts
```

### Paper trade (live data, simulated execution):
```bash
python orchestrator.py --mode paper
```

### Run tests:
```bash
python -m pytest tests/ -v
```

### Backtest with 60/20/20 split:
```bash
python final_backtest.py  # Requires btc_1h_2020_2026.parquet in cache
```

---

## 12. Known Issues & Limitations

| Issue | Severity | Details |
|---|---|---|
| **Regime change failure** | High | Model trained on bull market fails on bear market. Ensemble router is the fix. |
| **Train-test gap** | High | Sharpe 21 on train vs -4 on test — classic overfitting pattern when market regime changes |
| **Limited short history** | Medium | Bitget spot BTC/USDT only goes back to 2020. Binance has data from 2018. |
| **No cross-validation on hyperparams** | Medium | Single parameter set used. Full grid search OOM-killed due to memory. |
| **Bitget API quirks** | Medium | Limit>100 ignores `since` — requires manual pagination workaround |
| **No Grafana dashboards** | Low | Docker compose has Grafana but no dashboards configured |
| **Windows Unicode** | Low | Emoji/Unicode crashes on CP1250 terminal — use ASCII in print statements |

---

## 13. Future Plans

### Immediate (Phase 8 completion):
- [ ] Wire ensemble router into backtest engine (currently XGBoost-only)
- [ ] Backtest with regime routing: TRENDING→MACD, RANGING→MR, VOLATILE→FLAT
- [ ] Compare ensemble vs single-strategy performance

### Optimization:
- [ ] Full hyperparameter sweep with memory-efficient batching
- [ ] Cross-validate cost filter lambda (currently fixed at 4.0)
- [ ] Experiment with prediction horizons: 8H, 12H, 24H
- [ ] Add regime-specific XGBoost models (one per regime)

### Production:
- [ ] Paper trading on Bitget live data (2-week minimum)
- [ ] Grafana dashboard setup with real-time metrics
- [ ] Docker production deployment
- [ ] Automated daily backtest + Telegram summary

### Research:
- [ ] Fetch even more data (Binance 2018-2020 gap)
- [ ] Meta-labeling: second model to filter XGBoost signals
- [ ] Ensemble of multiple ML models (LightGBM, CatBoost)
- [ ] Include on-chain data as features (hash rate, active addresses, etc.)
- [ ] Reinforcement learning for position management (when to exit)

---

*Document generated from a live implementation with 117 passing tests and real BTC backtesting.*
