# bocik — BTC Algorithmic Trading Bot

**Multi-asset algorithmic trading bot for crypto (Bitget, 1H/4H timeframes) with regime-aware ensemble routing, rigorous risk management, and realistic backtesting methodology.**

## Realistic Backtest Results (OOS, train/test split)

Test period: 2023-11-15 → 2026-06-15 (~2.5 years, 40% of data)
Warmup period: 2020-01 → 2023-11-15 (60% of data, no trading)

| Symbol | Trades | PnL | Sharpe | Max DD | Win Rate |
|---|---|---|---|---|---|
| BTC/USDT | 450 | +$3,791 | **2.38** | 4.3% | 50.4% |
| ETH/USDT | 644 | +$6,192 | **3.26** | 3.6% | 51.5% |
| **Portfolio** (shared $10k) | 1,094 | +$9,983 (+99.8%) | **2.45** | 5.4% | 51.1% |

### Random-Entry Baseline (1000 simulations)

| Metric | Actual | Random Mean | Random P95 |
|---|---|---|---|
| Sharpe | **2.45** | 1.96 | 2.77 |
| PnL | **+$9,983** | +$4,658 | — |
| Max DD | **5.4%** | 7.2% | — |

**Cost of Parameter Uncertainty (walk-forward overfitting): 3.64 Sharpe points**

The walk-forward optimization (which picks optimal params per 3-month fold) inflates Sharpe from **2.45** (true OOS) to **6.10** (optimistic). Always use fixed-parameter backtest with train/test split for honest results.

### Bear Market Test (2022 crypto winter, BTC $47k→$16k)

| Metric | Bot | Random Mean |
|---|---|---|
| Sharpe | **1.17** | 1.09 |
| PnL | +$2,256 (+22.6%) | +$1,363 |
| Max DD | 2.6% | 4.9% |
| Trades | 425 | — |

Bot survives bear market: regime filter keeps it FLAT ~60% of the time. Positive Sharpe in a -66% BTC year.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure API keys
cp config/.env.example config/.env
# Edit config/.env with your Bitget keys

# Full run (first time, ~12 min)
python research/robust_optimizer.py

# Fast iteration (skip WF optimization, load from cache)
python research/robust_optimizer.py --skip-wf

# Fastest (skip random baselines too)
python research/robust_optimizer.py --skip-wf --quick --runs 100

# Paper trading
python orchestrator.py --mode paper
```

## Backtest CLI Flags

| Flag | Effect | Runtime |
|---|---|---|
| *(none)* | Full: WF optimization + multi-asset + random baseline + bear test | ~12 min |
| `--skip-wf` | Load WF params from cache, skip grid search | **~1 min** |
| `--skip-wf --quick` | Skip random-entry baselines too | **~45 sec** |
| `--runs N` | Custom random baseline runs (default 1000, dev: 100) | — |

WF parameters are cached to `data/cache/wf_params.json` after each full run.

## Architecture

```
Data Layer → Feature Engine (67 features) → Regime Classifier → Ensemble Router
                                                                       ↓
                                               MTF MACD Elder (TRENDING) / FLAT (other)
                                                                       ↓
                                               Shared-Capital Execution (multi-asset)
                                                                       ↓
                                               Risk Layer (Circuit Breakers + Exit Chain)
```

- **Data Layer:** Bitget WebSocket + REST, OHLCV resampling, data validation
- **Feature Engine:** 67 features across 7 categories (price, momentum, trend, volatility, volume, patterns, multi-TF)
- **Strategy Layer:** MTF MACD + Elder Filter (primary), Mean Reversion (experimental)
- **Ensemble:** Regime classifier (TRENDING/RANGING/VOLATILE/UNCLEAR) + router
- **Risk Layer:** Kelly position sizing, 6 circuit breakers, exit chain (TP/SL/trailing/time)
- **Execution Layer:** ccxt adapter, limit maker orders, position tracking
- **Monitoring:** Telegram alerts, daily/weekly reports, Grafana (optional)

## Corrected MACD Parameters (modal selection, not averaged)

| Symbol | MACD | Trailing Stop | ATR Stop |
|---|---|---|---|
| BTC | (10, 20, **9**) | 2% | 3.0× |
| ETH | (5, 13, **8**) | 2% | 2.5× |
| XRP | (5, 13, **8**) | 2% | 2.0× |
| SOL | (10, 20, **9**) | 3% | 3.0× |
| LTC | (5, 13, **8**) | 2% | 2.5× |

*Parameters selected by modal (most frequent) triplet across walk-forward folds, not by averaging. Averaging MACD(5,13,8) and MACD(12,26,9) produces MACD(8,20,9) — an indicator no fold ever selected.*

## Project Structure

```
bocik/
├── orchestrator.py            # Main trading loop
├── run_backtest.py            # CLI for backtests
├── config/                    # settings.yaml, .env
├── data/                      # Ingestion (WS, REST) + storage (SQLAlchemy)
├── features/                  # 67 features, 19 indicators, 12 candle patterns
├── strategies/                # MTF MACD, Mean Reversion, XGBoost (experimental)
├── ensemble/                  # Regime classifier + router
├── risk/                      # Position sizing, circuit breakers, risk monitor
├── execution/                 # ccxt adapter, order manager, position tracker
├── backtest/                  # Walk-forward engine, metrics, visualization
├── research/
│   └── robust_optimizer.py    # Main backtest & optimization script
├── monitoring/                # Telegram, logging
├── dashboard/                 # Web dashboard
├── tests/                     # Unit tests
└── docs/                      # Full documentation
```

## Key Methodology Notes

1. **True Multi-Asset Backtest**: All symbols share ONE $10k capital account. Positions are sized from shared equity with max 3 concurrent positions.
2. **Train/Test Split**: 60% warmup (features only, no trading), 40% OOS test. No look-ahead bias.
3. **Modal MACD**: Full triplet (fast, slow, signal) selected atomically. No per-parameter averaging.
4. **Conditional Correlation**: Correlation of PnL calculated ONLY on days with non-zero PnL for both assets — reveals true co-movement risk (e.g., BTC-ETH conditional 0.48 vs unconditional 0.23).
5. **WF Parameter Cache**: After first run, params saved to cache. Use `--skip-wf` for 16× faster iteration.

## License

Private project.
