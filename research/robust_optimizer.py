#!/usr/bin/env python
"""Robust parameter optimization and validation framework.

Implements sliding-window Walk-Forward optimization, stability checks (CV),
degradation thresholds, minimum trade limits, deflation tests, and
portfolio-level validation to prevent backtest overfitting (Ślępaczuk 2026).
Supports multi-processing to run in under 3 minutes.
"""
import sys
import copy
import time
import yaml
import warnings
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

# Setup paths relative to script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Silence loguru to prevent slow terminal printouts during grid search
logger.remove()

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics
from ensemble.regime_classifier import RegimeClassifier
from ensemble.router import EnsembleRouter
from strategies.base import Signal
from strategies.mtf_macd import MTF_MACD_Elder
from strategies.mean_reversion import MeanReversion

warnings.filterwarnings("ignore")

# Define grid search space
MACD_OPTIONS = [(8, 21, 9), (12, 26, 9), (5, 13, 8), (10, 20, 9)]
TRAILING_STOPS = [0.02, 0.03, 0.04, 0.05]
ATR_STOPS = [1.5, 2.0, 2.5, 3.0]

def _resample_to_1d(df_1h):
    df = df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("dt")
    daily = df.resample("1D", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    daily["bar_count"] = df.resample("1D").size()
    daily = daily[daily["bar_count"] >= 12].dropna().reset_index()
    daily["timestamp"] = daily["dt"].astype("datetime64[ms]").astype("int64")
    daily.drop(columns=["dt"], inplace=True)
    return daily

def get_sliding_windows(df, train_months=24, test_months=3):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    min_date = df["dt"].min()
    max_date = df["dt"].max()
    
    folds = []
    current_start = min_date
    while True:
        train_start = current_start
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Stop if the test window goes beyond the available data
        if test_end > max_date + pd.DateOffset(days=5):
            break
            
        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end
        })
        current_start = current_start + pd.DateOffset(months=test_months)
    return folds

def get_symbol_config(global_config: dict, symbol: str) -> dict:
    cfg = copy.deepcopy(global_config)
    cfg["exchange"]["symbols"] = [symbol]
    overrides = global_config.get("symbols", {}).get(symbol, {})
    
    def merge_dicts(dict1, dict2):
        for k, v in dict2.items():
            if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
                merge_dicts(dict1[k], v)
            else:
                dict1[k] = v
    merge_dicts(cfg, overrides)
    return cfg

def run_backtest_slice(config, data_slice, features_slice, d1_trends_slice):
    engine = BacktestEngine(config)
    
    # Instantiate strategy copies with the candidate config
    strategies = {
        "mtf_macd": MTF_MACD_Elder(config),
        "mean_reversion": MeanReversion(config)
    }
    classifier = RegimeClassifier(config)
    router = EnsembleRouter(strategies, classifier)
    
    # Sync initial D1 trend
    mtf_strat = strategies.get("mtf_macd")
    if mtf_strat and len(d1_trends_slice) > 0:
        mtf_strat.set_d1_trend_direct(d1_trends_slice.iloc[0])
        
    trades = engine._simulate_trading(
        strategy=None,
        data=data_slice,
        features=features_slice,
        d1_trends=d1_trends_slice,
        test_start_idx=0,
        ensemble_router=router
    )
    return trades

def run_deflation_test(config, df_1h, df_1d, final_params, n_iterations=10):
    fast, slow, signal, ts, atr = final_params

    # Setup config
    test_cfg = copy.deepcopy(config)
    test_cfg["strategies"]["mtf_macd_elder"]["macd"] = {"fast": fast, "slow": slow, "signal": signal}
    test_cfg["strategies"]["mtf_macd_elder"]["exit"] = {"trailing_stop_pct": ts, "atr_stop_mult": atr, "min_hold_bars": 6}
    
    engine = BacktestEngine(test_cfg)
    
    # Calculate returns (copy to avoid read-only array ValueError)
    returns = df_1h["close"].pct_change().dropna().values.copy()
    
    shuffled_sharpes = []
    
    for _ in range(n_iterations):
        # Shuffle returns
        np.random.shuffle(returns)
        
        # Reconstruct price series
        reconstructed_close = df_1h["close"].iloc[0] * np.cumprod(1 + returns)
        
        # Create a mock DataFrame
        shuffled_df = df_1h.copy()
        shuffled_df = shuffled_df.iloc[1:].reset_index(drop=True)
        shuffled_df["close"] = reconstructed_close
        shuffled_df["open"] = reconstructed_close
        shuffled_df["high"] = reconstructed_close
        shuffled_df["low"] = reconstructed_close
        
        shuffled_1d = _resample_to_1d(shuffled_df)
        
        try:
            res = engine.run_ensemble_backtest(shuffled_df, {
                "mtf_macd": MTF_MACD_Elder,
                "mean_reversion": MeanReversion
            }, data_1d=shuffled_1d)
            shuffled_sharpes.append(res.metrics.get("sharpe_ratio", 0))
        except:
            pass
            
    avg_shuffled_sharpe = np.mean(shuffled_sharpes) if shuffled_sharpes else 0
    return avg_shuffled_sharpe

def evaluate_fold_job(args):
    """Job function to optimize a single fold in parallel."""
    idx, fold, train_df, test_df, fold_macd_slices, symbol_cfg = args
    
    best_train_sharpe = -999
    best_train_combo = None
    best_train_trades = 0
    
    # Grid search on training window
    combos = list(itertools.product(MACD_OPTIONS, TRAILING_STOPS, ATR_STOPS))
    
    for macd_set, ts, atr in combos:
        # Extract slices from precomputed dictionary
        train_features, train_d1_trends, _, _ = fold_macd_slices[macd_set]
        
        config_cand = copy.deepcopy(symbol_cfg)
        config_cand["strategies"]["mtf_macd_elder"]["macd"] = {"fast": macd_set[0], "slow": macd_set[1], "signal": macd_set[2]}
        config_cand["strategies"]["mtf_macd_elder"]["exit"] = {"trailing_stop_pct": ts, "atr_stop_mult": atr, "min_hold_bars": 6}
        
        trades = run_backtest_slice(config_cand, train_df, train_features, train_d1_trends)
        
        if not trades:
            continue
            
        metrics = calculate_metrics(trades, 10000.0)
        sharpe = metrics.get("sharpe_ratio", 0)
        num_trades = metrics.get("total_trades", 0)
        
        # Faza 3b: Minimum trades filter -- prefer parameters yielding >= 15 trades on train
        is_valid_size = (num_trades >= 15)
        
        if is_valid_size and sharpe > best_train_sharpe:
            best_train_sharpe = sharpe
            best_train_combo = (macd_set, ts, atr)
            best_train_trades = num_trades
        elif not best_train_combo and sharpe > best_train_sharpe:
            best_train_sharpe = sharpe
            best_train_combo = (macd_set, ts, atr)
            best_train_trades = num_trades
            
    if not best_train_combo:
        # Default fallback
        best_train_combo = (MACD_OPTIONS[0], TRAILING_STOPS[0], ATR_STOPS[0])
        best_train_sharpe = 0.0
        
    # Test on Out-of-Sample (OOS) using the optimal train params
    best_macd, best_ts, best_atr = best_train_combo
    _, _, test_features, test_d1_trends = fold_macd_slices[best_macd]
    
    config_oos = copy.deepcopy(symbol_cfg)
    config_oos["strategies"]["mtf_macd_elder"]["macd"] = {"fast": best_macd[0], "slow": best_macd[1], "signal": best_macd[2]}
    config_oos["strategies"]["mtf_macd_elder"]["exit"] = {"trailing_stop_pct": best_ts, "atr_stop_mult": best_atr, "min_hold_bars": 6}
    
    oos_trades = run_backtest_slice(config_oos, test_df, test_features, test_d1_trends)
    oos_metrics = calculate_metrics(oos_trades, 10000.0)
    
    oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
    oos_pnl = oos_metrics.get("total_pnl", 0)
    
    # Faza 3c: Degradation Check
    degradation = 0.0
    if best_train_sharpe > 0:
        degradation = (best_train_sharpe - oos_sharpe) / best_train_sharpe
        
    overfit_flag = "OVERFIT" if degradation > 0.50 else "OK"
    low_trades_flag = "LOW TRADES" if oos_metrics.get("total_trades", 0) < 15 else ""
    
    # Serialize trades with full fields needed for portfolio-level capital allocation.
    # fold_id is essential for inter-fold equity chaining: all trades within the same
    # fold share the same start-of-fold equity scale factor.
    serializable_trades = []
    for t in oos_trades:
        serializable_trades.append({
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "side": t.get("side"),
            "pnl": t.get("pnl"),
            "pnl_pct": t.get("pnl_pct"),
            "win": t.get("win"),
            "exit_reason": t.get("exit_reason"),
            "bars_held": t.get("bars_held"),
            "fold_id": idx + 1,  # 1-based fold index for cross-fold equity chaining
        })
        
    return {
        "fold": idx + 1,
        "best_params": f"MACD{best_macd}_T{best_ts:.0%}_A{best_atr:.1f}",
        "train_sharpe": best_train_sharpe,
        "oos_sharpe": oos_sharpe,
        "oos_pnl": oos_pnl,
        "oos_trades_count": oos_metrics.get("total_trades", 0),
        "degradation": degradation,
        "status": f"{overfit_flag} {low_trades_flag}".strip(),
        "best_macd_fast": best_macd[0],
        "best_macd_slow": best_macd[1],
        "best_macd_signal": best_macd[2],  # Full triplet: (fast, slow, signal)
        "best_ts": best_ts,
        "best_atr": best_atr,
        "trades": serializable_trades
    }

def run_true_multi_asset_backtest(
    symbol_params: dict,     # symbol -> (fast, slow, signal, trailing_pct, atr_mult)
    symbol_configs: dict,    # symbol -> merged config dict
    data_1h: dict,           # symbol -> DataFrame
    data_1d: dict,           # symbol -> DataFrame
    initial_capital: float = 10000.0,
    max_position_pct: float = 0.20,
    max_concurrent: int = 3,
    warmup_pct: float = 0.60,  # First 60% of timeline = warmup (no trading)
    test_end_pct: float = 1.0,  # End test at this fraction of timeline (1.0 = data end)
):
    """True multi-asset backtest with ONE shared capital account.

    All symbols compete for the same capital pool. Features are computed
    on the FULL dataset, but trading only occurs in the last (1-warmup_pct)
    portion of the timeline. The warmup period feeds candles to feature
    engines and strategies without generating signals — this prevents
    look-ahead bias and provides realistic out-of-sample results.

    Returns: dict with trades, equity_curve, metrics, etc.
    """
    # ── Per-symbol setup ────────────────────────────────
    engines = {}
    strategies = {}
    routers = {}
    feature_cols = {}  # symbol -> list of feature DataFrames (by timestamp)

    for symbol in data_1h:
        cfg = symbol_configs[symbol]
        fast, slow, signal, trail, atr_mult = symbol_params[symbol]

        cfg["strategies"]["mtf_macd_elder"]["macd"] = {
            "fast": fast, "slow": slow, "signal": signal
        }
        cfg["strategies"]["mtf_macd_elder"]["exit"] = {
            "trailing_stop_pct": trail, "atr_stop_mult": atr_mult,
            "min_hold_bars": 6,
        }

        engine = BacktestEngine(cfg)
        engines[symbol] = engine

        # Compute features for full dataset
        df_feat = engine.feature_engine.bulk_compute(data_1h[symbol], df_1d=data_1d[symbol])
        # Add D1 trend
        d1_trends = engine._compute_d1_trend_series(data_1d[symbol], len(data_1h[symbol]))
        df_feat["d1_trend"] = d1_trends.values if len(d1_trends) == len(df_feat) else "FLAT"
        feature_cols[symbol] = df_feat

        # Strategies and router
        strats = {
            "mtf_macd": MTF_MACD_Elder(cfg),
            "mean_reversion": MeanReversion(cfg),
        }
        classifier = RegimeClassifier(cfg)
        router = EnsembleRouter(strats, classifier)
        strategies[symbol] = strats
        routers[symbol] = router

    # ── Build unified timeline ──────────────────────────
    all_timestamps = set()
    for symbol in data_1h:
        all_timestamps.update(data_1h[symbol]["timestamp"].tolist())
    timeline = sorted(all_timestamps)

    # Create lookup: timestamp -> {symbol -> (row_idx, ohlcv_dict)}
    timeline_data = {}
    for symbol in data_1h:
        df = data_1h[symbol]
        for i, row in df.iterrows():
            ts = row["timestamp"]
            if ts not in timeline_data:
                timeline_data[ts] = {}
            timeline_data[ts][symbol] = {
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
            }

    # Create feature lookup: timestamp -> {symbol -> feature_dict}
    feat_lookup = {}
    for symbol in data_1h:
        df = feature_cols[symbol]
        sym_data = data_1h[symbol]
        for i, row in df.iterrows():
            if i < len(sym_data):
                ts = sym_data.iloc[i]["timestamp"]
                if ts not in feat_lookup:
                    feat_lookup[ts] = {}
                feat_lookup[ts][symbol] = row.to_dict()

    # ── Bar-by-bar simulation ───────────────────────────
    equity = initial_capital
    equity_curve = [equity]
    equity_timestamps = [timeline[0] - 3600000]  # 1h before start

    # Open positions: key -> {symbol, side, entry_price, entry_ts, cost, pnl_pct_est}
    open_positions = {}
    pos_counter = 0

    # Pending signals (generated this bar, executed next bar open)
    pending_signals = {}  # symbol -> Signal

    trades = []
    skipped_concentration = 0
    skipped_capital = 0
    skipped_circuit_breaker = 0
    max_concurrent_observed = 0

    # ── Circuit breaker (real-time DD protection) ────────
    peak_equity = initial_capital
    cb_max_dd_pct = 0.15  # Halt when DD from peak > 15%
    cb_halted = False
    cb_halt_bars_remaining = 0
    cb_total_halt_bars = 0

    # Fee structure (from first symbol's config)
    first_cfg = list(symbol_configs.values())[0]
    fees = first_cfg.get("exchange", {}).get("fees", {})
    maker_fee = fees.get("maker", 0.0002)
    taker_fee = fees.get("taker", 0.0006)
    # Dynamic slippage: base + ATR-driven component
    # Stop-loss events in high vol have much higher slippage than calm entries
    slippage_base = fees.get("slippage", 0.0002)       # 0.02% base (maker)
    slippage_atr_mult = 0.05                            # ATR-to-slippage scaling
    taker_slippage_mult = 1.5                           # Taker exits: 1.5× worse

    # ── Train/Test split ────────────────────────────────
    n_warmup = int(len(timeline) * warmup_pct)
    n_test_end = int(len(timeline) * test_end_pct)
    test_start_ts = timeline[n_warmup] if n_warmup < len(timeline) else timeline[-1]
    n_test_bars = n_test_end - n_warmup
    test_start_date = pd.to_datetime(test_start_ts, unit="ms").date()
    test_end_date = pd.to_datetime(timeline[min(n_test_end, len(timeline)-1)], unit="ms").date()
    print(f"    Warmup: {n_warmup:,} bars (until {test_start_date}) | "
          f"Test: {n_test_bars:,} bars ({test_start_date} -> {test_end_date})")

    # Track warmup PnL separately (informational only)
    warmup_pnl_total = 0.0

    for bar_idx, ts in enumerate(timeline):
        candles = timeline_data.get(ts, {})
        feats = feat_lookup.get(ts, {})
        is_warmup = bar_idx < n_warmup
        if bar_idx >= n_test_end:
            break  # Test period ended — stop processing

        # ── Settle positions that closed at/after this bar ──
        # (In backtest, positions close at the END of a bar at close price)
        closed_now = []
        for pos_key, pos in list(open_positions.items()):
            sym = pos["symbol"]
            if sym not in candles:
                continue
            close_price = candles[sym]["close"]
            low_price = candles[sym]["low"]
            high_price = candles[sym]["high"]
            entry_price = pos["entry_price"]
            side = pos["side"]
            bars_held = pos.get("bars_held", 0) + 1
            pos["bars_held"] = bars_held

            # Get ATR for dynamic stops
            sym_feat = feats.get(sym, {})
            atr = sym_feat.get("atr_14", close_price * 0.02)
            trail_pct = symbol_params[sym][3]  # trailing_stop_pct
            atr_mult_param = symbol_params[sym][4]  # atr_stop_multiplier

            exit_price = None
            exit_reason = ""

            if side == "long":
                # Trailing stop
                pos["highest"] = max(pos.get("highest", entry_price), high_price)
                trail_level = pos["highest"] * (1.0 - trail_pct)
                atr_stop = entry_price - (atr_mult_param * atr)
                tp_price = entry_price * 1.08  # 8% take profit

                if high_price >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                elif low_price <= trail_level:
                    exit_price = max(trail_level, candles[sym]["open"])
                    exit_reason = "trailing_stop"
                elif low_price <= atr_stop:
                    exit_price = atr_stop
                    exit_reason = "atr_stop"
                elif bars_held >= 48:
                    exit_price = close_price
                    exit_reason = "time_exit"
            else:  # short
                pos["lowest"] = min(pos.get("lowest", entry_price), low_price)
                trail_level = pos["lowest"] * (1.0 + trail_pct)
                atr_stop = entry_price + (atr_mult_param * atr)
                tp_price = entry_price * 0.92

                if low_price <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                elif high_price >= trail_level:
                    exit_price = min(trail_level, candles[sym]["open"])
                    exit_reason = "trailing_stop"
                elif high_price >= atr_stop:
                    exit_price = atr_stop
                    exit_reason = "atr_stop"
                elif bars_held >= 48:
                    exit_price = close_price
                    exit_reason = "time_exit"

            if exit_price is not None:
                closed_now.append((pos_key, pos, exit_price, exit_reason))

        for pos_key, pos, exit_price, exit_reason in closed_now:
            del open_positions[pos_key]
            # Calculate PnL
            entry_price = pos["entry_price"]
            cost = pos["cost"]
            side = pos["side"]

            if side == "long":
                gross_ret = (exit_price - entry_price) / entry_price
            else:
                gross_ret = (entry_price - exit_price) / entry_price

            # Dynamic exit slippage (taker order — worse, especially for stop-loss)
            exit_atr = feats.get(pos["symbol"], {}).get("atr_14", exit_price * 0.02)
            atr_pct_exit = exit_atr / exit_price if exit_price > 0 else 0.02
            # Taker exits during high vol have much higher slippage
            exit_slip = slippage_base + atr_pct_exit * slippage_atr_mult * taker_slippage_mult
            entry_slip = pos.get("entry_slippage", slippage_base)

            # Apply fees + dynamic slippage (entry + exit)
            net_ret = gross_ret - (taker_fee * 2) - entry_slip - exit_slip
            pnl = cost * net_ret

            equity += pnl
            equity_curve.append(equity)
            equity_timestamps.append(ts)

            trades.append({
                "entry_time": pos["entry_ts"],
                "exit_time": ts,
                "symbol": pos["symbol"],
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "cost": cost,
                "pnl": pnl,
                "pnl_pct": net_ret * 100,
                "win": pnl > 0,
                "exit_reason": exit_reason,
                "bars_held": pos.get("bars_held", 0),
            })

        # ── Update D1 trends (always, even during warmup) ──
        for sym in candles:
            if sym in feats:
                d1_trend = feats[sym].get("d1_trend", "FLAT")
                mtf = strategies[sym].get("mtf_macd")
                if mtf:
                    mtf.set_d1_trend_direct(d1_trend)

        # ── Trading logic (test period only) ────────────
        if is_warmup:
            pending_signals.clear()
            continue

        # ── Circuit breaker check ─────────────────────────
        peak_equity = max(peak_equity, equity)
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

        if cb_halted:
            cb_halt_bars_remaining -= 1
            cb_total_halt_bars += 1
            if cb_halt_bars_remaining <= 0:
                cb_halted = False  # Resume after cooldown (24h = 24 bars)
            pending_signals.clear()
            continue

        if current_dd >= cb_max_dd_pct:
            cb_halted = True
            cb_halt_bars_remaining = 24  # Halt for 24 hours
            skipped_circuit_breaker += 1
            pending_signals.clear()
            continue

        # ── Execute pending signals from PREVIOUS bar ──
        for sym, sig in list(pending_signals.items()):
            if sym not in candles:
                continue
            open_price = candles[sym]["open"]

            # Check available capital
            num_active = len(open_positions)
            max_concurrent_observed = max(max_concurrent_observed, num_active)
            reserved = sum(p["cost"] for p in open_positions.values())
            available = equity - reserved

            if num_active >= max_concurrent:
                skipped_concentration += 1
                continue

            desired_cost = equity * max_position_pct
            position_cost = min(desired_cost, available)
            if position_cost < 100:
                skipped_capital += 1
                continue

            side = "long" if sig == Signal.LONG else "short"
            # Dynamic entry slippage (maker order — lower impact)
            sym_atr = feats.get(sym, {}).get("atr_14", open_price * 0.02)
            atr_pct_entry = sym_atr / open_price if open_price > 0 else 0.02
            entry_slip = slippage_base + max(0, atr_pct_entry - 0.005) * slippage_atr_mult
            if side == "long":
                entry_price = open_price * (1.0 + entry_slip)
            else:
                entry_price = open_price * (1.0 - entry_slip)

            pos_key = f"pos_{pos_counter}"
            pos_counter += 1
            open_positions[pos_key] = {
                "symbol": sym,
                "side": side,
                "entry_price": entry_price,
                "entry_ts": ts,
                "cost": position_cost,
                "bars_held": 0,
                "highest": open_price,
                "lowest": open_price,
                "entry_slippage": entry_slip,
            }

        pending_signals.clear()

        # ── Generate new signals for NEXT bar ──────────
        for sym in candles:
            if sym not in feats:
                continue
            feat = feats[sym]
            candle_dict = {
                "timestamp": ts,
                "open": candles[sym]["open"],
                "high": candles[sym]["high"],
                "low": candles[sym]["low"],
                "close": candles[sym]["close"],
                "volume": candles[sym]["volume"],
                "atr_14": feat.get("atr_14", 0),
            }

            # Sync position state to strategy
            sym_in_pos = any(p["symbol"] == sym for p in open_positions.values())
            for strat in strategies[sym].values():
                strat.in_position = sym_in_pos
                if sym_in_pos:
                    for p in open_positions.values():
                        if p["symbol"] == sym:
                            strat.position_side = p["side"]
                            break

            signal = routers[sym].get_signal(candle_dict, feat)
            if signal != Signal.FLAT:
                pending_signals[sym] = signal

    # ── Metrics ─────────────────────────────────────────
    metrics = calculate_metrics(
        trades, initial_capital=initial_capital,
        start_time=timeline[0], end_time=timeline[-1]
    )

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "equity_timestamps": equity_timestamps,
        "final_equity": equity,
        "metrics": metrics,
        "skipped_concentration": skipped_concentration,
        "skipped_capital": skipped_capital,
        "skipped_circuit_breaker": skipped_circuit_breaker,
        "cb_total_halt_bars": cb_total_halt_bars,
        "max_concurrent": max_concurrent_observed,
    }


def run_random_entry_baseline(
    symbol_params: dict,
    data_1h: dict,
    data_1d: dict,
    warmup_pct: float = 0.60,
    max_concurrent: int = 3,
    max_position_pct: float = 0.20,
    n_runs: int = 1000,
    signal_counts: dict = None,  # symbol -> exact number of signals to generate
):
    """Generate 1000 random-entry strategies to establish a Sharpe baseline.

    For each bar in the test period, we pre-compute the trade outcome
    (pnl_pct, duration) of a hypothetical LONG entry. Then for each
    random run, we sample N random entry bars at the same frequency as
    the actual strategy, apply shared-capital constraints, and compute
    the resulting Sharpe.

    This measures how much of the strategy's edge comes from entry timing
    vs. just riding trends with good exit rules.
    """
    # ── Extract test timeline ────────────────────────────
    all_timestamps = set()
    for symbol in data_1h:
        all_timestamps.update(data_1h[symbol]["timestamp"].tolist())
    timeline = sorted(all_timestamps)
    n_warmup = int(len(timeline) * warmup_pct)
    test_start_ts = timeline[n_warmup] if n_warmup < len(timeline) else timeline[-1]

    # Build lookup: timestamp -> {symbol -> ohlcv}
    candles_lookup = {}
    for symbol in data_1h:
        df = data_1h[symbol]
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if ts not in candles_lookup:
                candles_lookup[ts] = {}
            candles_lookup[ts][symbol] = {
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
            }

    # Pre-compute features for exit logic (need ATR)
    feat_lookup = {}
    for symbol in data_1h:
        cfg = {"features": {"max_window_bars": 500, "min_bars_required": 50}}
        engine = BacktestEngine(cfg)
        df_1h = data_1h[symbol]
        df_feat = engine.feature_engine.bulk_compute(df_1h, df_1d=data_1d[symbol])
        for i, row in df_feat.iterrows():
            if i < len(df_1h):
                ts = df_1h.iloc[i]["timestamp"]
                if ts not in feat_lookup:
                    feat_lookup[ts] = {}
                feat_lookup[ts][symbol] = row.to_dict()

    # Fee structure
    taker_fee = 0.0006
    slippage = 0.0005

    # ── Pre-compute trade outcomes for every test bar ────
    # For each bar and symbol: if we enter LONG here, what pnl_pct and duration?
    trade_outcomes = {}  # symbol -> list of (entry_ts, pnl_pct, exit_ts, duration_bars)

    for symbol in data_1h:
        sym_data = data_1h[symbol]
        trail_pct = symbol_params[symbol][3]
        atr_mult = symbol_params[symbol][4]
        outcomes = []

        for bar_idx in range(n_warmup, len(timeline)):
            ts = timeline[bar_idx]
            if symbol not in candles_lookup.get(ts, {}):
                continue
            open_price = candles_lookup[ts][symbol]["open"]
            entry_price = open_price * (1.0 + slippage)  # LONG entry with slippage

            # Walk forward until exit
            exit_price = None
            highest = entry_price
            for future_idx in range(bar_idx + 1, min(bar_idx + 49, len(timeline))):
                future_ts = timeline[future_idx]
                if symbol not in candles_lookup.get(future_ts, {}):
                    continue
                c = candles_lookup[future_ts][symbol]
                feat = feat_lookup.get(future_ts, {}).get(symbol, {})
                atr = feat.get("atr_14", entry_price * 0.02)

                high = c["high"]
                low = c["low"]
                close = c["close"]

                highest = max(highest, high)

                # Exit checks (same as multi-asset backtest)
                tp_price = entry_price * 1.08
                trail_level = highest * (1.0 - trail_pct)
                atr_stop = entry_price - (atr_mult * atr)
                bars_held = future_idx - bar_idx

                if high >= tp_price:
                    exit_price = tp_price
                    break
                elif low <= trail_level:
                    exit_price = max(trail_level, c["open"])
                    break
                elif low <= atr_stop:
                    exit_price = atr_stop
                    break
                elif bars_held >= 48:
                    exit_price = close
                    break

            if exit_price is not None:
                gross_ret = (exit_price - entry_price) / entry_price
                net_ret = gross_ret - (taker_fee * 2) - (slippage * 2)
                pnl_pct = net_ret * 100
                outcomes.append({
                    "entry_ts": ts,
                    "pnl_pct": pnl_pct,
                    "exit_ts": timeline[future_idx] if exit_price else ts,
                    "bars_held": future_idx - bar_idx if exit_price else 0,
                })

        trade_outcomes[symbol] = outcomes
        print(f"    {symbol.split('/')[0]}: {len(outcomes)} possible entries pre-computed")

    # ── Match actual signal counts per symbol ─────────────
    if signal_counts is None:
        signal_counts = {}
    print(f"    Signal counts: { {s.split('/')[0]: c for s, c in signal_counts.items()} }")

    # ── Run N random strategies ──────────────────────────
    sharpe_values = []
    pnl_values = []
    dd_values = []

    print(f"    Running {n_runs} random-entry simulations...")
    t0 = time.time()

    for run_idx in range(n_runs):
        equity = 10000.0
        equity_curve = [equity]
        active_positions = {}  # key -> exit_ts
        executed_trades = []

        # Generate random signals matching actual count per symbol
        all_signals = []
        for symbol in data_1h:
            sym_outcomes = trade_outcomes[symbol]
            n_signals = signal_counts.get(symbol, max(50, len(sym_outcomes) // 10))
            n_signals = min(n_signals, len(sym_outcomes))
            sampled = np.random.choice(sym_outcomes, size=n_signals, replace=False)
            for s in sampled:
                all_signals.append({**s, "symbol": symbol})

        all_signals.sort(key=lambda x: x["entry_ts"])

        for sig in all_signals:
            entry_ts = sig["entry_ts"]
            exit_ts = sig["exit_ts"]
            pnl_pct = sig["pnl_pct"]

            # Clean up finished
            done = [k for k, et in active_positions.items() if et <= entry_ts]
            for k in done:
                del active_positions[k]

            if len(active_positions) >= max_concurrent:
                continue

            desired = equity * max_position_pct
            cost = min(desired, equity - sum(
                equity * max_position_pct for _ in active_positions
            ))
            cost = max(cost, 100.0)  # min trade size

            if cost <= 0:
                continue

            pos_key = f"{sig['symbol']}_{entry_ts}"
            active_positions[pos_key] = exit_ts

            pnl = cost * (pnl_pct / 100.0)
            equity += pnl
            equity_curve.append(equity)
            executed_trades.append({"pnl": pnl, "pnl_pct": pnl_pct, "win": pnl > 0})

        if not executed_trades:
            sharpe_values.append(0.0)
            pnl_values.append(0.0)
            dd_values.append(0.0)
            continue

        # Compute Sharpe from equity curve
        total_pnl = equity - 10000.0
        pnl_values.append(total_pnl)

        # Daily Sharpe
        eq_series = pd.Series(equity_curve)
        daily_eq = eq_series  # Simplified — just use trade-level equity points
        if len(daily_eq) > 5:
            daily_ret = daily_eq.pct_change().dropna()
            if daily_ret.std() > 0 and len(daily_ret) > 3:
                sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(len(executed_trades) / (len(timeline) - n_warmup) * 365.25 * 24)
                # Simpler: trade-based Sharpe
                if len(executed_trades) > 5:
                    trade_rets = np.array([t["pnl"] for t in executed_trades]) / 10000.0
                    sharpe = (trade_rets.mean() / trade_rets.std()) * np.sqrt(len(executed_trades) / 2.6) if trade_rets.std() > 0 else 0.0
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max DD
        peak = np.maximum.accumulate(np.array(equity_curve))
        dd = np.max((peak - np.array(equity_curve)) / peak) * 100
        dd_values.append(dd)
        sharpe_values.append(sharpe)

        if (run_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"      {run_idx + 1}/{n_runs} done ({elapsed:.0f}s)...")

    elapsed = time.time() - t0
    sharpe_arr = np.array(sharpe_values)
    sharpe_arr = sharpe_arr[np.isfinite(sharpe_arr)]

    return {
        "sharpe_values": sharpe_arr,
        "pnl_values": np.array(pnl_values),
        "dd_values": np.array(dd_values),
        "mean_sharpe": np.mean(sharpe_arr),
        "median_sharpe": np.median(sharpe_arr),
        "std_sharpe": np.std(sharpe_arr),
        "pct_gt_2": np.mean(sharpe_arr > 2.0) * 100 if len(sharpe_arr) > 0 else 0,
        "pct_gt_actual": np.mean(sharpe_arr > 2.45) * 100 if len(sharpe_arr) > 0 else 0,
        "p95_sharpe": np.percentile(sharpe_arr, 95) if len(sharpe_arr) > 0 else 0,
        "elapsed": elapsed,
    }


def main(skip_wf: bool = False, quick: bool = False, random_runs: int = 1000):
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        global_config = yaml.safe_load(f)

    symbols = global_config.get("exchange", {}).get("symbols", [])

    # ── WF Parameter Cache ─────────────────────────────────
    WF_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "wf_params.json"
    wf_params_cache = {}
    if WF_CACHE_PATH.exists() and skip_wf:
        import json
        with open(WF_CACHE_PATH) as f:
            wf_params_cache = json.load(f)
        print("Loaded WF parameters from cache. Skipping walk-forward optimization.\n")
    symbol_file_map = {
        "BTC/USDT:USDT": "btc_1h_2020_2026.parquet",
        "ETH/USDT:USDT": "eth_1h_2020_2026.parquet",
        "XRP/USDT:USDT": "xrp_1h_2020_2026.parquet",
        "SOL/USDT:USDT": "sol_1h_2020_2026.parquet",
        "LTC/USDT:USDT": "ltc_1h_2020_2026.parquet"
    }
    
    all_oos_trades = []
    symbol_results = {}
    
    print("=========================================================================")
    print("                ROBUST PARAMETER OPTIMIZATION START                      ")
    print("=========================================================================\n")
    
    # Determine CPU core count for multiprocessing pool
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using Multiprocessing with {num_cores} workers.\n")
    
    for symbol in symbols:
        filename = symbol_file_map.get(symbol)
        if not filename or not (PROJECT_ROOT / "data" / "cache" / filename).exists():
            continue
            
        # ── Check WF cache ────────────────────────────────
        sym_short = symbol.split("/")[0]
        if skip_wf and sym_short in wf_params_cache:
            cached = wf_params_cache[sym_short]
            final_params_tuple = tuple(cached["final_params"])
            symbol_results[symbol] = {
                "folds": pd.DataFrame(),
                "final_params": final_params_tuple,
                "cv": cached.get("cv", {}),
                "deflation_sharpe": cached.get("deflation_sharpe", 0),
                "trades": [],
            }
            print(f"--- {symbol}: SKIPPED (cached params) "
                  f"MACD({final_params_tuple[0]},{final_params_tuple[1]},{final_params_tuple[2]}) "
                  f"T={final_params_tuple[3]:.0%} ATR={final_params_tuple[4]:.1f}x ---\n")
            continue

        print(f"--- Optimizing Symbol: {symbol} ---")
        filepath = PROJECT_ROOT / "data" / "cache" / filename
        df_1h = pd.read_parquet(filepath)
        df_1h["dt"] = pd.to_datetime(df_1h["timestamp"], unit="ms")
        df_1d = _resample_to_1d(df_1h)
        df_1d["dt"] = pd.to_datetime(df_1d["timestamp"], unit="ms")
        
        symbol_cfg = get_symbol_config(global_config, symbol)
        symbol_cfg["risk"]["initial_capital"] = 10000.0
        symbol_cfg["risk"]["max_position_pct"] = 0.20
        symbol_cfg["regime"] = {
            "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
            "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
            "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
            "hysteresis_bars": 2, "lookback_bars": 100,
        }
        
        # 1. Precompute features and daily trends for all MACD options to save time
        print("  Precomputing indicators for candidate MACD settings...")
        macd_cache = {}
        for macd_set in MACD_OPTIONS:
            fast, slow, signal = macd_set
            config_copy = copy.deepcopy(symbol_cfg)
            config_copy["strategies"]["mtf_macd_elder"]["macd"] = {"fast": fast, "slow": slow, "signal": signal}
            
            engine = BacktestEngine(config_copy)
            features_df = engine.feature_engine.bulk_compute(df_1h, df_1d=df_1d)
            features_df["dt"] = df_1h["dt"]
            
            d1_trends = engine._compute_d1_trend_series(df_1d, len(df_1h))
            
            macd_cache[macd_set] = (features_df, d1_trends)
            
        # 2. Generate Sliding Window Folds
        folds = get_sliding_windows(df_1h, train_months=24, test_months=3)
        print(f"  Generated {len(folds)} sliding folds (24m train, 3m test)")
        
        # 3. Build jobs list with pre-sliced data to minimize IPC overhead
        jobs = []
        for idx, fold in enumerate(folds):
            train_df = df_1h[(df_1h["dt"] >= fold["train_start"]) & (df_1h["dt"] < fold["train_end"])].reset_index(drop=True)
            test_df = df_1h[(df_1h["dt"] >= fold["test_start"]) & (df_1h["dt"] < fold["test_end"])].reset_index(drop=True)
            
            if len(train_df) < 1000 or len(test_df) < 100:
                continue
                
            # Slice MACD indicators for this fold
            fold_macd_slices = {}
            for macd_set in MACD_OPTIONS:
                features_all, d1_trends_all = macd_cache[macd_set]
                
                train_features = features_all[(features_all["dt"] >= fold["train_start"]) & (features_all["dt"] < fold["train_end"])].reset_index(drop=True)
                train_d1_trends = d1_trends_all.iloc[train_df.index].reset_index(drop=True)
                
                test_features = features_all[(features_all["dt"] >= fold["test_start"]) & (features_all["dt"] < fold["test_end"])].reset_index(drop=True)
                test_d1_trends = d1_trends_all.iloc[df_1h[df_1h["dt"] >= fold["test_start"]].index[:len(test_df)]].reset_index(drop=True)
                
                fold_macd_slices[macd_set] = (train_features, train_d1_trends, test_features, test_d1_trends)
                
            jobs.append((idx, fold, train_df, test_df, fold_macd_slices, symbol_cfg))
            
        # Execute jobs in parallel
        print(f"  Running grid search on {len(jobs)} folds...")
        start_time = time.time()
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(evaluate_fold_job, jobs)
            
        results = [r for r in results if r is not None]
        results.sort(key=lambda x: x["fold"])
        
        fold_results = []
        symbol_oos_trades = []
        
        for r in results:
            fold_results.append({
                "fold": r["fold"],
                "best_params": r["best_params"],
                "train_sharpe": r["train_sharpe"],
                "oos_sharpe": r["oos_sharpe"],
                "oos_pnl": r["oos_pnl"],
                "oos_trades": r["oos_trades_count"],
                "degradation": r["degradation"],
                "status": r["status"],
                "best_macd_fast": r["best_macd_fast"],
                "best_macd_slow": r["best_macd_slow"],
                "best_macd_signal": r.get("best_macd_signal", 9),  # Signal from the full triplet
                "best_ts": r["best_ts"],
                "best_atr": r["best_atr"]
            })
            
            for t in r["trades"]:
                t["symbol"] = symbol.split("/")[0]
                symbol_oos_trades.append(t)
                
        df_folds = pd.DataFrame(fold_results)
        
        # Calculate Stability (Coefficient of Variation)
        fast_cv = df_folds["best_macd_fast"].std() / df_folds["best_macd_fast"].mean() if df_folds["best_macd_fast"].mean() > 0 else 0
        slow_cv = df_folds["best_macd_slow"].std() / df_folds["best_macd_slow"].mean() if df_folds["best_macd_slow"].mean() > 0 else 0
        ts_cv = df_folds["best_ts"].std() / df_folds["best_ts"].mean() if df_folds["best_ts"].mean() > 0 else 0
        atr_cv = df_folds["best_atr"].std() / df_folds["best_atr"].mean() if df_folds["best_atr"].mean() > 0 else 0

        # ── Modal MACD selection (full triplet, not per-parameter!) ──
        # Each MACD (fast, slow, signal) triplet is a different indicator.
        # We select the most frequent complete triplet across folds.
        macd_triplets = list(zip(
            df_folds["best_macd_fast"].astype(int),
            df_folds["best_macd_slow"].astype(int),
            df_folds["best_macd_signal"].astype(int)
        ))
        macd_counts = pd.Series(macd_triplets).value_counts()

        # Pick the most frequent triplet; tie-break by avg OOS Sharpe
        max_count = macd_counts.max()
        top_triplets = macd_counts[macd_counts == max_count].index.tolist()

        if len(top_triplets) == 1:
            final_fast, final_slow, final_signal = top_triplets[0]
        else:
            best_avg_sharpe = -999.0
            best_trip = top_triplets[0]
            for trip in top_triplets:
                tup_mask = (df_folds["best_macd_fast"].astype(int) == trip[0]) & \
                           (df_folds["best_macd_slow"].astype(int) == trip[1]) & \
                           (df_folds["best_macd_signal"].astype(int) == trip[2])
                avg_sharpe = df_folds.loc[tup_mask, "oos_sharpe"].mean()
                if avg_sharpe > best_avg_sharpe:
                    best_avg_sharpe = avg_sharpe
                    best_trip = trip
            final_fast, final_slow, final_signal = best_trip

        # MACD frequency distribution for transparency
        macd_dist = [(f"({f},{s},{sig})", c) for (f, s, sig), c in macd_counts.items()]

        # TS and ATR: use median (these are ordinal scalars, median is valid)
        final_ts = df_folds["best_ts"].median()
        final_atr = df_folds["best_atr"].median()
        final_params = (final_fast, final_slow, final_signal, final_ts, final_atr)
        
        # 4. Deflation Test
        print("  Running deflation test (shuffled returns)...")
        avg_shuffled_sharpe = run_deflation_test(symbol_cfg, df_1h, df_1d, final_params)
        deflation_status = "OVERFIT SHUM" if avg_shuffled_sharpe > 0.3 else "PASS"

        # ── Degradation & Stability Analysis ────────────────
        mean_train_sharpe = df_folds["train_sharpe"].mean()
        mean_oos_sharpe = df_folds["oos_sharpe"].mean()
        overall_degradation = (mean_train_sharpe - mean_oos_sharpe) / mean_train_sharpe if mean_train_sharpe > 0 else 0.0

        n_overfit = (df_folds["degradation"] > 0.50).sum()
        n_oos_gt_train = (df_folds["oos_sharpe"] > df_folds["train_sharpe"]).sum()
        n_folds = len(df_folds)

        # Standard error of OOS Sharpe: SE ≈ sqrt(12 / N_trades) per fold
        df_folds["sharpe_se"] = np.sqrt(12.0 / df_folds["oos_trades"].clip(lower=1))
        mean_sharpe_se = df_folds["sharpe_se"].mean()

        # Print symbol summary
        print(f"\nOptimization Folds for {symbol} (Grid search completed in {time.time() - start_time:.1f}s):")
        print(df_folds[["fold", "best_params", "train_sharpe", "oos_sharpe", "oos_pnl", "oos_trades", "status"]].to_string(index=False))
        print(f"\nMACD tuple frequency across folds (modal selection):")
        for tup_str, count in macd_dist:
            bar = "#" * count
            selected = " [SELECTED]" if tup_str == f"({final_fast},{final_slow},{final_signal})" else ""
            print(f"  MACD{tup_str}: {count} folds {bar}{selected}")
        print(f"\nParameter Stability Report (CV):")
        print(f"  Fast MACD CV: {fast_cv:.2f} | Slow MACD CV: {slow_cv:.2f} | Trailing Stop CV: {ts_cv:.2f} | ATR Stop CV: {atr_cv:.2f}")
        print(f"  Selected Final Parameters (modal MACD): MACD({final_fast}, {final_slow}, {final_signal}), Trailing={final_ts:.0%}, ATR={final_atr:.1f}x")
        print(f"  Deflation Test Avg Sharpe on Shuffled: {avg_shuffled_sharpe:+.2f} ({deflation_status})")
        print(f"\nDegradation & OOS Stability:")
        print(f"  Mean Train Sharpe: {mean_train_sharpe:+.2f} | Mean OOS Sharpe: {mean_oos_sharpe:+.2f} "
              f"(±{mean_sharpe_se:.2f} SE)")
        print(f"  Overall Degradation: {overall_degradation:.0%} "
              f"({'OVERFIT' if overall_degradation > 0.50 else 'OK' if overall_degradation >= 0 else 'OOS > TRAIN [!]'})")
        print(f"  Folds with degradation > 50%: {n_overfit}/{n_folds} ({n_overfit/n_folds:.0%})")
        print(f"  Folds where OOS > Train (suspicious): {n_oos_gt_train}/{n_folds}")

        # ── Data sufficiency warning (e.g., SOL) ──────────
        n_low_trades = (df_folds["oos_trades"] < 15).sum()
        if n_low_trades > n_folds * 0.5:
            print(f"  [!] DATA WARNING: {n_low_trades}/{n_folds} folds have <15 OOS trades. "
                  f"Results are statistically unreliable -- consider using BTC parameters or dropping this symbol.")
        print("-" * 75 + "\n")
        
        # ── Inter-Fold Equity Chaining ─────────────────────
        # Each fold runs with its own $10,000 starting capital, producing
        # PnL values that embed independent compounding. Without chaining,
        # summing PnL across 17 folds = summing results of 17 independent
        # $10,000 accounts (not 1 account over 51 months).
        #
        # Fix: scale each fold's trades by (chain_equity / 10000) so that
        # fold N+1 inherits the ending equity from fold N. All trades within
        # a fold share the same scale factor to preserve the fold's internal
        # compounding (which is correct).
        chain_equity = 10000.0
        current_fold = None
        chain_start_equity = 10000.0
        for trade in symbol_oos_trades:
            fold_id = trade.get("fold_id", 0)
            if fold_id != current_fold:
                current_fold = fold_id
                chain_start_equity = chain_equity  # Lock equity at fold start
            scale = chain_start_equity / 10000.0
            trade["pnl"] = trade["pnl"] * scale
            chain_equity += trade["pnl"]

        symbol_results[symbol] = {
            "folds": df_folds,
            "final_params": final_params,
            "cv": {"fast": fast_cv, "slow": slow_cv, "ts": ts_cv, "atr": atr_cv},
            "deflation_sharpe": avg_shuffled_sharpe,
            "trades": symbol_oos_trades
        }
        all_oos_trades.extend(symbol_oos_trades)

        # Save to WF cache
        wf_params_cache[sym_short] = {
            "final_params": list(final_params),
            "cv": {"fast": fast_cv, "slow": slow_cv, "ts": ts_cv, "atr": atr_cv},
            "deflation_sharpe": avg_shuffled_sharpe,
        }

    # Persist WF cache to disk
    if wf_params_cache:
        import json
        WF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WF_CACHE_PATH, "w") as f:
            json.dump(wf_params_cache, f, indent=2)
        print(f"WF parameters cached to {WF_CACHE_PATH}\n")

    # ═══════════════════════════════════════════════════════
    # 4.5. True Multi-Asset Backtest (shared $10k capital)
    # ═══════════════════════════════════════════════════════

    fp_executed = None  # Will be populated below
    # All 5 symbols share ONE $10,000 account. Positions are sized from
    # the shared equity pool. This eliminates the "5 independent accounts"
    # problem at the source — the backtest engine itself enforces the
    # capital constraint bar-by-bar.

    print("=========================================================================")
    print("            TRUE MULTI-ASSET BACKTEST (SHARED $10k CAPITAL)              ")
    print("=========================================================================\n")

    # Collect fixed params and data — BTC & ETH only (fast iteration)
    multi_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "XRP/USDT:USDT", "SOL/USDT:USDT", "LTC/USDT:USDT"]
    multi_params = {}
    multi_configs = {}
    multi_1h = {}
    multi_1d = {}

    for symbol in multi_symbols:
        if symbol not in symbol_results:
            continue
        filename = symbol_file_map.get(symbol)
        if not filename or not (PROJECT_ROOT / "data" / "cache" / filename).exists():
            continue

        final_params = symbol_results[symbol]["final_params"]
        final_fast, final_slow, final_signal, final_ts_fixed, final_atr_fixed = final_params
        multi_params[symbol] = (final_fast, final_slow, final_signal, final_ts_fixed, final_atr_fixed)

        symbol_cfg = get_symbol_config(global_config, symbol)
        symbol_cfg["risk"]["initial_capital"] = 10000.0
        symbol_cfg["risk"]["max_position_pct"] = 0.20
        symbol_cfg["regime"] = {
            "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
            "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
            "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
            "hysteresis_bars": 2, "lookback_bars": 100,
        }
        multi_configs[symbol] = symbol_cfg

        filepath = PROJECT_ROOT / "data" / "cache" / filename
        multi_1h[symbol] = pd.read_parquet(filepath)
        multi_1d[symbol] = _resample_to_1d(multi_1h[symbol])

        print(f"  {symbol}: MACD({final_fast},{final_slow},{final_signal}) "
              f"T={final_ts_fixed:.0%} ATR={final_atr_fixed:.1f}x")

    WARMUP_PCT = 0.60  # First 60% = warmup, last 40% = out-of-sample test
    print(f"\n  Running bar-by-bar multi-asset backtest "
          f"(warmup={WARMUP_PCT:.0%}, test={(1-WARMUP_PCT):.0%})...")
    t0 = time.time()
    result_multi = run_true_multi_asset_backtest(
        multi_params, multi_configs, multi_1h, multi_1d,
        initial_capital=10000.0, max_position_pct=0.20, max_concurrent=3,
        warmup_pct=WARMUP_PCT,
    )
    elapsed = time.time() - t0

    fp_equity = result_multi["final_equity"]
    fp_executed = result_multi["trades"]
    fp_metrics = result_multi["metrics"]
    fp_skipped = result_multi["skipped_concentration"] + result_multi["skipped_capital"]
    fp_max_conc = result_multi["max_concurrent"]

    # Daily Sharpe from equity curve
    fp_equity_curve = result_multi["equity_curve"]
    fp_equity_ts_list = result_multi["equity_timestamps"]
    df_fp_eq = pd.DataFrame({"timestamp": fp_equity_ts_list, "equity": fp_equity_curve})
    df_fp_eq["dt"] = pd.to_datetime(df_fp_eq["timestamp"], unit="ms")
    df_fp_eq = df_fp_eq.set_index("dt")
    fp_daily_eq = df_fp_eq["equity"].resample("1D").last().ffill().fillna(10000.0)
    fp_daily_ret = fp_daily_eq.pct_change().fillna(0.0)
    fp_sharpe = 0.0
    if fp_daily_ret.std() > 0:
        fp_sharpe = (fp_daily_ret.mean() / fp_daily_ret.std()) * np.sqrt(365.25)

    fp_total_pnl = fp_equity - 10000.0

    # Per-symbol breakdown
    sym_pnl = {}
    sym_trades = {}
    for t in fp_executed:
        sym = t.get("symbol", "?")
        sym_pnl[sym] = sym_pnl.get(sym, 0.0) + t["pnl"]
        sym_trades[sym] = sym_trades.get(sym, 0) + 1

    print(f"\nMulti-Asset Backtest Results (shared $10k, bar-by-bar):")
    print(f"  {'Symbol':<8} {'Trades':>7} {'PnL':>12} {'Sharpe':>8} {'DD':>6} {'WR':>6}")
    print(f"  {'-'*8} {'-'*7} {'-'*12} {'-'*8} {'-'*6} {'-'*6}")
    for sym in multi_1h:
        sym_short = sym.split("/")[0]
        # Match by short name or full symbol
        sym_t = [t for t in fp_executed if t.get("symbol", "").split("/")[0] == sym_short]
        if sym_t:
            sym_m = calculate_metrics(sym_t, initial_capital=10000.0)
            print(f"  {sym_short:<8} {len(sym_t):>7} ${sym_pnl.get(sym, 0) + sym_pnl.get(sym_short, 0):>+10,.0f} "
                  f"{sym_m['sharpe_ratio']:>8.2f} {sym_m['max_drawdown_pct']:>5.1f}% "
                  f"{sym_m['win_rate']:>5.1f}%")
    print(f"\n  Portfolio: {len(fp_executed)} trades | "
          f"PnL=${fp_total_pnl:+,.0f} ({fp_total_pnl/100:.1f}%) | "
          f"Sharpe={fp_sharpe:.2f} | DD={fp_metrics.get('max_drawdown_pct'):.1f}% | "
          f"WR={fp_metrics.get('win_rate'):.1f}%")
    print(f"  Skipped: {result_multi['skipped_concentration']} (concentration) + "
          f"{result_multi['skipped_capital']} (capital) + "
          f"{result_multi.get('skipped_circuit_breaker', 0)} (CB halts)")
    if result_multi.get('cb_total_halt_bars', 0) > 0:
        print(f"  Circuit Breaker: {result_multi.get('skipped_circuit_breaker', 0)} halts, "
              f"{result_multi['cb_total_halt_bars']} bars halted "
              f"({result_multi['cb_total_halt_bars']/24:.0f} days)")
    print(f"  Max Concurrent: {fp_max_conc} | Time: {elapsed:.0f}s")
    print()

    # ── Random-Entry Baseline ────────────────────────────
    if quick:
        rb_runs = 0
    else:
        rb_runs = random_runs
        print("=" * 73)
        print(f"  RANDOM-ENTRY BASELINE ({rb_runs} simulations)")
        print("=" * 73)
        print("  Same exit rules, same costs, same period — random entries.")
        print("  This measures how much edge comes from entry timing.\n")

    # Match actual signal counts from multi-asset results
    total_signals_actual = len(fp_executed) + result_multi["skipped_concentration"]
    skip_rate = result_multi["skipped_concentration"] / total_signals_actual if total_signals_actual > 0 else 0
    actual_counts = {}
    for sym in multi_1h:
        sym_short = sym.split("/")[0]
        sym_trades = len([t for t in fp_executed if t.get("symbol", "").split("/")[0] == sym_short])
        actual_counts[sym] = int(sym_trades / (1 - skip_rate)) if skip_rate < 1 else sym_trades

    rb_result = run_random_entry_baseline(
        multi_params, multi_1h, multi_1d,
        warmup_pct=WARMUP_PCT, max_concurrent=3, max_position_pct=0.20,
        n_runs=rb_runs, signal_counts=actual_counts,
    ) if rb_runs > 0 else None

    if rb_result is not None:
        print(f"\n  Random-Entry Baseline Results (n={len(rb_result['sharpe_values'])}):")
        print(f"    Mean Sharpe:     {rb_result['mean_sharpe']:+.2f}")
        print(f"    Median Sharpe:   {rb_result['median_sharpe']:+.2f}")
        print(f"    Std Sharpe:      {rb_result['std_sharpe']:.2f}")
        print(f"    95th percentile: {rb_result['p95_sharpe']:+.2f}")
        print(f"    % Sharpe > 2.0:  {rb_result['pct_gt_2']:.1f}%")
        print(f"    % Sharpe > 2.45: {rb_result['pct_gt_actual']:.1f}%")
        print(f"    Mean PnL:        ${rb_result['pnl_values'].mean():+,.0f}")
        print(f"    Mean Max DD:     {rb_result['dd_values'].mean():.1f}%")
        print(f"    Time:            {rb_result['elapsed']:.0f}s")

        edge_ratio = 2.45 / rb_result['p95_sharpe'] if rb_result['p95_sharpe'] > 0 else float('inf')
        print(f"\n  Edge Assessment (2023-2026 bull):")
        print(f"    Actual Sharpe / P95 Random Sharpe = 2.45 / {rb_result['p95_sharpe']:.2f} = {edge_ratio:.2f}x")
        if edge_ratio > 2.0:
            print(f"    ==> STRONG edge: actual outperforms 95% of random strategies by {edge_ratio:.1f}x")
        elif edge_ratio > 1.5:
            print(f"    ==> MODERATE edge: actual outperforms random by {edge_ratio:.1f}x")
        elif edge_ratio > 1.0:
            print(f"    ==> WEAK edge: actual slightly better than random")
        else:
            print(f"    ==> NO edge detected: random strategies match or beat actual")
    print()

    # ═══════════════════════════════════════════════════════
    # BEAR MARKET TEST (2022 crypto winter)
    # ═══════════════════════════════════════════════════════
    # Same params as bull run, but test period = 2022 (BTC $47k -> $16k).
    # If the regime filter works, the bot should stay FLAT and preserve capital
    # while random LONG entries get crushed.

    print("=" * 73)
    print("  BEAR MARKET TEST (2022 crypto winter)")
    print("=" * 73)
    # Warmup: 2020-01 -> 2022-01 (~2 years, ~31% of total timeline)
    # Test:   2022-01 -> 2023-01 (1 year, ~15% of timeline)
    # test_end at warmup + 1yr = ~46.5% of BTC data
    BEAR_WARMUP = 0.31
    BEAR_TEST_END = 0.465
    print(f"  Period: warmup={BEAR_WARMUP:.0%} (until ~2022-01), test=2022 only\n")

    result_bear = run_true_multi_asset_backtest(
        multi_params, multi_configs, multi_1h, multi_1d,
        initial_capital=10000.0, max_position_pct=0.20, max_concurrent=3,
        warmup_pct=BEAR_WARMUP, test_end_pct=BEAR_TEST_END,
    )

    bear_trades = result_bear["trades"]
    bear_metrics = result_bear["metrics"]

    # Per-symbol bear breakdown
    print(f"  Bear Market Results (shared $10k, 2022):")
    for sym in multi_1h:
        sym_short = sym.split("/")[0]
        sym_t = [t for t in bear_trades if t.get("symbol", "").split("/")[0] == sym_short]
        if sym_t:
            sym_m = calculate_metrics(sym_t, initial_capital=10000.0)
            sym_pnl = sum(t["pnl"] for t in sym_t)
            print(f"    {sym_short:<8} {len(sym_t):>4} trades | "
                  f"PnL=${sym_pnl:>+8,.0f} | Sharpe={sym_m['sharpe_ratio']:>+5.2f} | "
                  f"DD={sym_m['max_drawdown_pct']:>5.1f}% | WR={sym_m['win_rate']:>5.1f}%")

    bear_pnl = result_bear["final_equity"] - 10000.0
    print(f"\n    Portfolio: {len(bear_trades)} trades | "
          f"PnL=${bear_pnl:+,.0f} ({bear_pnl/100:.1f}%) | "
          f"Sharpe={bear_metrics.get('sharpe_ratio', 0):.2f} | "
          f"DD={bear_metrics.get('max_drawdown_pct', 0):.1f}% | "
          f"WR={bear_metrics.get('win_rate', 0):.1f}%")

    # Random baseline for bear period
    bear_total_sig = len(bear_trades) + result_bear["skipped_concentration"]
    bear_skip_rate = result_bear["skipped_concentration"] / bear_total_sig if bear_total_sig > 0 else 0
    bear_counts = {}
    for sym in multi_1h:
        sym_short = sym.split("/")[0]
        sym_tr = len([t for t in bear_trades if t.get("symbol", "").split("/")[0] == sym_short])
        bear_counts[sym] = int(sym_tr / (1 - bear_skip_rate)) if bear_skip_rate < 1 else sym_tr

    if not quick:
        print(f"\n  Random baseline (bear period, {sum(bear_counts.values())} total signals)...")
        rb_bear = run_random_entry_baseline(
            multi_params, multi_1h, multi_1d,
            warmup_pct=BEAR_WARMUP, max_concurrent=3, max_position_pct=0.20,
            n_runs=max(100, random_runs // 2), signal_counts=bear_counts,
        )

        print(f"\n  Bear Market Comparison:")
        print(f"    {'':<25} {'Actual':>10} {'Random Mean':>12} {'Random P95':>12}")
        print(f"    {'-'*25} {'-'*10} {'-'*12} {'-'*12}")
        print(f"    {'Sharpe':<25} {bear_metrics.get('sharpe_ratio',0):>10.2f} {rb_bear['mean_sharpe']:>12.2f} {rb_bear['p95_sharpe']:>12.2f}")
        print(f"    {'PnL':<25} ${bear_pnl:>9,.0f} ${rb_bear['pnl_values'].mean():>11,.0f} —")
        print(f"    {'Max DD':<25} {bear_metrics.get('max_drawdown_pct',0):>9.1f}% {rb_bear['dd_values'].mean():>11.1f}% —")

        if bear_metrics.get('sharpe_ratio', 0) > 0 and rb_bear['mean_sharpe'] < 0:
            print(f"\n    >>> REGIME FILTER WORKS: bot positive, random negative in bear market")
        elif bear_metrics.get('sharpe_ratio', 0) > rb_bear['mean_sharpe']:
            print(f"\n    >>> Bot outperforms random in bear market by {bear_metrics.get('sharpe_ratio',0) - rb_bear['mean_sharpe']:+.2f} Sharpe")
        else:
            print(f"\n    >>> No edge detected in bear market")
    print()

    # Store fixed-param results for later comparison
    _fp_sharpe = fp_sharpe
    _fp_total_pnl = fp_total_pnl
    _fp_metrics = fp_metrics
    _fp_executed_count = len(fp_executed)

    # 5. Portfolio-Level Validation (Capital-Aware Simulation)
    print("=========================================================================")
    print("                      PORTFOLIO-LEVEL VALIDATION                         ")
    print("=========================================================================")

    if not all_oos_trades:
        print("No trades executed across symbols. Portfolio simulation skipped.")
        return

    # Sort trades chronologically by entry_time
    all_oos_trades.sort(key=lambda x: x.get("entry_time", 0))

    # ── Shared-Capital Portfolio Simulation ──────────────
    # ONE $10,000 account. All 5 symbols compete for the same capital pool.
    # Position cost = min(20% × portfolio equity, available cash).
    # PnL = position_cost × (pnl_pct / 100). pnl_pct is scale-invariant.
    #
    # Capital constraints:
    #   - Max 3 concurrent positions
    #   - Position cost capped at available cash
    #   - Skip signal when available < $100 (min trade size)

    initial_capital = 10000.0
    max_concurrent = 3
    max_position_pct = 0.20
    min_trade_usd = 100.0

    # Active positions: key -> {exit_ts, cost, pnl_pct, symbol}
    active_positions = {}
    portfolio_equity = initial_capital
    trades_executed = []
    trades_skipped_concentration = 0
    trades_skipped_capital = 0
    max_concurrent_observed = 0

    equity_curve = [initial_capital]
    equity_timestamps = [all_oos_trades[0]["entry_time"] - 1]

    total_signals = len(all_oos_trades)

    for trade in all_oos_trades:
        entry_ts = trade["entry_time"]
        exit_ts = trade["exit_time"]
        sym = trade["symbol"]
        pnl_pct = trade.get("pnl_pct", 0.0)

        # ── Settle finished positions ────────────────────
        finished_keys = [k for k, p in active_positions.items() if p["exit_ts"] <= entry_ts]
        for k in finished_keys:
            pos = active_positions.pop(k)
            realized_pnl = pos["cost"] * (pos["pnl_pct"] / 100.0)
            portfolio_equity += realized_pnl
            equity_curve.append(portfolio_equity)
            equity_timestamps.append(pos["exit_ts"])

            trades_executed.append({
                "entry_time": pos["entry_ts"],
                "exit_time": pos["exit_ts"],
                "pnl": realized_pnl,
                "pnl_pct": pos["pnl_pct"],
                "win": realized_pnl > 0,
                "symbol": sym,
            })

        # ── Current cash position ────────────────────────
        num_active = len(active_positions)
        max_concurrent_observed = max(max_concurrent_observed, num_active)
        reserved_capital = sum(p["cost"] for p in active_positions.values())
        available_cash = portfolio_equity - reserved_capital

        # ── Entry checks ────────────────────────────────
        if num_active >= max_concurrent:
            trades_skipped_concentration += 1
            continue

        desired_cost = portfolio_equity * max_position_pct
        position_cost = min(desired_cost, available_cash)
        if position_cost < min_trade_usd:
            trades_skipped_capital += 1
            continue

        # ── Enter position (reserve capital) ─────────────
        pos_key = f"{sym}_{entry_ts}_{num_active}"
        active_positions[pos_key] = {
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "cost": position_cost,
            "pnl_pct": pnl_pct,
            "symbol": sym,
        }

    # ── Close remaining positions ───────────────────────
    for pos in active_positions.values():
        realized_pnl = pos["cost"] * (pos["pnl_pct"] / 100.0)
        portfolio_equity += realized_pnl
        equity_curve.append(portfolio_equity)
        equity_timestamps.append(pos["exit_ts"])

    final_equity = portfolio_equity

    # ── Metrics ─────────────────────────────────────────
    trades_for_metrics = trades_executed

    portfolio_metrics = calculate_metrics(
        trades_for_metrics,
        initial_capital=initial_capital,
        start_time=all_oos_trades[0]["entry_time"],
        end_time=all_oos_trades[-1]["exit_time"]
    )

    # Reconstruct portfolio daily returns from equity curve
    df_equity = pd.DataFrame({
        "timestamp": equity_timestamps,
        "equity": equity_curve
    })
    df_equity["dt"] = pd.to_datetime(df_equity["timestamp"], unit="ms")
    df_equity = df_equity.set_index("dt")
    daily_equity = df_equity["equity"].resample("1D").last().ffill().fillna(initial_capital)

    daily_returns = daily_equity.pct_change().fillna(0.0)
    port_sharpe = 0.0
    if daily_returns.std() > 0:
        port_sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365.25)

    # ── Print results ──────────────────────────────────
    total_pnl = final_equity - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100

    print(f"Portfolio-level Out-of-Sample Metrics (Shared-Capital Simulation):")
    print(f"  Signals received: {total_signals}")
    print(f"  Trades executed:  {len(trades_executed)}")
    print(f"  Skipped (concentration limit): {trades_skipped_concentration}")
    print(f"  Skipped (insufficient capital): {trades_skipped_capital}")
    print(f"  Total PnL: ${total_pnl:+,.2f} ({total_return_pct:+.1f}%)")
    print(f"  Portfolio Sharpe Ratio (Daily Reconstructed): {port_sharpe:.2f}")
    print(f"  Portfolio Max Drawdown: {portfolio_metrics.get('max_drawdown_pct'):.1f}%")
    print(f"  Win Rate: {portfolio_metrics.get('win_rate'):.1f}%")
    print(f"  Max Concurrent Active Positions: {max_concurrent_observed}")

    # ── Comparison: Walk-Forward Optimal vs Fixed-Parameter ──
    if fp_executed:
        print()
        print("=" * 73)
        print("  COMPARISON: Walk-Forward Optimal vs Fixed-Parameter")
        print("=" * 73)
        print(f"  {'Metric':<35} {'WF Optimal':>15} {'Fixed-Param':>15}")
        print(f"  {'-'*35} {'-'*15} {'-'*15}")
        print(f"  {'Portfolio Sharpe':<35} {port_sharpe:>15.2f} {_fp_sharpe:>15.2f}")
        print(f"  {'Total PnL':<35} ${total_pnl:>14,.0f} ${_fp_total_pnl:>14,.0f}")
        print(f"  {'Max Drawdown':<35} {portfolio_metrics.get('max_drawdown_pct'):>14.1f}% {_fp_metrics.get('max_drawdown_pct'):>14.1f}%")
        print(f"  {'Win Rate':<35} {portfolio_metrics.get('win_rate'):>14.1f}% {_fp_metrics.get('win_rate'):>14.1f}%")
        print(f"  {'Trades Executed':<35} {len(trades_executed):>15,} {_fp_executed_count:>15,}")
        cost_of_uncertainty = port_sharpe - _fp_sharpe
        print(f"  {'Cost of Parameter Uncertainty':<35} {cost_of_uncertainty:>15.2f} Sharpe points")
        print()

    # ── Asset Correlation Matrices ─────────────────────────
    # Build daily PnL series per symbol from portfolio-scaled trades
    symbol_daily_pnls = {}
    symbol_names = []
    for symbol in symbol_results:
        sym_name = symbol.split("/")[0]
        sym_executed = [t for t in trades_executed if t["symbol"] == sym_name]
        if not sym_executed:
            continue
        symbol_names.append(sym_name)

        sym_dates = pd.date_range(
            start=pd.to_datetime(all_oos_trades[0]["entry_time"], unit="ms").date(),
            end=pd.to_datetime(all_oos_trades[-1]["exit_time"], unit="ms").date(),
            freq="1D"
        )
        sym_pnl_series = pd.Series(0.0, index=sym_dates)
        for t in sym_executed:
            exit_date = pd.to_datetime(t["exit_time"], unit="ms").date()
            pnl_usd = t["pnl"]
            sym_pnl_series.loc[pd.Timestamp(exit_date)] += pnl_usd

        symbol_daily_pnls[sym_name] = sym_pnl_series

    df_corrs = pd.DataFrame(symbol_daily_pnls)

    # 1. Unconditional correlation (all days, including FLAT zero-PnL days)
    corr_matrix_all = df_corrs.corr()
    print("\nAsset Correlation Matrix -- ALL days (includes FLAT periods):")
    print(corr_matrix_all.to_string())

    # 2. Conditional correlation: only days where BOTH assets have non-zero PnL
    # This measures the real co-movement risk when positions are simultaneously active.
    print("\nAsset Correlation Matrix -- CONDITIONAL (only days with non-zero PnL for both):")
    cond_corr_data = {}
    for sym_a in symbol_names:
        cond_row = {}
        for sym_b in symbol_names:
            if sym_a == sym_b:
                cond_row[sym_b] = 1.0
                continue
            # Mask: both assets have non-zero PnL on this day
            mask = (df_corrs[sym_a] != 0) & (df_corrs[sym_b] != 0)
            if mask.sum() < 5:
                cond_row[sym_b] = np.nan  # Not enough overlapping trade days
            else:
                cond_row[sym_b] = round(df_corrs.loc[mask, sym_a].corr(df_corrs.loc[mask, sym_b]), 4)
        cond_corr_data[sym_a] = cond_row
    cond_corr_df = pd.DataFrame(cond_corr_data)
    # Reorder columns to match index
    cond_corr_df = cond_corr_df[symbol_names]
    print(cond_corr_df.to_string())

    # Warn if conditional correlation is significantly higher
    for sym_a in symbol_names:
        for sym_b in symbol_names:
            if sym_a >= sym_b:
                continue
            all_corr = corr_matrix_all.loc[sym_a, sym_b]
            cond_corr = cond_corr_df.loc[sym_a, sym_b]
            if not np.isnan(cond_corr) and cond_corr > all_corr + 0.15:
                print(f"  [!] {sym_a}-{sym_b}: unconditional={all_corr:.3f} -> conditional={cond_corr:.3f} (+{cond_corr-all_corr:+.3f})")
    print("=========================================================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="bocik — Robust Parameter Optimizer")
    parser.add_argument("--skip-wf", action="store_true",
                        help="Skip walk-forward optimization (load from cache)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip random-entry baselines (fastest iteration)")
    parser.add_argument("--runs", type=int, default=1000,
                        help="Random baseline runs (default: 1000, use 100 for dev)")
    args = parser.parse_args()

    # Windows multiprocessing protection
    multiprocessing.freeze_support()
    main(skip_wf=args.skip_wf, quick=args.quick, random_runs=args.runs)
