"""Meta-Labeling v2 -- Multi-Symbol (BTC+ETH), optimized with signal-specific features and parallel grid search.

Improvements over v1:
    - BTC + ETH symbols (2x training data)
    - Signal-specific features: bars_since_last_signal, regime encoding, volatility context, MACD state
    - 60/20/20 chronological split (train/val/test)
    - Parallel hyperparameter grid search (joblib if available, sequential fallback)
    - Configurable min_confidence via val set optimization

Usage:
    python research/meta_labeling_optimized.py              # full grid search
    python research/meta_labeling_optimized.py --quick       # fast grid (12 combos)
    python research/meta_labeling_optimized.py --symbols BTC # single symbol
"""

import os, sys, time, itertools, copy
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics
from strategies.mtf_macd import MTF_MACD_Elder
from strategies.meta_labeling import MetaLabeler

# ─── Config ─────────────────────────────────────────────────────

BASE_CONFIG = {
    "exchange": {"name": "bitget", "symbols": ["BTC/USDT"], "type": "spot",
                 "fees": {"taker": 0.0006, "maker": 0.0002, "slippage": 0.0005}},
    "risk": {"initial_capital": 10000, "max_position_pct": 0.20},
    "features": {"max_window_bars": 500, "min_bars_required": 50},
    "backtest": {"walk_forward_folds": 5, "min_train_fraction": 0.33,
                 "min_signal_exit_bars": 6, "cooldown_bars_after_loss": 2},
    "strategies": {"mtf_macd_elder": {
        "enabled": True, "macd": {"fast": 10, "slow": 20, "signal": 9},
        "exit": {"trailing_stop_pct": 0.02, "atr_stop_mult": 3.0, "min_hold_bars": 6},
        "elder_filter": {"require_volume_confirm": False, "allow_shorts": True}}},
    "meta_labeling": {"enabled": True, "model": "xgboost",
                      "training_samples": 500, "min_confidence": 0.55},
    "regime": {"enabled": True, "hysteresis_bars": 2,
               "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
               "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
               "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08}},
}

SYMBOLS_AVAILABLE = {
    "BTC": "btc_1h_2020_2026.parquet",
    "ETH": "eth_1h_2020_2026.parquet",
    "XRP": "xrp_1h_2020_2026.parquet",
    "SOL": "sol_1h_2020_2026.parquet",
    "LTC": "ltc_1h_2020_2026.parquet",
}


# ─── Signal-specific feature engineering ────────────────────────

def add_signal_features(trades: list[dict]) -> list[dict]:
    """Add signal-specific features to each trade's features_at_signal dict.

    These features capture the context of the signal itself, not just the market:
        - bars_since_last_signal: time since previous signal (hours)
        - bars_since_last_trade: time since previous trade (hours)
        - regime_is_trending / regime_is_ranging: one-hot regime encoding
        - atr_pct_of_price: ATR relative to current price
        - macd_hist_sign: direction of MACD histogram
        - bb_position: where price is within Bollinger Bands (0=lower, 1=upper)
        - adx_strength: ADX value normalized

    Modifies trades IN PLACE for efficiency.
    """
    if not trades:
        return trades

    # Sort by entry time for correct "since last" computation
    sorted_trades = sorted(trades, key=lambda t: t.get("entry_time", 0))

    for i, t in enumerate(sorted_trades):
        feats = t.get("features_at_signal")
        if not feats or not isinstance(feats, dict):
            continue

        # --- Temporal spacing features ---
        if i > 0:
            prev_entry = sorted_trades[i - 1].get("entry_time", 0)
            curr_entry = t.get("entry_time", 0)
            delta_hours = (curr_entry - prev_entry) / 3_600_000 if curr_entry > prev_entry else 999
            feats["bars_since_last_signal"] = min(delta_hours, 999)
        else:
            feats["bars_since_last_signal"] = 999

        # Same for exit_time (last trade close to this entry)
        if i > 0:
            prev_exit = sorted_trades[i - 1].get("exit_time", 0)
            curr_entry = t.get("entry_time", 0)
            delta_hours_exit = (curr_entry - prev_exit) / 3_600_000 if curr_entry > prev_exit else 999
            feats["hours_since_last_trade"] = min(delta_hours_exit, 999)
        else:
            feats["hours_since_last_trade"] = 999

        # --- Regime context ---
        regime = t.get("regime", "unknown")
        feats["regime_is_trending"] = 1.0 if regime == "trending" else 0.0
        feats["regime_is_ranging"] = 1.0 if regime == "ranging" else 0.0
        feats["regime_is_volatile"] = 1.0 if regime == "volatile" else 0.0

        # --- Price context ---
        close = feats.get("close", feats.get("price", 0))
        atr = feats.get("atr_14", 0)
        feats["atr_pct_of_price"] = (atr / close * 100) if close > 0 and atr > 0 else 0.0

        # --- MACD state ---
        macd_hist = feats.get("macd_hist", 0)
        feats["macd_hist_sign"] = 1.0 if macd_hist > 0 else (-1.0 if macd_hist < 0 else 0.0)
        feats["macd_hist_strength"] = abs(macd_hist) / (close + 1e-8) * 100 if close > 0 else 0.0

        # --- Bollinger context ---
        feats["bb_position_signal"] = feats.get("bb_position", 0.5)
        feats["bb_width_signal"] = feats.get("bb_width", 0.0)

        # --- Volume context ---
        feats["volume_ratio_signal"] = feats.get("volume_sma_ratio", 1.0)

        # --- Momentum at signal ---
        feats["rsi_at_signal"] = feats.get("rsi_14", 50)
        feats["ema_slope_sign"] = 1.0 if feats.get("ema_20_slope", 0) > 0 else (-1.0 if feats.get("ema_20_slope", 0) < 0 else 0.0)

        # --- Trade-specific ---
        feats["signal_is_long"] = 1.0 if t.get("signal_type") == "LONG" else 0.0
        feats["signal_is_short"] = 1.0 if t.get("signal_type") == "SHORT" else 0.0

    return sorted_trades  # Return sorted for downstream consistency


# ─── Data loading ───────────────────────────────────────────────

def resample_1h_to_1d(df_1h):
    df = df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("dt")
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()
    daily = daily.reset_index()
    # datetime64[ms] internal representation is already in ms — no division needed
    daily["timestamp"] = daily["dt"].astype("int64")
    return daily.drop(columns=["dt"])


def load_symbol_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 1H data and compute 1D resample for a symbol."""
    fname = SYMBOLS_AVAILABLE.get(symbol)
    if not fname:
        raise ValueError(f"Unknown symbol: {symbol}")
    path = PROJECT_ROOT / "data" / "cache" / fname
    df = pd.read_parquet(path)
    if df["timestamp"].dtype == "datetime64[ns]":
        df["timestamp"] = df["timestamp"].astype("int64") // 1_000_000
    df_1d = resample_1h_to_1d(df)
    return df, df_1d


# ─── Backtest runner ────────────────────────────────────────────

def run_backtest_for_trades(data_1h, data_1d, config, strategy_class):
    """Run walk-forward backtest and return trades with signal features, plus metrics."""
    engine = BacktestEngine(config)
    result = engine.run_walk_forward(data_1h, strategy_class, data_1d=data_1d)
    trades = add_signal_features(result.trades)
    metrics = calculate_metrics(trades, initial_capital=10000)
    return trades, metrics


# ─── Grid search evaluation ─────────────────────────────────────

def evaluate_hyperparams(combo: dict, train_trades: list, val_trades: list,
                         base_config: dict) -> dict:
    """Train MetaLabeler with given hyperparams, evaluate on val set.

    Returns dict with combo, val metrics, and training diagnostics.
    """
    cfg = copy.deepcopy(base_config)
    cfg["meta_labeling"]["min_confidence"] = combo["min_confidence"]

    # Train
    labeler = MetaLabeler(cfg)
    labeler._model_params["max_depth"] = combo["max_depth"]
    labeler._model_params["n_estimators"] = combo["n_estimators"]
    labeler._model_params["learning_rate"] = combo.get("learning_rate", 0.03)
    labeler._model_params["reg_alpha"] = combo.get("reg_alpha", 2.0)
    labeler._model_params["reg_lambda"] = combo.get("reg_lambda", 3.0)

    ok = labeler.train(train_trades)
    if not ok:
        return {**combo, "val_sharpe": -99, "val_trades": 0,
                "val_pnl": 0, "train_accuracy": 0, "error": "train_failed"}

    diag = labeler.get_diagnostics()

    # Filter val trades
    filtered = []
    rejected = 0
    for t in val_trades:
        feats = t.get("features_at_signal")
        if feats and isinstance(feats, dict):
            from strategies.base import Signal
            sig = Signal.LONG if t.get("side") == "long" else Signal.SHORT
            if labeler.evaluate(sig, feats):
                filtered.append(t)
            else:
                rejected += 1
        else:
            filtered.append(t)

    val_metrics = calculate_metrics(filtered, initial_capital=10000) if filtered else \
        {"total_trades": 0, "total_pnl": 0, "sharpe_ratio": 0, "win_rate": 0,
         "profit_factor": 0, "max_drawdown_pct": 0}

    return {**combo,
            "val_sharpe": val_metrics["sharpe_ratio"],
            "val_trades": val_metrics["total_trades"],
            "val_pnl": val_metrics["total_pnl"],
            "val_wr": val_metrics["win_rate"],
            "val_pf": val_metrics.get("profit_factor", 0) or 0,
            "rejected": rejected,
            "train_accuracy": diag.get("val_accuracy", 0),
            "train_samples": diag.get("training_samples", 0)}


# ─── Main ───────────────────────────────────────────────────────

def main(quick: bool = False, symbols_str: str = "BTC,ETH"):
    symbols = [s.strip() for s in symbols_str.split(",")]
    print("=" * 72)
    print(f"  META-LABELING v2: {'+'.join(symbols)} | {'Quick' if quick else 'Full'} Grid")
    print("=" * 72)

    # --- 1. Load all symbol data ---
    all_data = {}
    for sym in symbols:
        df_1h, df_1d = load_symbol_data(sym)
        print(f"  {sym}: {len(df_1h):,} 1H bars | {len(df_1d):,} 1D bars | "
              f"{pd.to_datetime(df_1h['timestamp'].iloc[0], unit='ms').date()} -> "
              f"{pd.to_datetime(df_1h['timestamp'].iloc[-1], unit='ms').date()}")
        all_data[sym] = (df_1h, df_1d)

    # --- 2. 60/20/20 split (per symbol) ---
    sym_splits = {}
    for sym, (df_1h, df_1d) in all_data.items():
        n = len(df_1h)
        n_train = int(n * 0.60)
        n_val = int(n * 0.20)
        train_1h = df_1h.iloc[:n_train].reset_index(drop=True)
        val_1h = df_1h.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_1h = df_1h.iloc[n_train + n_val:].reset_index(drop=True)
        sym_splits[sym] = (train_1h, val_1h, test_1h, df_1d)

        t0 = pd.to_datetime(train_1h["timestamp"].iloc[0], unit="ms").date()
        t1 = pd.to_datetime(train_1h["timestamp"].iloc[-1], unit="ms").date()
        v1 = pd.to_datetime(val_1h["timestamp"].iloc[-1], unit="ms").date()
        t2 = pd.to_datetime(test_1h["timestamp"].iloc[-1], unit="ms").date()
        print(f"  {sym} split: TRAIN {t0}->{t1} | VAL {t1}->{v1} | TEST {v1}->{t2}")

    # --- 3. Generate training data (per symbol, parallel) ---
    print("\n" + "=" * 72)
    print("  PHASE 1: Generate labeled signals (MTF_MACD on TRAIN+VAL)")
    print("=" * 72)

    all_train_trades = []
    all_val_trades = []
    all_test_trades_raw = []
    total_t0 = time.monotonic()

    for sym, (train_1h, val_1h, test_1h, df_1d) in sym_splits.items():
        # Train backtest
        t0 = time.monotonic()
        train_trades, train_m = run_backtest_for_trades(train_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        train_t = time.monotonic() - t0
        wins = sum(1 for t in train_trades if t.get("pnl", 0) > 0)
        print(f"  {sym} TRAIN: {len(train_trades)} trades ({wins} wins) | "
              f"PnL=${train_m['total_pnl']:+,.0f} | Sharpe={train_m['sharpe_ratio']:.2f} | "
              f"WR={train_m['win_rate']:.1f}% | {train_t:.0f}s")

        # Val backtest
        t0 = time.monotonic()
        val_trades, val_m = run_backtest_for_trades(val_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        val_t = time.monotonic() - t0
        wins = sum(1 for t in val_trades if t.get("pnl", 0) > 0)
        print(f"  {sym} VAL:   {len(val_trades)} trades ({wins} wins) | "
              f"PnL=${val_m['total_pnl']:+,.0f} | Sharpe={val_m['sharpe_ratio']:.2f} | "
              f"WR={val_m['win_rate']:.1f}% | {val_t:.0f}s")

        all_train_trades.extend(train_trades)
        all_val_trades.extend(val_trades)

        # Keep test data for final phase
        all_test_trades_raw.append((sym, test_1h, df_1d))

    total_elapsed = time.monotonic() - total_t0
    print(f"\n  Combined: {len(all_train_trades)} train + {len(all_val_trades)} val signals "
          f"({total_elapsed:.0f}s)")

    # --- 4. Grid search ---
    print("\n" + "=" * 72)
    print("  PHASE 2: Hyperparameter Grid Search")
    print("=" * 72)

    if quick:
        grid = {
            "min_confidence": [0.52, 0.55, 0.60],
            "max_depth": [3, 5],
            "n_estimators": [50, 150],
            "learning_rate": [0.03],
            "reg_alpha": [2.0],
            "reg_lambda": [3.0],
        }
    else:
        grid = {
            "min_confidence": [0.50, 0.52, 0.55, 0.58, 0.60, 0.65],
            "max_depth": [2, 3, 4, 5],
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.02, 0.03, 0.05],
            "reg_alpha": [1.0, 2.0, 4.0],
            "reg_lambda": [2.0, 3.0, 5.0],
        }

    keys = list(grid.keys())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]
    n_combos = len(combos)
    print(f"  Grid: {n_combos} combos ({' x '.join(str(len(grid[k])) for k in keys)})")

    # Parallel grid search
    try:
        import joblib
        backend = "loky"
        n_jobs = -1
        print(f"  Using joblib with n_jobs=-1")
    except ImportError:
        joblib = None
        backend = None
        n_jobs = 1
        print(f"  Using ProcessPoolExecutor")

    t0 = time.monotonic()

    if joblib:
        results = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(evaluate_hyperparams)(c, all_train_trades, all_val_trades, BASE_CONFIG)
            for c in combos
        )
    else:
        results = []
        with ProcessPoolExecutor() as ex:
            futures = {ex.submit(evaluate_hyperparams, c, all_train_trades, all_val_trades, BASE_CONFIG): c
                      for c in combos}
            for f in as_completed(futures):
                results.append(f.result())

    elapsed = time.monotonic() - t0
    print(f"  Done: {elapsed:.0f}s ({elapsed/n_combos:.1f}s/combo)")

    # Sort and pick best
    valid = [r for r in results if r.get("val_trades", 0) >= 3 and r.get("val_sharpe", -99) > -99]
    if not valid:
        print("  ERROR: No valid hyperparameter combos. Aborting.")
        return
    valid.sort(key=lambda r: r["val_sharpe"], reverse=True)
    best = valid[0]

    print(f"\n  Top 5 by Val Sharpe:")
    print(f"  {'Conf':<6} {'D':<3} {'N':<4} {'LR':<5} {'a':<4} {'l':<4} "
          f"{'Val S':<8} {'Val T':<6} {'Val PnL':<10} {'Val WR':<7} {'Acc':<6} {'Rej':<5}")
    print(f"  {'-'*75}")
    for r in valid[:5]:
        print(f"  {r['min_confidence']:<6.2f} {r['max_depth']:<3} {r['n_estimators']:<4} "
              f"{r.get('learning_rate', 0.03):<5.3f} {r.get('reg_alpha', 2.0):<4.1f} "
              f"{r.get('reg_lambda', 3.0):<4.1f} "
              f"{r['val_sharpe']:<8.2f} {r['val_trades']:<6} "
              f"${r['val_pnl']:>+8.0f}  {r.get('val_wr', 0):<6.1f}% "
              f"{r.get('train_accuracy', 0):<6.3f} {r.get('rejected', 0):<5}")

    # --- 5. Final test ---
    print("\n" + "=" * 72)
    print("  PHASE 3: Final Test (best params on TEST)")
    print("=" * 72)
    print(f"  Best: conf={best['min_confidence']:.2f} D={best['max_depth']} "
          f"N={best['n_estimators']} lr={best.get('learning_rate', 0.03):.3f} "
          f"alpha={best.get('reg_alpha', 2.0)} lambda={best.get('reg_lambda', 3.0)}")

    # Train final model on TRAIN+VAL combined
    final_cfg = copy.deepcopy(BASE_CONFIG)
    final_cfg["meta_labeling"]["min_confidence"] = best["min_confidence"]
    final_labeler = MetaLabeler(final_cfg)
    final_labeler._model_params["max_depth"] = best["max_depth"]
    final_labeler._model_params["n_estimators"] = best["n_estimators"]
    final_labeler._model_params["learning_rate"] = best.get("learning_rate", 0.03)
    final_labeler._model_params["reg_alpha"] = best.get("reg_alpha", 2.0)
    final_labeler._model_params["reg_lambda"] = best.get("reg_lambda", 3.0)

    all_train_val = all_train_trades + all_val_trades
    final_labeler.train(all_train_val)
    diag = final_labeler.get_diagnostics()
    print(f"  Final model: {diag['training_samples']} samples, "
          f"val_acc={diag['val_accuracy']:.3f}")

    # Test each symbol
    all_test_base = []
    all_test_filtered = []
    total_rejected = 0

    for sym, test_1h, df_1d in all_test_trades_raw:
        test_trades, test_base_m = run_backtest_for_trades(test_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        all_test_base.extend(test_trades)

        # Filter with meta-labeler
        filtered = []
        rejected = 0
        for t in test_trades:
            feats = t.get("features_at_signal")
            if feats and isinstance(feats, dict):
                from strategies.base import Signal
                sig = Signal.LONG if t.get("side") == "long" else Signal.SHORT
                if final_labeler.evaluate(sig, feats):
                    filtered.append(t)
                else:
                    rejected += 1
            else:
                filtered.append(t)
        all_test_filtered.extend(filtered)
        total_rejected += rejected

        test_f_m = calculate_metrics(filtered, initial_capital=10000) if filtered else \
            {"total_trades": 0, "total_pnl": 0, "sharpe_ratio": 0, "win_rate": 0,
             "profit_factor": 0, "max_drawdown_pct": 0}

        print(f"  {sym}: {len(test_trades)} -> {len(filtered)} trades "
              f"({rejected} rejected) | "
              f"Sharpe {test_base_m['sharpe_ratio']:.2f} -> {test_f_m['sharpe_ratio']:.2f}")

    # --- 6. Results ---
    base_m = calculate_metrics(all_test_base, initial_capital=10000)
    ml_m = calculate_metrics(all_test_filtered, initial_capital=10000) if all_test_filtered else \
        {"total_trades": 0, "total_pnl": 0, "sharpe_ratio": 0, "win_rate": 0,
         "profit_factor": 0, "max_drawdown_pct": 0}

    print("\n" + "=" * 72)
    print("  RESULTS: MTF_MACD vs MTF_MACD + MetaLabeler (v2)")
    print("=" * 72)

    print(f"\n  {'Metric':<22} {'MTF_MACD':<14} {'+ MetaLabeler':<14} {'Change':<10}")
    print(f"  {'-'*60}")
    print(f"  {'Trades':<22} {base_m['total_trades']:<14} {ml_m['total_trades']:<14} "
          f"{ml_m['total_trades'] - base_m['total_trades']:+d}")
    print(f"  {'Win Rate':<22} {base_m['win_rate']:<13.1f}% {ml_m['win_rate']:<13.1f}% "
          f"{ml_m['win_rate'] - base_m['win_rate']:+.1f}%")
    print(f"  {'Total PnL':<22} ${base_m['total_pnl']:<13,.0f} ${ml_m['total_pnl']:<13,.0f} "
          f"${ml_m['total_pnl'] - base_m['total_pnl']:+,.0f}")
    print(f"  {'Sharpe':<22} {base_m['sharpe_ratio']:<14.2f} {ml_m['sharpe_ratio']:<14.2f} "
          f"{ml_m['sharpe_ratio'] - base_m['sharpe_ratio']:+.2f}")
    print(f"  {'Max DD':<22} {base_m['max_drawdown_pct']:<13.1f}% {ml_m['max_drawdown_pct']:<13.1f}% "
          f"{ml_m['max_drawdown_pct'] - base_m['max_drawdown_pct']:+.1f}%")
    bpf = base_m.get("profit_factor", 0) or 0
    mpf = ml_m.get("profit_factor", 0) or 0
    print(f"  {'Profit Factor':<22} {bpf:<14.2f} {mpf:<14.2f} {mpf - bpf:+.2f}")

    total_trades = len(all_test_base)
    print(f"\n  Rejected: {total_rejected}/{total_trades} signals "
          f"({total_rejected/total_trades*100:.0f}%)" if total_trades else "  No trades")

    # Best hyperparams summary
    print(f"\n  Best params: conf={best['min_confidence']:.2f} | "
          f"max_depth={best['max_depth']} | n_estimators={best['n_estimators']} | "
          f"lr={best.get('learning_rate', 0.03)} | "
          f"reg_alpha={best.get('reg_alpha', 2.0)} | reg_lambda={best.get('reg_lambda', 3.0)}")
    print(f"  Train samples: {len(all_train_trades)} (+{len(all_val_trades)} val)")

    sharpe_diff = ml_m['sharpe_ratio'] - base_m['sharpe_ratio']
    if sharpe_diff > 0.1:
        print(f"  [OK] MetaLabeler v2 IMPROVES MTF_MACD (dSharpe=+{sharpe_diff:.2f})")
    elif sharpe_diff < -0.1:
        print(f"  [!!] MetaLabeler v2 DEGRADES MTF_MACD (dSharpe={sharpe_diff:.2f})")
    else:
        print(f"  [--] MetaLabeler v2 has negligible impact (dSharpe={sharpe_diff:+.2f})")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Meta-Labeling v2 Backtest")
    p.add_argument("--quick", action="store_true", help="Fast grid (12 combos)")
    p.add_argument("--symbols", type=str, default="BTC,ETH", help="Symbols (comma-separated)")
    args = p.parse_args()
    main(quick=args.quick, symbols_str=args.symbols)
