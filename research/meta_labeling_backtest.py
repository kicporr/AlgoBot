"""Meta-Labeling Backtest: ML filter for MTF_MACD signals.

Trains an XGBoost meta-labeler on historical MTF_MACD signals, then compares
MTF_MACD alone vs MTF_MACD + MetaLabeler on test data.

Methodology:
    1. Run MTF_MACD backtest on TRAIN period (60%) -> collect signals with features + outcomes
    2. Train MetaLabeler: X = features_at_signal, y = (pnl > 0)
    3. Run MTF_MACD + MetaLabeler backtest on TEST period (20%)
    4. Compare: trades, PnL, Sharpe, win rate, profit factor

Usage:
    python research/meta_labeling_backtest.py
"""

import os
import sys
import time
from pathlib import Path

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
    "exchange": {
        "name": "bitget", "symbols": ["BTC/USDT"], "type": "spot",
        "fees": {"taker": 0.0006, "maker": 0.0002, "slippage": 0.0005},
    },
    "risk": {"initial_capital": 10000, "max_position_pct": 0.20},
    "features": {"max_window_bars": 500, "min_bars_required": 50},
    "backtest": {
        "walk_forward_folds": 5, "min_train_fraction": 0.33,
        "min_signal_exit_bars": 6, "cooldown_bars_after_loss": 2,
    },
    "strategies": {
        "mtf_macd_elder": {
            "enabled": True,
            "macd": {"fast": 10, "slow": 20, "signal": 9},
            "exit": {"trailing_stop_pct": 0.02, "atr_stop_mult": 3.0, "min_hold_bars": 6},
            "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
        },
    },
    "meta_labeling": {
        "enabled": True,
        "model": "xgboost",
        "training_samples": 1000,
        "min_confidence": 0.55,
    },
    "regime": {
        "enabled": True, "hysteresis_bars": 2,
        "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
        "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
        "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
    },
}


def resample_1h_to_1d(df_1h):
    df = df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("dt")
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    daily = daily.reset_index()
    daily["timestamp"] = daily["dt"].astype("int64") // 1_000_000
    return daily.drop(columns=["dt"])


def _empty_metrics():
    return {"total_trades": 0, "total_pnl": 0, "total_return_pct": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0, "win_rate": 0,
            "profit_factor": 0, "avg_bars_held": 0}


# ─── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  META-LABELING BACKTEST: MTF_MACD + ML Filter")
    print("=" * 72)

    # 1. Load data
    data_path = PROJECT_ROOT / "data" / "cache" / "btc_1h_2020_2026.parquet"
    print(f"\nLoading: {data_path}")
    df_1h = pd.read_parquet(data_path)
    if df_1h["timestamp"].dtype == "datetime64[ns]":
        df_1h["timestamp"] = df_1h["timestamp"].astype("int64") // 1_000_000

    df_1d = resample_1h_to_1d(df_1h)
    print(f"  1H candles: {len(df_1h):,}  |  1D candles: {len(df_1d):,}")

    # 60/20/20 split
    n = len(df_1h)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)

    train_1h = df_1h.iloc[:n_train].reset_index(drop=True)
    val_1h = df_1h.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_1h = df_1h.iloc[n_train + n_val:].reset_index(drop=True)

    # D1 data: include buffer of 60 days before train/test for MACD computation
    # (MACD slow=26 + signal=9 = 35 daily candles minimum)
    train_ts_start = train_1h["timestamp"].iloc[0]
    train_ts_end = train_1h["timestamp"].iloc[-1]
    test_ts_start = test_1h["timestamp"].iloc[0]
    test_ts_end = test_1h["timestamp"].iloc[-1]
    buffer_ms = 60 * 86400_000  # 60 days in ms

    train_1d = df_1d[(df_1d["timestamp"] >= train_ts_start - buffer_ms) &
                     (df_1d["timestamp"] <= train_ts_end)]
    test_1d = df_1d[(df_1d["timestamp"] >= test_ts_start - buffer_ms) &
                    (df_1d["timestamp"] <= test_ts_end)]

    print(f"  TRAIN: {len(train_1h):,} bars (60%)  |  D1: {len(train_1d)} daily")
    print(f"  TEST:  {len(test_1h):,} bars (20%)  |  D1: {len(test_1d)} daily\n")

    # 2. Phase 1: Run MTF_MACD on TRAIN to generate labeled signals
    print("=" * 72)
    print("  PHASE 1: Generate training data (MTF_MACD on TRAIN)")
    print("=" * 72)

    t0 = time.monotonic()
    engine = BacktestEngine(BASE_CONFIG)
    # Use full D1 data for both train and test (engine needs enough daily history for MACD)
    train_result = engine.run_walk_forward(train_1h, MTF_MACD_Elder, data_1d=df_1d)

    train_trades = train_result.trades
    train_metrics = calculate_metrics(train_trades, initial_capital=10000)
    elapsed = time.monotonic() - t0

    wins = sum(1 for t in train_trades if t.get("pnl", 0) > 0)
    has_features = sum(1 for t in train_trades if t.get("features_at_signal"))
    print(f"  Trades: {len(train_trades)} ({wins} wins, {len(train_trades) - wins} losses)")
    print(f"  With features: {has_features}/{len(train_trades)}")
    print(f"  MTF_MACD train: {train_metrics['total_trades']} trades | "
          f"PnL=${train_metrics['total_pnl']:+,.0f} | "
          f"Sharpe={train_metrics['sharpe_ratio']:.2f} | "
          f"WR={train_metrics['win_rate']:.1f}% | "
          f"Elapsed={elapsed:.0f}s")

    # 3. Phase 2: Train MetaLabeler
    print("\n" + "=" * 72)
    print("  PHASE 2: Train MetaLabeler on historical signals")
    print("=" * 72)

    labeler = MetaLabeler(BASE_CONFIG)
    t0 = time.monotonic()
    ok = labeler.train(train_trades)
    elapsed = time.monotonic() - t0

    if ok:
        diag = labeler.get_diagnostics()
        print(f"  Trained in {elapsed:.0f}s: {diag['training_samples']} samples, "
              f"{diag['features_used']} features, val_acc={diag['val_accuracy']}")
    else:
        print(f"  Training failed or insufficient data")
        return

    # 4. Phase 3: Filtered backtest on TEST
    print("\n" + "=" * 72)
    print("  PHASE 3: MTF_MACD + MetaLabeler on TEST")
    print("=" * 72)

    # Re-run with meta-labeler filtering
    engine2 = BacktestEngine(BASE_CONFIG)
    t0 = time.monotonic()
    test_result_raw = engine2.run_walk_forward(test_1h, MTF_MACD_Elder, data_1d=df_1d)

    # Apply meta-labeler filter to the test trades
    all_test_trades = test_result_raw.trades
    filtered_trades = []
    rejected_count = 0
    for t in all_test_trades:
        feats = t.get("features_at_signal")
        if feats and isinstance(feats, dict):
            # Determine signal type from trade side
            signal = __import__('strategies.base', fromlist=['Signal']).Signal.LONG if t.get("side") == "long" else __import__('strategies.base', fromlist=['Signal']).Signal.SHORT
            if labeler.evaluate(signal, feats):
                filtered_trades.append(t)
            else:
                rejected_count += 1
        else:
            filtered_trades.append(t)  # No features -> pass through

    elapsed = time.monotonic() - t0

    # MTF_MACD alone metrics on test
    test_base = calculate_metrics(all_test_trades, initial_capital=10000)
    # MTF_MACD + MetaLabeler metrics on test
    test_ml = calculate_metrics(filtered_trades, initial_capital=10000) if filtered_trades else _empty_metrics()

    # 5. Results
    print("\n" + "=" * 72)
    print("  RESULTS: MTF_MACD vs MTF_MACD + MetaLabeler")
    print("=" * 72)

    print(f"\n  {'Metric':<22} {'MTF_MACD':<14} {'+ MetaLabeler':<14} {'Change':<10}")
    print(f"  {'-'*60}")
    trades_change = f"{(test_ml['total_trades'] - test_base['total_trades']):+d}"
    print(f"  {'Trades':<22} {test_base['total_trades']:<14} {test_ml['total_trades']:<14} {trades_change:<10}")
    print(f"  {'Win Rate':<22} {test_base['win_rate']:<13.1f}% {test_ml['win_rate']:<13.1f}% "
          f"{test_ml['win_rate'] - test_base['win_rate']:+.1f}%")
    print(f"  {'Total PnL':<22} ${test_base['total_pnl']:<13,.0f} ${test_ml['total_pnl']:<13,.0f} "
          f"${test_ml['total_pnl'] - test_base['total_pnl']:+,.0f}")
    print(f"  {'Sharpe':<22} {test_base['sharpe_ratio']:<14.2f} {test_ml['sharpe_ratio']:<14.2f} "
          f"{test_ml['sharpe_ratio'] - test_base['sharpe_ratio']:+.2f}")
    print(f"  {'Max DD':<22} {test_base['max_drawdown_pct']:<13.1f}% {test_ml['max_drawdown_pct']:<13.1f}% "
          f"{test_ml['max_drawdown_pct'] - test_base['max_drawdown_pct']:+.1f}%")
    pf_base = test_base.get("profit_factor", 0) or 0
    pf_ml = test_ml.get("profit_factor", 0) or 0
    print(f"  {'Profit Factor':<22} {pf_base:<14.2f} {pf_ml:<14.2f} "
          f"{pf_ml - pf_base:+.2f}")

    print(f"\n  MetaLabeler rejected: {rejected_count}/{len(all_test_trades)} signals "
          f"({rejected_count/len(all_test_trades)*100:.0f}%)" if all_test_trades else "  No trades")

    sharpe_diff = test_ml['sharpe_ratio'] - test_base['sharpe_ratio']
    if sharpe_diff > 0.05:
        print(f"  [OK] MetaLabeler IMPROVES MTF_MACD (dSharpe=+{sharpe_diff:.2f})")
    elif sharpe_diff < -0.05:
        print(f"  [!!] MetaLabeler DEGRADES MTF_MACD (dSharpe={sharpe_diff:.2f})")
    else:
        print(f"  [--] MetaLabeler has negligible impact (dSharpe={sharpe_diff:+.2f})")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
