"""XGBoost ML Backtest -- 60/20/20 Train/Val/Test with hyperparameter grid search.

Compares XGBoostCostAware against MTF_MACD_Elder baseline on BTC/USDT 1H data.

Methodology (60/20/20 chronological split, no walk-forward within splits):
    Phase 1 - TRAIN (60%): Grid search over cost_lambda x confidence x max_depth
                            x lr x n_estimators. Top 10 by train Sharpe.
    Phase 2 - VAL (20%):    Evaluate top 10 on validation. Best by val Sharpe.
    Phase 3 - TEST (20%):   Final untouched evaluation + MTF_MACD baseline + buy-hold.

Usage:
    python research/xgboost_backtest.py              # default grid
    python research/xgboost_backtest.py --quick      # fast grid (25 combos)
    python research/xgboost_backtest.py --data path.parquet  # custom data
"""

import os
import sys
import time
import itertools
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics
from strategies.xgb_cost_aware import XGBoostCostAware
from strategies.mtf_macd import MTF_MACD_Elder


# ─── Helpers ────────────────────────────────────────────────────

def resample_1h_to_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV to 1D candles."""
    df = df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("dt")

    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    daily = daily.reset_index()
    daily["timestamp"] = daily["dt"].astype("int64") // 1_000_000
    daily = daily.drop(columns=["dt"])
    return daily


def build_config(lam: float, confidence: float, max_depth: int,
                 learning_rate: float, n_estimators: int) -> dict:
    """Build complete BacktestEngine config dict for a given hyperparameter combo."""
    return {
        "exchange": {
            "name": "bitget",
            "symbols": ["BTC/USDT"],
            "type": "spot",
            "fees": {
                "taker": 0.0006,
                "maker": 0.0002,
                "slippage": 0.0005,
            },
        },
        "risk": {
            "initial_capital": 10000,
            "max_position_pct": 0.20,
            "per_trade": {
                "max_duration_hours": 72,
            },
        },
        "features": {
            "max_window_bars": 500,
            "min_bars_required": 50,
        },
        "backtest": {
            "walk_forward_folds": 5,
            "min_train_fraction": 0.33,
            "min_signal_exit_bars": 6,
            "cooldown_bars_after_loss": 2,
        },
        "strategies": {
            "xgboost_cost_aware": {
                "model_params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": 0.7,
                    "colsample_bytree": 0.6,
                    "reg_alpha": 2.0,
                    "reg_lambda": 3.0,
                    "min_child_weight": 1,
                    "random_state": 42,
                    "eval_metric": "logloss",
                },
                "training": {
                    "retrain_every_candles": 500,
                    "min_train_samples": 1000,
                    "validation_fraction": 0.2,
                    "early_stopping_rounds": 20,
                },
                "cost_filter": {
                    "lambda": lam,
                    "transaction_cost_bps": 30,
                },
                "trading": {
                    "confidence_threshold": confidence,
                    "allow_shorts": True,
                },
                "target": {
                    "horizon": 4,
                    "dead_zone_pct": 0.001,
                },
            },
            "mtf_macd_elder": {
                "enabled": True,
                "macd": {"fast": 10, "slow": 20, "signal": 9},
                "exit": {"trailing_stop_pct": 0.02, "atr_stop_mult": 3.0, "min_hold_bars": 6},
                "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
            },
        },
        "regime": {
            "enabled": True,
            "hysteresis_bars": 2,
            "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
            "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
            "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
        },
    }


def run_backtest(data: pd.DataFrame, data_1d: pd.DataFrame,
                 config: dict, strategy_class) -> dict:
    """Run a walk-forward backtest and return metrics dict."""
    engine = BacktestEngine(config)
    try:
        result = engine.run_walk_forward(data, strategy_class, data_1d=data_1d)
        if result.trades:
            return calculate_metrics(result.trades, initial_capital=10000)
        return _empty_metrics()
    except Exception as e:
        print(f"  Backtest error: {e}")
        return _empty_metrics(error=str(e))


def _empty_metrics(error: str = "") -> dict:
    return {
        "total_trades": 0, "total_pnl": 0, "total_return_pct": 0,
        "sharpe_ratio": 0, "sortino_ratio": 0, "max_drawdown_pct": 0,
        "win_rate": 0, "profit_factor": 0, "max_consecutive_losses": 0,
        "avg_bars_held": 0, "error": error,
    }


def buy_hold_return(data: pd.DataFrame, split_start: int, split_end: int) -> float:
    """Buy-and-hold return over a data slice."""
    if split_start >= split_end or split_end >= len(data):
        return 0.0
    start_price = data.iloc[split_start]["close"]
    end_price = data.iloc[split_end - 1]["close"]
    return (end_price / start_price - 1) * 100 if start_price > 0 else 0.0


# ─── Main ───────────────────────────────────────────────────────

def main(data_path: str = None, quick: bool = False):
    # ── 1. Load Data ───────────────────────────────────────────
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "cache" / "btc_1h_2020_2026.parquet"

    print("=" * 72)
    print("  XGBOOST ML BACKTEST -- 60/20/20 Chronological Split")
    print("=" * 72)

    print(f"\nLoading: {data_path}")
    df_1h = pd.read_parquet(data_path)
    if "timestamp" not in df_1h.columns and df_1h.index.name != "timestamp":
        df_1h = df_1h.reset_index()
    if df_1h["timestamp"].dtype == "datetime64[ns]":
        df_1h["timestamp"] = df_1h["timestamp"].astype("int64") // 1_000_000

    print(f"  1H candles: {len(df_1h):,}  |  "
          f"{pd.to_datetime(df_1h['timestamp'].iloc[0], unit='ms').date()}"
          f" -> {pd.to_datetime(df_1h['timestamp'].iloc[-1], unit='ms').date()}")

    # Resample to 1D
    df_1d = resample_1h_to_1d(df_1h)
    print(f"  1D candles: {len(df_1d):,}")

    # 60/20/20 split (by row index, sequential)
    n = len(df_1h)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    n_test = n - n_train - n_val

    train_1h = df_1h.iloc[:n_train].reset_index(drop=True)
    val_1h = df_1h.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_1h = df_1h.iloc[n_train + n_val:].reset_index(drop=True)

    # Split 1D data to match 1H time ranges
    train_ts_end = train_1h["timestamp"].iloc[-1]
    val_ts_end = val_1h["timestamp"].iloc[-1]
    train_1d = df_1d[df_1d["timestamp"] <= train_ts_end]
    val_1d = df_1d[(df_1d["timestamp"] > train_ts_end) & (df_1d["timestamp"] <= val_ts_end)]
    test_1d = df_1d[df_1d["timestamp"] > val_ts_end]

    print(f"\n  Split: TRAIN={len(train_1h):,} (60%) | VAL={len(val_1h):,} (20%) | TEST={len(test_1h):,} (20%)")
    print(f"  BTC buy-hold (TEST): {buy_hold_return(test_1h, 0, len(test_1h)):+.1f}%")

    # ── 2. PHASE 1: Grid Search on TRAIN ────────────────────────
    print("\n" + "=" * 72)
    print("  PHASE 1: GRID SEARCH (TRAIN, 60%)")
    print("=" * 72)

    if quick:
        grid = {
            "lam": [3.0, 5.0],
            "confidence": [0.55, 0.60],
            "max_depth": [3, 5],
            "learning_rate": [0.02, 0.05],
            "n_estimators": [100, 200],
        }
    else:
        grid = {
            "lam": [2.0, 3.0, 4.0, 5.0, 6.0],
            "confidence": [0.52, 0.55, 0.58, 0.60],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.02, 0.05, 0.08],
            "n_estimators": [100, 200, 300],
        }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    n_combos = len(combos)
    print(f"  Grid: {n_combos} combos ({' x '.join(str(len(grid[k])) for k in keys)})")
    print(f"  Dims: {' x '.join(keys)}")

    results = []
    t0 = time.monotonic()
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg = build_config(**params)

        try:
            metrics = run_backtest(train_1h, train_1d, cfg, XGBoostCostAware)
        except Exception as e:
            metrics = _empty_metrics(error=str(e))

        results.append({**params, **metrics})
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.monotonic() - t0
            eta = (elapsed / (i + 1)) * (n_combos - i - 1)
            valid = sum(1 for r in results if r["total_trades"] >= 5)
            print(f"  [{i+1}/{n_combos}] {valid} valid | elapsed={elapsed:.0f}s | ETA={eta:.0f}s")

    elapsed = time.monotonic() - t0
    print(f"\n  Grid search complete: {elapsed:.0f}s")

    # Filter and sort
    valid_results = [r for r in results if r["total_trades"] >= 5]
    if not valid_results:
        print("  ERROR: No valid combos (all produced < 5 trades). Aborting.")
        return

    valid_results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)
    top10 = valid_results[:10]

    print(f"\n  Top 10 by Train Sharpe:")
    print(f"  {'Rank':<5} {'lambda':<6} {'Conf':<6} {'D':<4} {'LR':<6} {'N':<5} {'Sharpe':<8} {'Trades':<7} {'PnL':<10} {'DD':<6} {'WR':<6}")
    print(f"  {'-'*70}")
    for i, r in enumerate(top10):
        print(f"  {i+1:<5} {r['lam']:<6.1f} {r['confidence']:<6.2f} {r['max_depth']:<4} "
              f"{r['learning_rate']:<6.3f} {r['n_estimators']:<5} "
              f"{r['sharpe_ratio']:<8.2f} {r['total_trades']:<7} "
              f"${r['total_pnl']:>+8.0f}  {r['max_drawdown_pct']:<5.1f}% {r['win_rate']:<5.1f}%")

    # ── 3. PHASE 2: Validation ──────────────────────────────────
    print("\n" + "=" * 72)
    print("  PHASE 2: VALIDATION (VAL, 20%)")
    print("=" * 72)

    for i, r in enumerate(top10):
        cfg = build_config(r["lam"], r["confidence"], r["max_depth"],
                          r["learning_rate"], r["n_estimators"])
        val_m = run_backtest(val_1h, val_1d, cfg, XGBoostCostAware)
        top10[i]["val_sharpe"] = val_m["sharpe_ratio"]
        top10[i]["val_pnl"] = val_m["total_pnl"]
        top10[i]["val_trades"] = val_m["total_trades"]
        top10[i]["val_dd"] = val_m["max_drawdown_pct"]
        print(f"  [{i+1}/10] lambda={r['lam']:.1f} conf={r['confidence']:.2f} "
              f"train_S={r['sharpe_ratio']:.2f} val_S={val_m['sharpe_ratio']:.2f} "
              f"val_PnL=${val_m['total_pnl']:+.0f}")

    top10.sort(key=lambda r: r.get("val_sharpe", 0), reverse=True)
    best = top10[0]
    gap = best["sharpe_ratio"] - best.get("val_sharpe", 0)
    print(f"\n  Best (val): lambda={best['lam']:.1f} conf={best['confidence']:.2f} "
          f"D={best['max_depth']} lr={best['learning_rate']:.3f} N={best['n_estimators']}")
    if gap > 1.0:
        print(f"  !!️  Train-Val Sharpe gap: {gap:.2f} -- possible overfitting")

    # ── 4. BASELINE: MTF_MACD on TEST ────────────────────────────
    print("\n" + "=" * 72)
    print("  BASELINE: MTF_MACD_Elder on TEST")
    print("=" * 72)

    base_cfg = build_config(lam=4.0, confidence=0.55, max_depth=4,
                           learning_rate=0.03, n_estimators=200)
    base_m = run_backtest(test_1h, test_1d, base_cfg, MTF_MACD_Elder)
    print(f"  MTF_MACD: {base_m['total_trades']} trades | "
          f"PnL=${base_m['total_pnl']:+,.0f} | "
          f"Sharpe={base_m['sharpe_ratio']:.2f} | "
          f"DD={base_m['max_drawdown_pct']:.1f}% | "
          f"WR={base_m['win_rate']:.1f}%")

    # ── 5. PHASE 3: Final TEST ──────────────────────────────────
    print("\n" + "=" * 72)
    print("  PHASE 3: FINAL TEST (TEST, 20%)")
    print("=" * 72)

    best_cfg = build_config(best["lam"], best["confidence"], best["max_depth"],
                           best["learning_rate"], best["n_estimators"])
    print(f"  Running XGBoostCostAware with best params...")
    test_m = run_backtest(test_1h, test_1d, best_cfg, XGBoostCostAware)

    bh_ret = buy_hold_return(test_1h, 0, len(test_1h))

    # ── 6. Results Table ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS: XGBoost vs MTF_MACD vs Buy-Hold")
    print("=" * 72)

    def fmt_metric(name, xgb, macd, bh):
        return f"  {name:<20} {str(xgb):<12} {str(macd):<12} {str(bh):<12}"

    print(f"  {'Metric':<20} {'XGBoost':<12} {'MTF_MACD':<12} {'Buy-Hold':<12}")
    print(f"  {'-'*56}")
    print(fmt_metric("Trades", test_m["total_trades"], base_m["total_trades"], "-"))
    print(fmt_metric("Win Rate",
           f"{test_m['win_rate']:.1f}%", f"{base_m['win_rate']:.1f}%", "-"))
    print(fmt_metric("Total PnL",
           f"${test_m['total_pnl']:+,.0f}", f"${base_m['total_pnl']:+,.0f}",
           f"${(bh_ret/100)*10000:+,.0f}"))
    print(fmt_metric("Return",
           f"{test_m['total_return_pct']:+.1f}%", f"{base_m['total_return_pct']:+.1f}%",
           f"{bh_ret:+.1f}%"))
    print(fmt_metric("Sharpe",
           f"{test_m['sharpe_ratio']:.2f}", f"{base_m['sharpe_ratio']:.2f}", "-"))
    print(fmt_metric("Sortino",
           f"{test_m['sortino_ratio']:.2f}", f"{base_m['sortino_ratio']:.2f}", "-"))
    print(fmt_metric("Max DD",
           f"{test_m['max_drawdown_pct']:.1f}%", f"{base_m['max_drawdown_pct']:.1f}%",
           f"{abs(bh_ret/2):.1f}%*"))
    print(fmt_metric("Profit Factor",
           f"{test_m.get('profit_factor', 0):.2f}", f"{base_m.get('profit_factor', 0):.2f}", "-"))

    # ── 7. Overfitting Detection ────────────────────────────────
    print("\n" + "-" * 56)
    warnings = []
    if test_m["sharpe_ratio"] > 3.0:
        warnings.append(f"test Sharpe {test_m['sharpe_ratio']:.2f} > 3.0 -- unusually high")
    if test_m["max_drawdown_pct"] < 10.0 and test_m["total_trades"] > 10:
        warnings.append(f"test DD {test_m['max_drawdown_pct']:.1f}% < 10% -- unusually low")
    test_gap = abs(best["sharpe_ratio"] - test_m["sharpe_ratio"])
    max_s = max(abs(test_m["sharpe_ratio"]), 0.01)
    if test_gap / max_s > 0.5:
        warnings.append(f"train-test Sharpe gap {test_gap:.2f} > 50% -- overfitting suspected")

    if warnings:
        print("  !!️  OVERFITTING WARNINGS:")
        for w in warnings:
            print(f"     * {w}")
        print("  Consider: fewer trees, higher reg_alpha/reg_lambda, or more restrictive cost lambda")
    else:
        print("  [OK] No overfitting flags triggered")

    # Final verdict
    print(f"\n  XGBoost vs MTF_MACD: "
          f"{'ML WINS' if test_m['sharpe_ratio'] > base_m['sharpe_ratio'] else 'RULE-BASED WINS'} "
          f"(DeltaSharpe={test_m['sharpe_ratio'] - base_m['sharpe_ratio']:+.2f})")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XGBoost ML Backtest")
    parser.add_argument("--quick", action="store_true", help="Fast grid (25 combos)")
    parser.add_argument("--data", type=str, default=None, help="Path to parquet file")
    args = parser.parse_args()
    main(data_path=args.data, quick=args.quick)
