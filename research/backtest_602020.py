"""Proper 60/20/20 walk-forward backtest with overfitting detection.

Split:
    Train (60%): Walk-forward CV to find best hyperparameters
    Validation (20%): Evaluate selected params, tune thresholds
    Test (20%): Final out-of-sample evaluation — NEVER touched during tuning

Overfitting flags:
    - Sharpe > 3 on test set (unrealistic for directional BTC)
    - Drawdown < 10% on test set (too good to be true)
    - Huge gap between train and test performance
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import time, itertools, pandas as pd, numpy as np

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics
from strategies.xgb_cost_aware import XGBoostCostAware

# Load data
df_1h = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "btc_1h_d365.parquet")

# Resample 1D
df = df_1h.copy()
df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.set_index("dt")
daily = df.resample("1D", closed="left", label="left").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
})
daily["bar_count"] = df.resample("1D").size()
daily = daily[daily["bar_count"] >= 12].dropna().reset_index()
daily["timestamp"] = daily["dt"].astype("int64") // 1_000_000
daily.drop(columns=["dt"], inplace=True)
df_1d = daily

n_total = len(df_1h)
n_train = int(n_total * 0.60)
n_val = int(n_total * 0.20)
n_test = n_total - n_train - n_val

train_data = df_1h.iloc[:n_train]
val_data = df_1h.iloc[n_train:n_train + n_val]
test_data = df_1h.iloc[n_train + n_val:]

train_1d = df_1d[df_1d["timestamp"] <= train_data["timestamp"].iloc[-1]]
val_1d = df_1d[(df_1d["timestamp"] > train_data["timestamp"].iloc[-1]) &
                (df_1d["timestamp"] <= val_data["timestamp"].iloc[-1])]
test_1d = df_1d[df_1d["timestamp"] > val_data["timestamp"].iloc[-1]]

print(f"Total: {n_total:,} bars | Train: {n_train:,} ({n_train/n_total:.0%}) | "
      f"Val: {n_val:,} ({n_val/n_total:.0%}) | Test: {n_test:,} ({n_test/n_total:.0%})")

base_config = {
    "exchange": {"name": "bitget", "symbols": ["BTC/USDT"],
                 "fees": {"taker": 0.001, "maker": 0.001, "slippage": 0.0005}},
    "risk": {"initial_capital": 10000, "max_position_pct": 0.95},
    "features": {"max_window_bars": 500, "min_bars_required": 50},
    "strategies": {
        "mean_reversion": {
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
            "bollinger": {"period": 20, "std_dev": 2}, "require_both_signals": True,
        },
        "mtf_macd_elder": {
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "exit": {"trailing_stop_pct": 0.03, "atr_stop_mult": 2.0, "min_hold_bars": 1},
            "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
        },
    },
}

# ─── Phase 1: TRAIN — Sweep hyperparameters on train set only ───

print("\n=== PHASE 1: Training (60% of data) ===")

# Parameter grid — reduced for speed
lambdas = [3.0, 5.0]
confidence_thresholds = [0.55, 0.60]
max_depths = [3, 4]
learning_rates = [0.02, 0.05]
n_estimators_list = [100, 200]

combos = list(itertools.product(lambdas, confidence_thresholds, max_depths, learning_rates, n_estimators_list))

best_val_sharpe = -999
best_config = None
train_results = []

print(f"Testing {len(combos)} combinations on TRAIN set only...")
for lam, conf, depth, lr, nest in combos:
    config = base_config.copy()
    config["backtest"] = {"walk_forward_folds": 5, "min_train_fraction": 0.33}
    config["strategies"]["xgboost_cost_aware"] = {
        "model_params": {
            "n_estimators": nest, "max_depth": depth, "learning_rate": lr,
            "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 2.0, "reg_lambda": 3.0, "random_state": 42,
        },
        "training": {"retrain_every_candles": 500, "min_train_samples": 1000},
        "cost_filter": {"lambda": lam, "transaction_cost_bps": 30},
        "trading": {"confidence_threshold": conf, "allow_shorts": True},
    }

    try:
        engine = BacktestEngine(config)
        result = engine.run_walk_forward(train_data, XGBoostCostAware, data_1d=train_1d)
        m = result.metrics
        sharpe = m.get("sharpe_ratio", -999)
        pnl = m.get("total_pnl", 0)
        trades = m.get("total_trades", 0)
        dd = m.get("max_drawdown_pct", 100)

        if trades < 5:
            continue

        train_results.append((lam, conf, depth, lr, nest, sharpe, pnl, trades, dd))
    except Exception:
        continue

# ─── Phase 2: VALIDATION — Evaluate top params on val set ───

print("\n=== PHASE 2: Validation (20% of data) ===")

train_results.sort(key=lambda x: -x[5])  # Sort by Sharpe
top_n = min(10, len(train_results))
val_results = []

for i, (lam, conf, depth, lr, nest, train_sharpe, train_pnl, train_trades, train_dd) in enumerate(train_results[:top_n]):
    config = base_config.copy()
    config["backtest"] = {"walk_forward_folds": 4, "min_train_fraction": 0.33}
    config["strategies"]["xgboost_cost_aware"] = {
        "model_params": {
            "n_estimators": nest, "max_depth": depth, "learning_rate": lr,
            "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 2.0, "reg_lambda": 3.0, "random_state": 42,
        },
        "training": {"retrain_every_candles": 500, "min_train_samples": 1000},
        "cost_filter": {"lambda": lam, "transaction_cost_bps": 30},
        "trading": {"confidence_threshold": conf, "allow_shorts": True},
    }

    try:
        engine = BacktestEngine(config)
        result = engine.run_walk_forward(val_data, XGBoostCostAware, data_1d=val_1d)
        m = result.metrics
        val_sharpe = m.get("sharpe_ratio", -999)
        val_pnl = m.get("total_pnl", 0)
        val_trades = m.get("total_trades", 0)
        val_dd = m.get("max_drawdown_pct", 100)

        val_results.append((lam, conf, depth, lr, nest,
                           train_sharpe, val_sharpe, train_pnl, val_pnl,
                           train_trades, val_trades, train_dd, val_dd))

        print(f"  #{i+1}: lam={lam} conf={conf} depth={depth} lr={lr} nest={nest}")
        print(f"       Train Sharpe={train_sharpe:.2f} PnL=${train_pnl:.0f} "
              f"-> Val Sharpe={val_sharpe:.2f} PnL=${val_pnl:.0f}")
    except Exception as e:
        print(f"  #{i+1}: FAILED {e}")

# Select best on validation
if val_results:
    val_results.sort(key=lambda x: -x[6])  # Sort by val Sharpe
    best = val_results[0]
    lam, conf, depth, lr, nest = best[0], best[1], best[2], best[3], best[4]
    train_sh, val_sh = best[5], best[6]
    print(f"\nBest on validation: lam={lam} conf={conf} depth={depth} lr={lr} nest={nest}")
else:
    print("No validation results — using defaults")
    lam, conf, depth, lr, nest = 4.0, 0.55, 4, 0.03, 150

# ─── Phase 3: TEST — Final evaluation on untouched data ───

print("\n=== PHASE 3: Test (20% of data — NEVER seen before) ===")

config = base_config.copy()
config["backtest"] = {"walk_forward_folds": 4, "min_train_fraction": 0.33}
config["strategies"]["xgboost_cost_aware"] = {
    "model_params": {
        "n_estimators": nest, "max_depth": depth, "learning_rate": lr,
        "subsample": 0.7, "colsample_bytree": 0.6,
        "reg_alpha": 2.0, "reg_lambda": 3.0, "random_state": 42,
    },
    "training": {"retrain_every_candles": 500, "min_train_samples": 1000},
    "cost_filter": {"lambda": lam, "transaction_cost_bps": 30},
    "trading": {"confidence_threshold": conf, "allow_shorts": True},
}

engine = BacktestEngine(config)
result = engine.run_walk_forward(test_data, XGBoostCostAware, data_1d=test_1d)
m = result.metrics

print()
print("=" * 60)
print("  FINAL TEST RESULTS (untouched data)")
print("=" * 60)
print(f"  Trades:         {m.get('total_trades', 0):>6}")
print(f"  Win Rate:       {m.get('win_rate', 0):>6.1f}%")
print(f"  PnL:           ${m.get('total_pnl', 0):>10,.2f}")
print(f"  Return:         {m.get('total_return_pct', 0):>6.1f}%")
print(f"  Sharpe:         {m.get('sharpe_ratio', 0):>6.2f}")
print(f"  Sortino:        {str(m.get('sortino_ratio', 0)):>10}")
print(f"  Max DD:         {m.get('max_drawdown_pct', 0):>6.1f}%")
print(f"  Avg Bars:       {m.get('avg_bars_held', 0):>6.1f}")
print("-" * 60)

# ─── Overfitting Detection ──────────────────────────────────

test_sharpe = m.get("sharpe_ratio", 0)
test_dd = m.get("max_drawdown_pct", 0)
test_pnl = m.get("total_pnl", 0)
test_winrate = m.get("win_rate", 0)

warnings = []
if test_sharpe > 3:
    warnings.append(f"HIGH SHARPE ({test_sharpe:.1f} > 3) — likely overfit")
if test_dd < 10:
    warnings.append(f"LOW DRAWDOWN ({test_dd:.1f}% < 10%) — likely overfit")

# Train-val gap
if val_results:
    gap = abs(val_sh - test_sharpe) / max(abs(val_sh), 0.001)
    if gap > 0.5:
        warnings.append(f"LARGE TRAIN-TEST GAP ({gap:.1%}) — overfitting detected")
    print(f"  Train Sharpe:   {train_sh:.2f} -> Val: {val_sh:.2f} -> Test: {test_sharpe:.2f}")

if warnings:
    print()
    print("  ⚠️  OVERFITTING WARNINGS:")
    for w in warnings:
        print(f"     - {w}")
else:
    print()
    print("  ✓ No overfitting detected")

# Buy-hold comparison
start_p = test_data["close"].iloc[0]
end_p = test_data["close"].iloc[-1]
bh_return = (end_p / start_p - 1) * 100
bh_pnl = 10000 * (end_p / start_p - 1)
print()
print(f"  Buy-hold:       {bh_return:+.1f}% (${bh_pnl:+.0f})")
print(f"  Alpha vs BH:    {m.get('total_return_pct', 0) - bh_return:+.1f}%")

print("-" * 60)
print(f"  Exit reasons: {m.get('exit_reasons', {})}")
print("=" * 60)
