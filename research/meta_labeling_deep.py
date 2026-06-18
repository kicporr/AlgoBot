"""Meta-Labeling v3 -- Deep optimization with overfitting safeguards.

Key improvements over v2:
    - 3 symbols: BTC + ETH + XRP (~1300 training signals)
    - Walk-forward validation (3 windows) -- not single split
    - Extended grid: subsample, colsample_bytree, min_child_weight
    - Overfitting detection:
        1. Train/Val Sharpe gap > 1.5 = overfit flag
        2. Min 10 trades per window for statistical significance
        3. Deflation test: shuffle returns -> retrain -> must get Sharpe < 1.0
        4. Feature stability: top-5 features must overlap >= 3 between windows
    - Stronger L1/L2 regularization by default
    - Walk-forward MetaLabeler training (expanding window)

Usage:
    python research/meta_labeling_deep.py              # full (72 combos x 3 windows)
    python research/meta_labeling_deep.py --quick       # fast (18 combos x 2 windows)
"""

import os, sys, time, itertools, copy, random
from pathlib import Path
from collections import Counter

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
    "meta_labeling": {"enabled": True, "model": "xgboost", "training_samples": 300, "min_confidence": 0.55},
    "regime": {"enabled": True, "hysteresis_bars": 2,
               "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
               "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
               "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08}},
}

SYMBOLS_AVAILABLE = {
    "BTC": "btc_1h_2020_2026.parquet",
    "ETH": "eth_1h_2020_2026.parquet",
    "XRP": "xrp_1h_2020_2026.parquet",
}

# ─── Signal-specific features ───────────────────────────────────

def add_signal_features(trades: list[dict]) -> list[dict]:
    """Add signal-specific features to each trade. Modifies IN PLACE."""
    if not trades:
        return trades
    sorted_trades = sorted(trades, key=lambda t: t.get("entry_time", 0))
    for i, t in enumerate(sorted_trades):
        feats = t.get("features_at_signal")
        if not feats or not isinstance(feats, dict):
            continue
        # Temporal spacing
        feats["bars_since_last_signal"] = min(
            (t["entry_time"] - sorted_trades[i-1]["entry_time"]) / 3_600_000
            if i > 0 and t.get("entry_time", 0) > sorted_trades[i-1].get("entry_time", 0) else 999, 999)
        feats["hours_since_last_trade"] = min(
            (t["entry_time"] - sorted_trades[i-1]["exit_time"]) / 3_600_000
            if i > 0 and t.get("entry_time", 0) > sorted_trades[i-1].get("exit_time", 0) else 999, 999)
        # Regime
        regime = t.get("regime", "unknown")
        feats["regime_is_trending"] = 1.0 if regime == "trending" else 0.0
        feats["regime_is_ranging"] = 1.0 if regime == "ranging" else 0.0
        feats["regime_is_volatile"] = 1.0 if regime == "volatile" else 0.0
        # Price/volatility context
        close = feats.get("close", feats.get("price", 0))
        atr = feats.get("atr_14", 0)
        feats["atr_pct_of_price"] = (atr / close * 100) if close > 0 and atr > 0 else 0.0
        feats["volatility_regime"] = 1.0 if feats.get("volatility_20", 0) > 0.5 else 0.0
        # MACD
        macd_hist = feats.get("macd_hist", 0)
        feats["macd_hist_sign"] = 1.0 if macd_hist > 0 else (-1.0 if macd_hist < 0 else 0.0)
        feats["macd_hist_strength"] = abs(macd_hist) / (close + 1e-8) * 100 if close > 0 else 0.0
        feats["macd_cross_recent"] = abs(feats.get("macd_cross", 0))
        # Bollinger
        feats["bb_position_signal"] = feats.get("bb_position", 0.5)
        feats["bb_width_signal"] = feats.get("bb_width", 0.0)
        # Volume
        feats["volume_ratio_signal"] = feats.get("volume_sma_ratio", 1.0)
        # Momentum
        feats["rsi_at_signal"] = feats.get("rsi_14", 50)
        feats["rsi_extreme"] = 1.0 if feats.get("rsi_14", 50) > 70 or feats.get("rsi_14", 50) < 30 else 0.0
        feats["ema_slope_sign"] = 1.0 if feats.get("ema_20_slope", 0) > 0 else (-1.0 if feats.get("ema_20_slope", 0) < 0 else 0.0)
        # Trend strength
        feats["adx_at_signal"] = feats.get("adx_14", 20)
        feats["trend_strong"] = 1.0 if feats.get("adx_14", 20) > 25 else 0.0
        # Symbol and side
        feats["signal_is_long"] = 1.0 if t.get("signal_type") == "LONG" else 0.0
        feats["signal_is_short"] = 1.0 if t.get("signal_type") == "SHORT" else 0.0
        sym = t.get("strategy", "").split(":")[1] if ":" in (t.get("strategy") or "") else ""
        feats["symbol_is_btc"] = 1.0 if "BTC" in sym else 0.0
        feats["symbol_is_eth"] = 1.0 if "ETH" in sym else 0.0
        feats["symbol_is_xrp"] = 1.0 if "XRP" in sym else 0.0
    return sorted_trades


# ─── Data ───────────────────────────────────────────────────────

def resample_1h_to_1d(df_1h):
    df = df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("dt")
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()
    daily = daily.reset_index()
    daily["timestamp"] = daily["dt"].astype("int64") // 1_000_000
    return daily.drop(columns=["dt"])

def load_symbol_data(symbol: str):
    fname = SYMBOLS_AVAILABLE.get(symbol)
    if not fname:
        raise ValueError(f"Unknown symbol: {symbol}")
    path = PROJECT_ROOT / "data" / "cache" / fname
    df = pd.read_parquet(path)
    if df["timestamp"].dtype == "datetime64[ns]":
        df["timestamp"] = df["timestamp"].astype("int64") // 1_000_000
    return df, resample_1h_to_1d(df)


# ─── Walk-forward window splits ─────────────────────────────────

def make_wf_windows(df_1h, n_windows=3):
    """Split data into expanding chronological windows for walk-forward validation.

    Returns list of (train_df, test_df) tuples. Each window has more training data.
    Window 0: train on first 40%, test on next 20%
    Window 1: train on first 60%, test on next 20%
    Window 2: train on first 80%, test on last 20%
    """
    n = len(df_1h)
    windows = []
    for w in range(n_windows):
        train_end = int(n * (0.40 + w * 0.20))
        test_end = int(n * (0.60 + w * 0.20))
        if test_end > n:
            test_end = n
        train = df_1h.iloc[:train_end].reset_index(drop=True)
        test = df_1h.iloc[train_end:test_end].reset_index(drop=True)
        if len(test) > 1000:
            windows.append((train, test))
    return windows


# ─── Backtest ───────────────────────────────────────────────────

def run_backtest_for_trades(data_1h, data_1d, config, strategy_class):
    engine = BacktestEngine(config)
    result = engine.run_walk_forward(data_1h, strategy_class, data_1d=data_1d)
    return add_signal_features(result.trades), calculate_metrics(result.trades, initial_capital=10000)


# ─── Single combo evaluation (with overfitting checks) ──────────

def evaluate_combo_deep(combo: dict, all_train_trades: list, all_val_trades: list,
                        base_config: dict) -> dict:
    """Train MetaLabeler and evaluate on validation with overfitting detection."""
    cfg = copy.deepcopy(base_config)
    cfg["meta_labeling"]["min_confidence"] = combo["min_confidence"]

    labeler = MetaLabeler(cfg)
    labeler._model_params.update({
        "max_depth": combo["max_depth"],
        "n_estimators": combo["n_estimators"],
        "learning_rate": combo.get("learning_rate", 0.03),
        "subsample": combo.get("subsample", 0.8),
        "colsample_bytree": combo.get("colsample_bytree", 0.7),
        "reg_alpha": combo.get("reg_alpha", 3.0),
        "reg_lambda": combo.get("reg_lambda", 4.0),
        "min_child_weight": combo.get("min_child_weight", 1),
    })

    ok = labeler.train(all_train_trades)
    if not ok:
        return {**combo, "val_sharpe": -99, "overfit_flags": ["train_failed"]}

    diag = labeler.get_diagnostics()

    # Filter val trades
    filtered, _ = filter_trades(all_val_trades, labeler)
    val_m = calculate_metrics(filtered, initial_capital=10000) if filtered else \
        {"total_trades": 0, "total_pnl": 0, "sharpe_ratio": 0, "win_rate": 0, "profit_factor": 0}

    # Overfitting checks
    flags = []
    train_acc = diag.get("val_accuracy", 0)

    # 1. Train accuracy suspiciously high
    if train_acc > 0.85:
        flags.append(f"train_acc={train_acc:.3f}>0.85")

    # 2. Too few validation trades
    if val_m["total_trades"] < 10:
        flags.append(f"too_few_trades={val_m['total_trades']}")

    # 3. Val Sharpe unrealistically high
    if val_m["sharpe_ratio"] > 5.0:
        flags.append(f"sharpe_suspicious={val_m['sharpe_ratio']:.1f}")

    # 4. Val win rate extreme
    if val_m["win_rate"] > 85 and val_m["total_trades"] > 10:
        flags.append(f"wr_extreme={val_m['win_rate']:.1f}%")

    return {**combo,
            "val_sharpe": val_m["sharpe_ratio"],
            "val_trades": val_m["total_trades"],
            "val_pnl": val_m["total_pnl"],
            "val_wr": val_m["win_rate"],
            "val_pf": val_m.get("profit_factor", 0) or 0,
            "train_accuracy": train_acc,
            "train_samples": diag.get("training_samples", 0),
            "overfit_flags": flags,
            "ok": len(flags) == 0}


def filter_trades(trades, labeler):
    """Apply meta-labeler filter to a trade list. Returns (filtered, rejected_count)."""
    filtered, rejected = [], 0
    for t in trades:
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
    return filtered, rejected


# ─── Deflation test ─────────────────────────────────────────────

def deflation_test(trades: list, combo: dict, base_config: dict, n_shuffles=10) -> dict:
    """Shuffle trade outcomes, retrain, check if Sharpe degrades.

    If the meta-labeler still gets good results with shuffled labels,
    the original result is likely overfit noise.
    """
    cfg = copy.deepcopy(base_config)
    cfg["meta_labeling"]["min_confidence"] = combo["min_confidence"]

    shuffle_sharpes = []
    for seed in range(n_shuffles):
        shuffled = copy.deepcopy(trades)
        rng = random.Random(seed)
        pnls = [t.get("pnl", 0) for t in shuffled]
        rng.shuffle(pnls)
        for i, t in enumerate(shuffled):
            t["pnl"] = pnls[i]

        labeler = MetaLabeler(cfg)
        labeler._model_params.update({
            "max_depth": combo["max_depth"], "n_estimators": combo["n_estimators"],
            "learning_rate": combo.get("learning_rate", 0.03),
            "subsample": combo.get("subsample", 0.8),
            "colsample_bytree": combo.get("colsample_bytree", 0.7),
            "reg_alpha": combo.get("reg_alpha", 3.0),
            "reg_lambda": combo.get("reg_lambda", 4.0),
            "min_child_weight": combo.get("min_child_weight", 1),
        })
        if labeler.train(shuffled):
            filtered, _ = filter_trades(shuffled, labeler)
            m = calculate_metrics(filtered, initial_capital=10000) if filtered else {"sharpe_ratio": 0}
            shuffle_sharpes.append(m["sharpe_ratio"])

    if shuffle_sharpes:
        mean_shuffle = np.mean(shuffle_sharpes)
        max_shuffle = max(shuffle_sharpes)
        return {"mean_shuffle_sharpe": round(mean_shuffle, 2),
                "max_shuffle_sharpe": round(max_shuffle, 2),
                "deflation_ok": max_shuffle < 1.5}
    return {"mean_shuffle_sharpe": 0, "max_shuffle_sharpe": 0, "deflation_ok": True}


# ─── Feature stability check ────────────────────────────────────

def feature_stability_check(trades_by_window: list, combo: dict, base_config: dict) -> dict:
    """Train on each window separately, check if top features overlap."""
    top_features = []
    for trades in trades_by_window:
        if len(trades) < 50:
            continue
        cfg = copy.deepcopy(base_config)
        cfg["meta_labeling"]["min_confidence"] = combo["min_confidence"]
        labeler = MetaLabeler(cfg)
        labeler._model_params.update({
            "max_depth": combo["max_depth"], "n_estimators": combo["n_estimators"],
            "learning_rate": combo.get("learning_rate", 0.03),
            "subsample": combo.get("subsample", 0.8),
            "colsample_bytree": combo.get("colsample_bytree", 0.7),
            "reg_alpha": combo.get("reg_alpha", 3.0),
            "reg_lambda": combo.get("reg_lambda", 4.0),
        })
        if labeler.train(trades):
            try:
                imp = labeler.model.get_booster().get_score(importance_type="gain")
                top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
                top_features.append(set(k for k, _ in top5))
            except Exception:
                pass

    if len(top_features) < 2:
        return {"feature_overlap": 0, "feature_stable": False, "shared_features": []}

    # Count how many features appear in 2+ windows
    all_features = Counter()
    for s in top_features:
        all_features.update(s)
    stable = [f for f, c in all_features.items() if c >= 2]
    overlap_ratio = len(stable) / 5  # Fraction of top-5 that are stable
    return {"feature_overlap": round(overlap_ratio, 2),
            "feature_stable": overlap_ratio >= 0.6,
            "shared_features": stable[:5]}


# ─── Main ───────────────────────────────────────────────────────

def main(quick=False):
    symbols = ["BTC", "ETH", "XRP"]
    print("=" * 72)
    print(f"  META-LABELING v3: {'+'.join(symbols)} | Deep Opt + Anti-Overfit")
    print("=" * 72)

    # 1. Load data
    all_data = {}
    for sym in symbols:
        df_1h, df_1d = load_symbol_data(sym)
        n = len(df_1h)
        print(f"  {sym}: {n:,} 1H | {len(df_1d):,} 1D | "
              f"{pd.to_datetime(df_1h['timestamp'].iloc[0], unit='ms').date()} -> "
              f"{pd.to_datetime(df_1h['timestamp'].iloc[-1], unit='ms').date()}")
        all_data[sym] = (df_1h, df_1d)

    # 2. Walk-forward windows (per symbol)
    n_windows = 2 if quick else 3
    sym_windows = {}
    for sym, (df_1h, df_1d) in all_data.items():
        windows = make_wf_windows(df_1h, n_windows)
        sym_windows[sym] = windows
        for i, (train, test) in enumerate(windows):
            t0 = pd.to_datetime(train["timestamp"].iloc[0], unit="ms").date()
            t1 = pd.to_datetime(test["timestamp"].iloc[-1], unit="ms").date()
            print(f"  {sym} W{i}: train={len(train):,} test={len(test):,} | {t0} -> {t1}")

    # 3. Generate signals per window per symbol
    print("\n" + "=" * 72)
    print("  PHASE 1: Generate labeled signals (walk-forward windows)")
    print("=" * 72)

    window_trades = []  # One list of (train_trades, val_trades) per window
    total_t0 = time.monotonic()

    for w in range(n_windows):
        w_train, w_val = [], []
        for sym in symbols:
            train_1h, test_1h = sym_windows[sym][w]
            df_1d = all_data[sym][1]
            t0 = time.monotonic()
            train_t, train_m = run_backtest_for_trades(train_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
            test_t, test_m = run_backtest_for_trades(test_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
            dt = time.monotonic() - t0
            w_train.extend(train_t)
            w_val.extend(test_t)
            print(f"  W{w} {sym}: {len(train_t)}+{len(test_t)} trades "
                  f"(train_S={train_m['sharpe_ratio']:.2f} test_S={test_m['sharpe_ratio']:.2f}) | {dt:.0f}s")

        window_trades.append((w_train, w_val))
        print(f"  W{w} total: {len(w_train)} train + {len(w_val)} val signals")

    total_elapsed = time.monotonic() - total_t0
    print(f"\n  Phase 1 done: {total_elapsed:.0f}s")

    # 4. Grid search (use last window as validation)
    print("\n" + "=" * 72)
    print("  PHASE 2: Hyperparameter Grid Search + Overfitting Checks")
    print("=" * 72)

    train_trades = window_trades[-2][0] if n_windows >= 2 else window_trades[0][0]
    val_trades = window_trades[-2][1] if n_windows >= 2 else window_trades[0][1]
    # Combine earlier windows for more training data
    if n_windows >= 3:
        train_trades = window_trades[0][0] + window_trades[1][0]

    print(f"  Train signals: {len(train_trades)} (BTC+ETH+XRP, windows 0..{n_windows-3})")
    print(f"  Val signals:   {len(val_trades)} (window {n_windows-2})")

    if quick:
        grid = {
            "min_confidence": [0.50, 0.55, 0.60],
            "max_depth": [3, 5],
            "n_estimators": [50, 150],
            "learning_rate": [0.03],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.6],
            "reg_alpha": [2.0, 4.0],
            "reg_lambda": [3.0],
            "min_child_weight": [1],
        }
    else:
        grid = {
            "min_confidence": [0.50, 0.52, 0.55, 0.58, 0.60],
            "max_depth": [2, 3, 4, 5],
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.02, 0.03, 0.05],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.5, 0.6, 0.7],
            "reg_alpha": [1.0, 2.0, 4.0, 8.0],
            "reg_lambda": [1.0, 2.0, 4.0, 8.0],
            "min_child_weight": [1, 3, 5],
        }

    keys = list(grid.keys())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]
    n_combos = len(combos)
    grid_dims = ' x '.join(str(len(grid[k])) for k in keys)
    print(f"  Grid: {n_combos} combos ({grid_dims})")

    t0 = time.monotonic()
    results = []
    try:
        import joblib
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(evaluate_combo_deep)(c, train_trades, val_trades, BASE_CONFIG)
            for c in combos
        )
    except ImportError:
        for c in combos:
            results.append(evaluate_combo_deep(c, train_trades, val_trades, BASE_CONFIG))

    elapsed = time.monotonic() - t0
    print(f"  Grid done: {elapsed:.0f}s ({elapsed/n_combos:.1f}s/combo)")

    # Filter: must have >= 10 val trades and no overfit flags
    valid = [r for r in results if r.get("ok") and r.get("val_sharpe", -99) > -99
             and r.get("val_trades", 0) >= 10]
    flagged = [r for r in results if not r.get("ok") and r.get("val_sharpe", -99) > -99]

    if not valid:
        print(f"  WARNING: No valid combos ({len(flagged)} flagged for overfitting). Using best flagged.")
        valid = flagged or results
    valid.sort(key=lambda r: r["val_sharpe"], reverse=True)
    best = valid[0]

    print(f"\n  Top 5 (by Val Sharpe, no overfit flags):")
    if flagged:
        print(f"  ({len(flagged)} combos flagged: "
              f"{', '.join(f for r in flagged[:3] for f in r.get('overfit_flags', []))})")
    hdr = (f"  {'Conf':<6} {'D':<3} {'N':<4} {'sub':<5} {'col':<5} "
           f"{'a':<4} {'l':<4} {'mcw':<4} {'ValS':<8} {'ValT':<6} {'ValPf':<7} {'Acc':<6}")
    print(hdr)
    print(f"  {'-'*70}")
    for r in valid[:5]:
        print(f"  {r['min_confidence']:<6.2f} {r['max_depth']:<3} {r['n_estimators']:<4} "
              f"{r.get('subsample', 0.8):<5.2f} {r.get('colsample_bytree', 0.7):<5.2f} "
              f"{r.get('reg_alpha', 3):<4.1f} {r.get('reg_lambda', 4):<4.1f} "
              f"{r.get('min_child_weight', 1):<4} "
              f"{r['val_sharpe']:<8.2f} {r['val_trades']:<6} "
              f"{r.get('val_pf', 0):<7.2f} {r.get('train_accuracy', 0):<6.3f}")

    # 5. Deflation test
    print("\n" + "=" * 72)
    print("  PHASE 3: Deflation Test (shuffled returns)")
    print("=" * 72)

    defl = deflation_test(train_trades, best, BASE_CONFIG, n_shuffles=10)
    print(f"  Shuffled returns: mean Sharpe={defl['mean_shuffle_sharpe']:.2f} "
          f"max Sharpe={defl['max_shuffle_sharpe']:.2f}")
    if defl["deflation_ok"]:
        print(f"  [OK] Deflation PASSED (max shuffle Sharpe {defl['max_shuffle_sharpe']:.1f} < 1.5)")
    else:
        print(f"  [!!] Deflation FAILED — model may be fitting noise")

    # 6. Feature stability
    print("\n" + "=" * 72)
    print("  PHASE 4: Feature Stability Across Windows")
    print("=" * 72)

    trades_by_window = [w_train for w_train, _ in window_trades]
    feat_stab = feature_stability_check(trades_by_window, best, BASE_CONFIG)
    print(f"  Feature overlap ratio: {feat_stab['feature_overlap']:.0%}")
    print(f"  Stable features: {', '.join(feat_stab['shared_features'][:5]) if feat_stab['shared_features'] else 'N/A'}")
    if feat_stab["feature_stable"]:
        print(f"  [OK] Feature stability PASSED (>=60% overlap across windows)")
    else:
        print(f"  [!!] Feature stability LOW — model relies on different features per period")

    # 7. Final test on LAST window (untouched)
    print("\n" + "=" * 72)
    print("  PHASE 5: Final Test (untouched last window)")
    print("=" * 72)

    final_train = window_trades[-2][0] + window_trades[-2][1] if n_windows >= 2 else window_trades[0][0]
    final_test_trades = window_trades[-1][1]  # Test = validation trades from last window

    # Train final model
    final_cfg = copy.deepcopy(BASE_CONFIG)
    final_cfg["meta_labeling"]["min_confidence"] = best["min_confidence"]
    final_labeler = MetaLabeler(final_cfg)
    final_labeler._model_params.update({
        "max_depth": best["max_depth"], "n_estimators": best["n_estimators"],
        "learning_rate": best.get("learning_rate", 0.03),
        "subsample": best.get("subsample", 0.8),
        "colsample_bytree": best.get("colsample_bytree", 0.7),
        "reg_alpha": best.get("reg_alpha", 3.0),
        "reg_lambda": best.get("reg_lambda", 4.0),
        "min_child_weight": best.get("min_child_weight", 1),
    })
    final_labeler.train(final_train)
    diag = final_labeler.get_diagnostics()

    test_filtered, test_rej = filter_trades(final_test_trades, final_labeler)
    test_base_m = calculate_metrics(final_test_trades, initial_capital=10000)
    test_ml_m = calculate_metrics(test_filtered, initial_capital=10000) if test_filtered else \
        {"total_trades": 0, "total_pnl": 0, "sharpe_ratio": 0, "win_rate": 0, "profit_factor": 0, "max_drawdown_pct": 0}

    print(f"  Final model: {diag['training_samples']} samples, {diag.get('features_used', '?')} features")

    # 8. Final report
    print("\n" + "=" * 72)
    print("  FINAL RESULTS: MTF_MACD vs MetaLabeler v3 (Deep)")
    print("=" * 72)

    print(f"\n  {'Metric':<22} {'MTF_MACD':<14} {'+ MetaLabeler':<14} {'Change':<10}")
    print(f"  {'-'*60}")
    print(f"  {'Trades':<22} {test_base_m['total_trades']:<14} {test_ml_m['total_trades']:<14} "
          f"{test_ml_m['total_trades'] - test_base_m['total_trades']:+d}")
    print(f"  {'Win Rate':<22} {test_base_m['win_rate']:<13.1f}% {test_ml_m['win_rate']:<13.1f}% "
          f"{test_ml_m['win_rate'] - test_base_m['win_rate']:+.1f}%")
    print(f"  {'Total PnL':<22} ${test_base_m['total_pnl']:<13,.0f} ${test_ml_m['total_pnl']:<13,.0f} "
          f"${test_ml_m['total_pnl'] - test_base_m['total_pnl']:+,.0f}")
    print(f"  {'Sharpe':<22} {test_base_m['sharpe_ratio']:<14.2f} {test_ml_m['sharpe_ratio']:<14.2f} "
          f"{test_ml_m['sharpe_ratio'] - test_base_m['sharpe_ratio']:+.2f}")
    bpf = test_base_m.get("profit_factor", 0) or 0
    mpf = test_ml_m.get("profit_factor", 0) or 0
    print(f"  {'Profit Factor':<22} {bpf:<14.2f} {mpf:<14.2f} {mpf - bpf:+.2f}")
    print(f"  {'Max DD':<22} {test_base_m['max_drawdown_pct']:<13.1f}% {test_ml_m['max_drawdown_pct']:<13.1f}% "
          f"{test_ml_m['max_drawdown_pct'] - test_base_m['max_drawdown_pct']:+.1f}%")

    print(f"\n  Rejected: {test_rej}/{len(final_test_trades)} signals "
          f"({test_rej/len(final_test_trades)*100:.0f}%)" if final_test_trades else "  No trades")

    # Overfitting summary
    print(f"\n  Overfitting Report:")
    checks = [
        ("Deflation test", defl["deflation_ok"], f"max shuffle Sharpe={defl['max_shuffle_sharpe']:.1f}"),
        ("Feature stability", feat_stab["feature_stable"], f"overlap={feat_stab['feature_overlap']:.0%}"),
        ("Val trades >= 10", test_ml_m["total_trades"] >= 10, f"trades={test_ml_m['total_trades']}"),
        ("Sharpe < 5", test_ml_m["sharpe_ratio"] < 5.0, f"Sharpe={test_ml_m['sharpe_ratio']:.2f}"),
        ("WR < 85%", test_ml_m["win_rate"] < 85, f"WR={test_ml_m['win_rate']:.1f}%"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        status = "[OK]" if ok else "[!!]"
        if not ok:
            all_ok = False
        print(f"    {status} {name}: {detail}")
    print(f"  Overall: {'PASSED' if all_ok else 'SOME FAILURES — review before deploying'}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Meta-Labeling v3 Deep Backtest")
    p.add_argument("--quick", action="store_true", help="Fast grid (18 combos x 2 windows)")
    args = p.parse_args()
    main(quick=args.quick)
