"""Meta-Labeling Diagnostics: 3 tests for overfitting detection.

Test 1 - Walk-forward OOS: Train on data up to last 3 months, test on unseen 3 months
Test 2 - Rejected signal analysis: Compare WR/PnL of accepted vs rejected trades
Test 3 - Confidence calibration: Histogram of prediction confidence scores

Based on: Problemy.md analysis
"""

import os, sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics
from strategies.mtf_macd import MTF_MACD_Elder
from strategies.meta_labeling import MetaLabeler
from research.meta_labeling_optimized import add_signal_features, load_symbol_data, run_backtest_for_trades, BASE_CONFIG, SYMBOLS_AVAILABLE


def fmt_metrics(m, label=""):
    return (f"{label}: {m['total_trades']} trades | PnL=${m['total_pnl']:+,.0f} | "
            f"Sharpe={m['sharpe_ratio']:.2f} | WR={m['win_rate']:.1f}% | "
            f"PF={m.get('profit_factor', 0) or 0:.2f} | DD={m['max_drawdown_pct']:.1f}%")


# ─── Test 1: Walk-forward OOS (last 3 months) ──────────────────

def test_walk_forward_oos(symbols=("BTC", "ETH")):
    """Train on all data except last 3 months. Test on last 3 months ONLY. No retraining."""
    print("=" * 72)
    print("  TEST 1: Walk-Forward OOS (last 3 months, no retraining)")
    print("=" * 72)

    all_train_trades = []
    all_test_trades = []
    three_months_ms = 90 * 86400_000

    for sym in symbols:
        df_1h, df_1d = load_symbol_data(sym)
        last_ts = df_1h["timestamp"].iloc[-1]
        cutoff_ts = last_ts - three_months_ms

        train_1h = df_1h[df_1h["timestamp"] < cutoff_ts].reset_index(drop=True)
        test_1h = df_1h[df_1h["timestamp"] >= cutoff_ts].reset_index(drop=True)

        train_start = pd.to_datetime(train_1h["timestamp"].iloc[0], unit="ms").date()
        train_end = pd.to_datetime(train_1h["timestamp"].iloc[-1], unit="ms").date()
        test_start = pd.to_datetime(test_1h["timestamp"].iloc[0], unit="ms").date()
        test_end = pd.to_datetime(test_1h["timestamp"].iloc[-1], unit="ms").date()
        print(f"\n  {sym}: TRAIN {train_start}->{train_end} ({len(train_1h):,} bars) | "
              f"TEST {test_start}->{test_end} ({len(test_1h):,} bars)")

        # Generate training signals
        train_t, train_m = run_backtest_for_trades(train_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        print(f"  {sym} train signals: {len(train_t)} ({train_m['total_trades']} trades, "
              f"WR={train_m['win_rate']:.1f}%)")
        all_train_trades.extend(train_t)

        # Generate test signals
        test_t, test_m = run_backtest_for_trades(test_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        print(f"  {sym} test signals:  {len(test_t)} ({test_m['total_trades']} trades, "
              f"WR={test_m['win_rate']:.1f}%)")
        all_test_trades.extend(test_t)

    # Train on all pre-cutoff data
    print(f"\n  Training MetaLabeler on {len(all_train_trades)} signals...")
    labeler = MetaLabeler(BASE_CONFIG)
    # Use conservative params from v2 (not over-optimized)
    labeler._model_params.update({
        "max_depth": 3, "n_estimators": 100, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 3.0, "reg_lambda": 4.0, "min_child_weight": 3,
    })
    ok = labeler.train(all_train_trades)
    if not ok:
        print("  Training failed!")
        return

    diag = labeler.get_diagnostics()
    print(f"  Trained: {diag['training_samples']} samples, acc={diag['val_accuracy']:.3f}")

    # Apply to TEST trades (NO retraining!)
    from strategies.base import Signal
    accepted, rejected = [], []
    confidences = []
    for t in all_test_trades:
        feats = t.get("features_at_signal")
        if feats and isinstance(feats, dict):
            sig = Signal.LONG if t.get("side") == "long" else Signal.SHORT
            # Get raw confidence
            try:
                prob = labeler._predict(feats)
                confidences.append(prob)
            except Exception:
                prob = 0.5
                confidences.append(0.5)
            if labeler.evaluate(sig, feats):
                accepted.append(t)
            else:
                rejected.append(t)
        else:
            accepted.append(t)
            confidences.append(0.5)

    acc_m = calculate_metrics(accepted, initial_capital=10000) if accepted else {"total_trades":0,"total_pnl":0,"sharpe_ratio":0,"win_rate":0}
    rej_m = calculate_metrics(rejected, initial_capital=10000) if rejected else {"total_trades":0,"total_pnl":0,"sharpe_ratio":0,"win_rate":0}
    all_m = calculate_metrics(all_test_trades, initial_capital=10000)

    print(f"\n  OOS Results (last 3 months, NO retraining):")
    print(f"  {'':<20} {'Accepted':<16} {'Rejected':<16} {'All':<16}")
    print(f"  {'-'*66}")
    print(f"  {'Trades':<20} {acc_m['total_trades']:<16} {rej_m['total_trades']:<16} {all_m['total_trades']:<16}")
    print(f"  {'Win Rate':<20} {acc_m['win_rate']:<15.1f}% {rej_m['win_rate']:<15.1f}% {all_m['win_rate']:<15.1f}%")
    print(f"  {'PnL':<20} ${acc_m['total_pnl']:<15,.0f} ${rej_m['total_pnl']:<15,.0f} ${all_m['total_pnl']:<15,.0f}")
    print(f"  {'Sharpe':<20} {acc_m['sharpe_ratio']:<16.2f} {rej_m['sharpe_ratio']:<16.2f} {all_m['sharpe_ratio']:<16.2f}")

    # Verdict
    wr_accepted = acc_m['win_rate']
    wr_rejected = rej_m['win_rate']
    wr_baseline = all_m['win_rate']
    total_test = len(all_test_trades)
    total_acc = len(accepted)
    rejection_pct = (len(rejected) / total_test * 100) if total_test else 0

    print(f"\n  Diagnostics:")
    print(f"  - Rejection rate: {rejection_pct:.0f}% ({len(rejected)}/{total_test})")
    print(f"  - WR accepted: {wr_accepted:.1f}%")
    print(f"  - WR rejected: {wr_rejected:.1f}%")
    print(f"  - WR baseline: {wr_baseline:.1f}%")
    print(f"  - WR delta (acc - rej): {wr_accepted - wr_rejected:+.1f}pp")

    if wr_accepted > 60 and wr_rejected < wr_baseline:
        print(f"  [OK] Model discriminates: accepted WR ({wr_accepted:.0f}%) > baseline ({wr_baseline:.0f}%) > rejected ({wr_rejected:.0f}%)")
    elif wr_accepted > wr_baseline + 5:
        print(f"  [OK] Model improves over baseline (WR +{wr_accepted-wr_baseline:.1f}pp)")
    elif abs(wr_accepted - wr_baseline) < 3:
        print(f"  [!!] Model does NOT discriminate — accepted WR ≈ baseline (within 3pp)")
    else:
        print(f"  [--] Inconclusive — need more data")

    # Confidence stats
    if confidences:
        conf_arr = np.array(confidences)
        print(f"\n  Confidence distribution:")
        print(f"  - Mean: {conf_arr.mean():.3f}")
        print(f"  - Std:  {conf_arr.std():.3f}")
        print(f"  - <0.45: {(conf_arr < 0.45).mean()*100:.0f}%")
        print(f"  - 0.45-0.55: {((conf_arr >= 0.45) & (conf_arr <= 0.55)).mean()*100:.0f}%")
        print(f"  - >0.55: {(conf_arr > 0.55).mean()*100:.0f}%")
        bimodal = (conf_arr < 0.45).mean() > 0.15 and (conf_arr > 0.55).mean() > 0.15
        if bimodal:
            print(f"  [OK] Bimodal distribution — model is confident")
        elif ((conf_arr >= 0.45) & (conf_arr <= 0.55)).mean() > 0.5:
            print(f"  [!!] Clustered around 0.5 — model is uncertain")
        else:
            print(f"  [--] Mixed distribution")

    return {"accepted_wr": wr_accepted, "rejected_wr": wr_rejected, "baseline_wr": wr_baseline,
            "rejection_pct": rejection_pct, "oos_sharpe": acc_m['sharpe_ratio']}


# ─── Test 2: Rejected signal analysis ──────────────────────────

def test_rejected_signals(symbols=("BTC", "ETH", "XRP")):
    """Analyze WR and PnL of rejected vs accepted signals."""
    print("\n" + "=" * 72)
    print("  TEST 2: Rejected Signal Analysis")
    print("=" * 72)

    # Use 60/20/20 split, train on TRAIN, analyze on VAL
    all_train, all_val = [], []
    for sym in symbols:
        df_1h, df_1d = load_symbol_data(sym)
        n = len(df_1h)
        n_train = int(n * 0.60)
        n_val = int(n * 0.20)
        train_1h = df_1h.iloc[:n_train].reset_index(drop=True)
        val_1h = df_1h.iloc[n_train:n_train + n_val].reset_index(drop=True)
        train_t, _ = run_backtest_for_trades(train_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        val_t, val_m = run_backtest_for_trades(val_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        all_train.extend(train_t)
        all_val.extend(val_t)
        print(f"  {sym}: {len(train_t)} train + {len(val_t)} val (baseline WR={val_m['win_rate']:.1f}%)")

    # Train MetaLabeler with conservative params
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["meta_labeling"]["min_confidence"] = 0.55  # Conservative
    labeler = MetaLabeler(cfg)
    labeler._model_params.update({
        "max_depth": 3, "n_estimators": 100, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 3.0, "reg_lambda": 4.0, "min_child_weight": 3,
    })
    labeler.train(all_train)

    from strategies.base import Signal
    accepted, rejected = [], []
    for t in all_val:
        feats = t.get("features_at_signal")
        if feats and isinstance(feats, dict):
            sig = Signal.LONG if t.get("side") == "long" else Signal.SHORT
            if labeler.evaluate(sig, feats):
                accepted.append(t)
            else:
                rejected.append(t)
        else:
            accepted.append(t)

    acc_m = calculate_metrics(accepted, initial_capital=10000) if accepted else {"total_trades":0,"total_pnl":0,"sharpe_ratio":0,"win_rate":0}
    rej_m = calculate_metrics(rejected, initial_capital=10000) if rejected else {"total_trades":0,"total_pnl":0,"sharpe_ratio":0,"win_rate":0}
    base_m = calculate_metrics(all_val, initial_capital=10000)

    print(f"\n  Signal Quality Analysis:")
    print(f"  {'Group':<15} {'Trades':<8} {'PnL':<12} {'WR':<8} {'Sharpe':<8} {'PF':<8}")
    print(f"  {'-'*59}")
    print(f"  {'Accepted':<15} {acc_m['total_trades']:<8} ${acc_m['total_pnl']:<11,.0f} "
          f"{acc_m['win_rate']:<7.1f}% {acc_m['sharpe_ratio']:<8.2f} {acc_m.get('profit_factor',0) or 0:<8.2f}")
    print(f"  {'Rejected':<15} {rej_m['total_trades']:<8} ${rej_m['total_pnl']:<11,.0f} "
          f"{rej_m['win_rate']:<7.1f}% {rej_m['sharpe_ratio']:<8.2f} {rej_m.get('profit_factor',0) or 0:<8.2f}")
    print(f"  {'All (baseline)':<15} {base_m['total_trades']:<8} ${base_m['total_pnl']:<11,.0f} "
          f"{base_m['win_rate']:<7.1f}% {base_m['sharpe_ratio']:<8.2f} {base_m.get('profit_factor',0) or 0:<8.2f}")

    wr_a, wr_r, wr_b = acc_m['win_rate'], rej_m['win_rate'], base_m['win_rate']
    print(f"\n  Verdict:")
    if wr_r < wr_b - 5:
        print(f"  [OK] Rejected WR ({wr_r:.0f}%) << baseline ({wr_b:.0f}%) — model GENUINELY filters bad signals")
    elif abs(wr_r - wr_b) < 3:
        print(f"  [!!] Rejected WR ({wr_r:.0f}%) ~= baseline ({wr_b:.0f}%) — model filters RANDOMLY (overfit)")
    else:
        print(f"  [--] Rejected WR ({wr_r:.0f}%) moderately below baseline ({wr_b:.0f}%)")

    if acc_m['sharpe_ratio'] > base_m['sharpe_ratio'] + 0.5:
        print(f"  [OK] Accepted Sharpe ({acc_m['sharpe_ratio']:.2f}) > baseline ({base_m['sharpe_ratio']:.2f})")
    else:
        print(f"  [!!] Accepted Sharpe not significantly better")

    return {"acc_wr": wr_a, "rej_wr": wr_r, "base_wr": wr_b,
            "discriminates": wr_r < wr_b - 5}


# ─── Test 3: Confidence calibration ────────────────────────────

def test_confidence_calibration(symbols=("BTC", "ETH", "XRP")):
    """Analyze confidence score distribution of the MetaLabeler."""
    print("\n" + "=" * 72)
    print("  TEST 3: Confidence Calibration")
    print("=" * 72)

    all_train, all_test = [], []
    for sym in symbols:
        df_1h, df_1d = load_symbol_data(sym)
        n = len(df_1h)
        n_train = int(n * 0.70)
        train_1h = df_1h.iloc[:n_train].reset_index(drop=True)
        test_1h = df_1h.iloc[n_train:].reset_index(drop=True)
        train_t, _ = run_backtest_for_trades(train_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        test_t, _ = run_backtest_for_trades(test_1h, df_1d, BASE_CONFIG, MTF_MACD_Elder)
        all_train.extend(train_t)
        all_test.extend(test_t)

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["meta_labeling"]["min_confidence"] = 0.50
    labeler = MetaLabeler(cfg)
    labeler._model_params.update({
        "max_depth": 3, "n_estimators": 100, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 3.0, "reg_lambda": 4.0, "min_child_weight": 3,
    })
    if not labeler.train(all_train):
        print("  Training failed!")
        return

    # Collect confidence scores for test trades
    scores = []
    outcomes = []  # 1=win, 0=loss
    for t in all_test:
        feats = t.get("features_at_signal")
        if feats and isinstance(feats, dict):
            try:
                prob = labeler._predict(feats)
                scores.append(prob)
                outcomes.append(1.0 if t.get("pnl", 0) > 0 else 0.0)
            except Exception:
                pass

    if not scores:
        print("  No confidence scores collected!")
        return

    scores = np.array(scores)
    outcomes = np.array(outcomes)

    # Binning analysis
    bins = [(0, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 1.0)]
    print(f"\n  Confidence Bins ({len(scores)} test trades):")
    print(f"  {'Bin':<15} {'Count':<8} {'%':<6} {'WR':<8} {'Interpretation':<20}")
    print(f"  {'-'*57}")
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi)
        n = mask.sum()
        if n > 0:
            wr = outcomes[mask].mean() * 100
            interp = ("Strong SHORT" if hi <= 0.45 else "Weak SHORT" if hi <= 0.50
                      else "Weak LONG" if lo >= 0.50 else "Neutral")
            print(f"  [{lo:.2f}-{hi:.2f}){'':>5} {n:<8} {n/len(scores)*100:<5.1f}% {wr:<7.1f}% {interp:<20}")
        else:
            print(f"  [{lo:.2f}-{hi:.2f}){'':>5} {0:<8} {0:<5.1f}% {'-':<8}")

    # Bimodality check
    low_conf = (scores < 0.40).mean()
    mid_conf = ((scores >= 0.45) & (scores <= 0.55)).mean()
    high_conf = (scores > 0.60).mean()

    print(f"\n  Distribution shape:")
    print(f"  - <0.40: {low_conf*100:.0f}% (very low confidence)")
    print(f"  - 0.40-0.60: {(1-low_conf-high_conf)*100:.0f}% (moderate)")
    print(f"  - >0.60: {high_conf*100:.0f}% (very high confidence)")
    print(f"  - Clustered near 0.5: {mid_conf*100:.0f}%")

    if low_conf > 0.10 and high_conf > 0.10:
        print(f"  [OK] Bimodal — model makes confident predictions at both extremes")
    elif mid_conf > 0.60:
        print(f"  [!!] Central clustering — model is uncertain, scores near 0.5 dominate")
    elif high_conf > 0.30:
        print(f"  [OK] Right-skewed — model is often confident in LONG direction")
    else:
        print(f"  [--] Mixed distribution")

    # Calibration: does higher confidence = higher WR?
    if len(scores) > 20:
        sorted_idx = np.argsort(scores)
        bottom_20 = outcomes[sorted_idx[:len(scores)//5]].mean() * 100
        top_20 = outcomes[sorted_idx[-len(scores)//5:]].mean() * 100
        print(f"\n  Calibration check:")
        print(f"  - Bottom 20% confidence WR: {bottom_20:.1f}%")
        print(f"  - Top 20% confidence WR:    {top_20:.1f}%")
        if top_20 > bottom_20 + 10:
            print(f"  [OK] Well-calibrated: higher confidence -> higher WR (+{top_20-bottom_20:.0f}pp)")
        elif top_20 > bottom_20:
            print(f"  [--] Weakly calibrated: higher confidence -> slightly higher WR (+{top_20-bottom_20:.0f}pp)")
        else:
            print(f"  [!!] NOT calibrated: higher confidence does NOT predict higher WR")


# ─── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("META-LABELING DIAGNOSTICS: 3 Overfitting Tests")
    print("Based on: Problemy.md analysis\n")

    results = {}
    results["oos"] = test_walk_forward_oos(symbols=("BTC", "ETH"))
    results["rejected"] = test_rejected_signals(symbols=("BTC", "ETH"))
    test_confidence_calibration(symbols=("BTC", "ETH"))

    print("\n" + "=" * 72)
    print("  FINAL VERDICT")
    print("=" * 72)

    oos = results.get("oos", {})
    rej = results.get("rejected", {})

    checks = []
    if oos.get("accepted_wr", 0) > oos.get("baseline_wr", 0) + 5:
        checks.append(("[OK]", f"OOS WR improves baseline by {oos['accepted_wr']-oos['baseline_wr']:.1f}pp"))
    else:
        checks.append(("[!!]", f"OOS WR does NOT significantly improve baseline"))

    if rej.get("discriminates", False):
        checks.append(("[OK]", f"Rejected WR ({rej['rej_wr']:.0f}%) << baseline ({rej['base_wr']:.0f}%)"))
    else:
        checks.append(("[!!]", f"Rejected WR ({rej['rej_wr']:.0f}%) near baseline ({rej['base_wr']:.0f}%)"))

    for status, msg in checks:
        print(f"  {status} {msg}")

    passed = sum(1 for s, _ in checks if s == "[OK]")
    if passed == 2:
        print(f"\n  [OK] ALL TESTS PASSED — MetaLabeler has real edge. Deploy with confidence.")
    elif passed == 1:
        print(f"\n  [--] MIXED RESULTS — Deploy with min_confidence >= 0.55 and monitor closely.")
    else:
        print(f"\n  [!!] ALL TESTS FAILED — MetaLabeler likely overfit. Do NOT deploy to production.\n"
              f"      Revert to pure MTF_MACD until more training data is available.")

    print("\n" + "=" * 72)
