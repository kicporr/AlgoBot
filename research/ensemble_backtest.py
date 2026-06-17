"""Ensemble backtest — 60/20/20 split with regime routing (Simplified without XGBoost)."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from backtest.engine import BacktestEngine
from strategies.mtf_macd import MTF_MACD_Elder
from strategies.mean_reversion import MeanReversion


def _resample_to_1d(df_1h):
    """Resample 1H candles to 1D."""
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


def print_header(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_metrics(name, m, bh_ret):
    trades = m.get("total_trades", 0)
    wr = m.get("win_rate", 0)
    pnl = m.get("total_pnl", 0)
    ret = m.get("total_return_pct", 0)
    sharpe = m.get("sharpe_ratio", 0)
    dd = m.get("max_drawdown_pct", 0)
    pf = m.get("profit_factor", 0)
    avg_pnl = m.get("avg_trade_pnl", 0)
    exit_reasons = m.get("exit_reasons", {})

    print(f"  {name:8s}: {trades:>4} trades | WR={wr:>5.1f}% | "
          f"PnL=${pnl:>8,.0f} ({ret:>+5.1f}%) | "
          f"Sharpe={sharpe:>+6.2f} | DD={dd:>4.1f}% | "
          f"PF={pf} | Avg=${avg_pnl:>+.0f} | BH={bh_ret:+.1f}%")

    if exit_reasons and trades > 0:
        parts = []
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / trades * 100
            parts.append(f"{reason}={count}({pct:.0f}%)")
        print(f"           Exits: {', '.join(parts)}")


def main():
    # Load data
    df_1h = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "btc_1h_2020_2026.parquet")
    df_1d = _resample_to_1d(df_1h)

    # 60/20/20 split
    n = len(df_1h)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    train = df_1h.iloc[:n_train]
    val = df_1h.iloc[n_train:n_train + n_val]
    test = df_1h.iloc[n_train + n_val:]

    # Split 1D data
    cut1 = train["timestamp"].iloc[-1]
    cut2 = val["timestamp"].iloc[-1]
    train_1d = df_1d[df_1d["timestamp"] <= cut1]
    val_1d = df_1d[(df_1d["timestamp"] > cut1) & (df_1d["timestamp"] <= cut2)]
    test_1d = df_1d[df_1d["timestamp"] > cut2]

    # Print data summary
    sd = datetime.fromtimestamp(train["timestamp"].iloc[0] / 1000, tz=timezone.utc)
    ed = datetime.fromtimestamp(test["timestamp"].iloc[-1] / 1000, tz=timezone.utc)
    print(f"Data: {n:,} bars ({sd.date()} -> {ed.date()}, {(ed - sd).days / 365:.1f} years)")
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Config — improved parameters
    config = {
        "exchange": {
            "name": "bitget", "symbols": ["BTC/USDT"],
            "fees": {"taker": 0.0006, "maker": 0.0002, "slippage": 0.0005},
        },
        "risk": {"initial_capital": 10000, "max_position_pct": 0.20},
        "features": {"max_window_bars": 500, "min_bars_required": 50},
        "strategies": {
            "mtf_macd_elder": {
                "macd": {"fast": 8, "slow": 21, "signal": 9},
                "exit": {"trailing_stop_pct": 0.02, "atr_stop_mult": 1.5, "min_hold_bars": 1},
                "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
            },
            "mean_reversion": {
                "rsi": {"period": 14, "oversold": 35, "overbought": 65},
                "bollinger": {"period": 20, "std_dev": 2},
                "require_both_signals": True,
                "allow_shorts": False,
            },
        },
        "regime": {
            "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
            "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
            "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
            "hysteresis_bars": 2, "lookback_bars": 100,
        },
    }

    # Strategy classes for ensemble
    ensemble_strategies = {
        "mtf_macd": MTF_MACD_Elder,
        "mean_reversion": MeanReversion,
    }

    # ── Run backtests ───────────────────────────────────────────
    splits = [
        ("TRAIN", train, train_1d, 15),
        ("VAL", val, val_1d, 8),
        ("TEST", test, test_1d, 8),
    ]

    print_header("ENSEMBLE: Regime-Routed (TREND->MACD, RANGE->MR, VOL->FLAT, UNCLEAR->FLAT)")
    ensemble_results = {}
    for name, data, d1, folds in splits:
        config["backtest"] = {
            "walk_forward_folds": folds,
            "min_train_fraction": 0.33,
            "min_signal_exit_bars": 6
        }
        engine = BacktestEngine(config)
        t0 = time.perf_counter()
        result = engine.run_ensemble_backtest(data, ensemble_strategies, data_1d=d1)
        elapsed = time.perf_counter() - t0

        bh_start = data["close"].iloc[0]
        bh_end = data["close"].iloc[-1]
        bh_ret = (bh_end / bh_start - 1) * 100

        print_metrics(name, result.metrics, bh_ret)
        ensemble_results[name] = result.metrics

        # Per-strategy breakdown
        if result.trades:
            strat_breakdown = {}
            for trade in result.trades:
                s = trade.get("strategy", "unknown")
                if s not in strat_breakdown:
                    strat_breakdown[s] = {"count": 0, "pnl": 0, "wins": 0, "gross_gains": 0, "gross_losses": 0, "exits": {}}
                stats = strat_breakdown[s]
                stats["count"] += 1
                pnl = trade.get("pnl", 0)
                stats["pnl"] += pnl
                if pnl > 0:
                    stats["wins"] += 1
                    stats["gross_gains"] += pnl
                else:
                    stats["gross_losses"] += abs(pnl)
                
                reason = trade.get("exit_reason", "unknown")
                stats["exits"][reason] = stats["exits"].get(reason, 0) + 1
            
            print(f"           Strategy breakdown:")
            for s, stats in sorted(strat_breakdown.items(), key=lambda x: -x[1]["pnl"]):
                wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
                pf = stats["gross_gains"] / stats["gross_losses"] if stats["gross_losses"] > 0 else float("inf")
                exits_str = ", ".join(f"{k}={v}" for k, v in sorted(stats["exits"].items(), key=lambda x: -x[1]))
                print(f"             {s:15s}: {stats['count']:>3} trades | WR={wr:.1f}% | PF={pf:.2f} | PnL=${stats['pnl']:>+8,.0f}\n"
                      f"                              Exits: {exits_str}")

    print()
    et = ensemble_results.get("TEST", {})
    test_pnl = et.get("total_pnl", 0)
    test_sharpe = et.get("sharpe_ratio", 0)
    if test_pnl > 0 and test_sharpe > 0:
        print("  [OK] Ensemble (without XGBoost) is PROFITABLE on test set!")
        print(f"  Sharpe on test: {test_sharpe:.2f} (target: >0.5)")
    else:
        print("  [CHECK] Ensemble (without XGBoost) failed to generate profit on test set")


if __name__ == "__main__":
    main()
