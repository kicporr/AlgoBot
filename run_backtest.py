#!/usr/bin/env python
"""Run a walk-forward backtest on real BTC/USDT data from Bitget.

Usage:
    python run_backtest.py --days 90
    python run_backtest.py --since 2024-01-01
    python run_backtest.py --cache data/cache/btc_1h.parquet
    python run_backtest.py --days 365 --trailing-stop 0.04 --folds 27
"""

import sys
import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "config" / ".env")

from data.ingestion.rest_client import BitgetRESTClient
from data.ingestion.resampler import OHLCVResampler, Timeframe
from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics, fold_stability_report
from strategies.mtf_macd import MTF_MACD_Elder

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resample_1h_to_1d(df_1h):
    """Resample 1H candles to 1D (min 12/24 hours)."""
    df = df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("dt")
    daily = df.resample("1D", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    daily["bar_count"] = df.resample("1D").size()
    daily = daily[daily["bar_count"] >= 12]
    daily = daily.dropna().reset_index()
    daily["timestamp"] = daily["dt"].astype("int64") // 1_000_000
    daily.drop(columns=["dt"], inplace=True)
    return daily


def fetch_or_load_data(client, days, since=None):
    cache_1h = CACHE_DIR / f"btc_1h_d{days}.parquet"
    cache_1d = CACHE_DIR / f"btc_1d_d{days}.parquet"

    if cache_1h.exists() and cache_1d.exists():
        print(f"[cache] Loading ({days}d)")
        return pd.read_parquet(cache_1h), pd.read_parquet(cache_1d)

    print(f"[fetch] Fetching {days}d of BTC/USDT 1H from Bitget...")
    t0 = time.perf_counter()

    if since:
        df_1h = client.fetch_since(timeframe="1h", since_date=since)
    else:
        df_1h = client.fetch_days(timeframe="1h", days=days)

    elapsed = time.perf_counter() - t0
    print(f"   {len(df_1h):,} candles in {elapsed:.1f}s")

    df_1d = _resample_1h_to_1d(df_1h)
    print(f"   Resampled to {len(df_1d)} 1D candles")

    df_1h.to_parquet(cache_1h)
    df_1d.to_parquet(cache_1d)
    print(f"   Cached to {cache_1h}")

    return df_1h, df_1d


def build_config(args):
    return {
        "exchange": {
            "name": "bitget", "symbols": ["BTC/USDT"],
            "fees": {"taker": args.taker_fee, "maker": args.maker_fee, "slippage": args.slippage},
            "rate_limit": {"max_requests_per_second": 10},
        },
        "risk": {"initial_capital": args.capital, "max_position_pct": 0.95},
        "backtest": {
            "walk_forward_folds": args.folds,
            "min_train_fraction": 0.33,
            "min_signal_exit_bars": 6
        },
        "strategies": {"mtf_macd_elder": {
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "exit": {"trailing_stop_pct": args.trailing_stop, "atr_stop_mult": args.atr_stop, "min_hold_bars": 1},
            "elder_filter": {"require_volume_confirm": args.volume_filter, "volume_mult": 1.2, "allow_shorts": args.shorts},
        }, "mean_reversion": {
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
            "bollinger": {"period": 20, "std_dev": 2},
            "require_both_signals": True,
        }},
        "regime": {
            "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
            "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
            "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
            "hysteresis_bars": 2, "lookback_bars": 100,
        },
        "features": {"max_window_bars": 500, "min_bars_required": 50},
    }


def print_results(result, df_1h, args):
    m = result.metrics
    sd = datetime.fromtimestamp(df_1h["timestamp"].iloc[0]/1000, tz=timezone.utc)
    ed = datetime.fromtimestamp(df_1h["timestamp"].iloc[-1]/1000, tz=timezone.utc)
    days = (ed - sd).days

    print()
    print("=" * 62)
    print("  BACKTEST RESULTS")
    print("=" * 62)
    print(f"  Period:       {sd.date()} -> {ed.date()} ({days:,} days)")
    print(f"  Bars:         {len(df_1h):,} 1H candles")
    print(f"  Params:       {args.folds}-fold | trail={args.trailing_stop:.0%} | ATR={args.atr_stop}x | shorts={args.shorts}")
    print("-" * 62)
    print(f"  TOTAL TRADES:        {m.get('total_trades', 0):>6}")
    print(f"  WIN RATE:            {m.get('win_rate', 0):>6.1f}%")
    print(f"  TOTAL PNL:          ${m.get('total_pnl', 0):>10,.2f}")
    print(f"  TOTAL RETURN:        {m.get('total_return_pct', 0):>6.1f}%")
    print(f"  ANNUALIZED:          {m.get('annualized_return_pct', 0):>6.1f}%")
    print("-" * 62)
    print(f"  SHARPE:              {m.get('sharpe_ratio', 0):>6.2f}")
    print(f"  SORTINO:             {str(m.get('sortino_ratio', 0)):>6}")
    print(f"  CALMAR:              {str(m.get('calmar_ratio', 0)):>6}")
    print(f"  MAX DD:              {m.get('max_drawdown_pct', 0):>6.1f}%")
    print(f"  MAX DD BARS:         {m.get('max_drawdown_duration_bars', 0):>6}")
    print("-" * 62)
    print(f"  AVG WIN:            ${m.get('avg_win', 0):>10,.2f}")
    print(f"  AVG LOSS:           ${m.get('avg_loss', 0):>10,.2f}")
    print(f"  W/L RATIO:           {str(m.get('win_loss_ratio', 0)):>6}")
    print(f"  PROFIT FACTOR:       {str(m.get('profit_factor', 0)):>6}")
    print(f"  EXPECTANCY:         ${m.get('expectancy', 0):>10,.2f}")
    print(f"  AVG BARS HELD:       {m.get('avg_bars_held', 0):>6.1f}")
    print(f"  MAX CONS LOSSES:     {m.get('max_consecutive_losses', 0):>6}")
    print("-" * 62)

    reasons = m.get("exit_reasons", {})
    if reasons:
        print("  EXIT BREAKDOWN:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / m["total_trades"] * 100 if m["total_trades"] else 0
            print(f"    {reason:20s} {count:>4} ({pct:5.1f}%)")
        print("-" * 62)

    if result.fold_metrics:
        stability = fold_stability_report(result.fold_metrics)
        if stability:
            print("  FOLD STABILITY (mean +/- std):")
            for key, stats in stability.items():
                print(f"    {key:20s} {stats['mean']:>8.4f} +/- {stats['std']:>8.4f}")

    if result.equity_curve:
        start = result.equity_curve[0]
        end = result.equity_curve[-1]
        peak = max(result.equity_curve)
        print("-" * 62)
        print(f"  EQUITY:  ${start:,.2f} -> ${end:,.2f}  (peak: ${peak:,.2f})")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(description="bocik backtest")
    dg = parser.add_mutually_exclusive_group(required=True)
    dg.add_argument("--days", type=int, help="Days of recent data")
    dg.add_argument("--since", type=str, help="Fetch since date (YYYY-MM-DD)")
    dg.add_argument("--cache", type=str, help="Path to cached 1H parquet")
    parser.add_argument("--trailing-stop", type=float, default=0.03)
    parser.add_argument("--atr-stop", type=float, default=2.0)
    parser.add_argument("--volume-filter", action="store_true", default=False)
    parser.add_argument("--shorts", action="store_true", default=False)
    parser.add_argument("--folds", type=int, default=27)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--taker-fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--save-trades", type=str, default=None)
    parser.add_argument("--save-equity", type=str, default=None)
    args = parser.parse_args()

    config = build_config(args)

    if args.cache:
        print(f"[cache] Loading from {args.cache}")
        df_1h = pd.read_parquet(args.cache)
        df_1d = OHLCVResampler.resample_bulk(df_1h, Timeframe.D1)
    else:
        client = BitgetRESTClient(config)
        if not client.is_connected():
            print("ERROR: Cannot connect to Bitget API")
            sys.exit(1)
        df_1h, df_1d = fetch_or_load_data(client, args.days or 365, args.since)

    if len(df_1h) < 200:
        print(f"ERROR: Need >=200 bars, got {len(df_1h)}")
        sys.exit(1)

    print(f"\n[bt] Running walk-forward backtest...")
    t0 = time.perf_counter()
    engine = BacktestEngine(config)
    result = engine.run_walk_forward(df_1h, MTF_MACD_Elder, data_1d=df_1d)
    elapsed = time.perf_counter() - t0
    print(f"   Done in {elapsed:.1f}s")

    print_results(result, df_1h, args)

    if args.save_trades and result.trades:
        pd.DataFrame(result.trades).to_csv(args.save_trades, index=False)
        print(f"\n[save] Trades -> {args.save_trades}")
    if args.save_equity and result.equity_curve:
        pd.DataFrame({"step": range(len(result.equity_curve)), "equity": result.equity_curve}).to_csv(args.save_equity, index=False)
        print(f"[save] Equity -> {args.save_equity}")


if __name__ == "__main__":
    main()
