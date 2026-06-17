#!/usr/bin/env python
"""Run ensemble backtests for all configured symbols using settings.yaml overrides."""
import sys
import os
import time
import yaml
import copy
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine
from strategies.mtf_macd import MTF_MACD_Elder
from strategies.mean_reversion import MeanReversion

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

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

def get_symbol_config(global_config: dict, symbol: str) -> dict:
    cfg = copy.deepcopy(global_config)
    cfg["exchange"]["symbols"] = [symbol]
    
    overrides = global_config.get("symbols", {}).get(symbol, {})
    if not overrides:
        return cfg
        
    def merge_dicts(dict1, dict2):
        for k, v in dict2.items():
            if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
                merge_dicts(dict1[k], v)
            else:
                dict1[k] = v
                
    merge_dicts(cfg, overrides)
    return cfg

def main():
    # Load configuration
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        global_config = yaml.safe_load(f)
        
    symbols = global_config.get("exchange", {}).get("symbols", [])
    print(f"Configured symbols: {symbols}\n")
    
    symbol_file_map = {
        "BTC/USDT:USDT": "btc_1h_2020_2026.parquet",
        "ETH/USDT:USDT": "eth_1h_2020_2026.parquet",
        "XRP/USDT:USDT": "xrp_1h_2020_2026.parquet",
        "SOL/USDT:USDT": "sol_1h_2020_2026.parquet",
        "LTC/USDT:USDT": "ltc_1h_2020_2026.parquet"
    }
    
    ensemble_strategies = {
        "mtf_macd": MTF_MACD_Elder,
        "mean_reversion": MeanReversion,
    }
    
    all_metrics = []
    
    for symbol in symbols:
        filename = symbol_file_map.get(symbol)
        if not filename:
            print(f"Skipping {symbol}: no mapping to parquet cache.")
            continue
            
        filepath = CACHE_DIR / filename
        if not filepath.exists():
            print(f"Skipping {symbol}: cache file {filepath} not found.")
            continue
            
        print(f"==================================================")
        print(f"Backtesting {symbol} from cache {filename}...")
        print(f"==================================================")
        
        df_1h = pd.read_parquet(filepath)
        df_1d = _resample_to_1d(df_1h)
        
        n = len(df_1h)
        n_train = int(n * 0.60)
        n_val = int(n * 0.20)
        
        train = df_1h.iloc[:n_train]
        val = df_1h.iloc[n_train:n_train + n_val]
        test = df_1h.iloc[n_train + n_val:]
        
        cut1 = train["timestamp"].iloc[-1]
        cut2 = val["timestamp"].iloc[-1]
        
        train_1d = df_1d[df_1d["timestamp"] <= cut1]
        val_1d = df_1d[(df_1d["timestamp"] > cut1) & (df_1d["timestamp"] <= cut2)]
        test_1d = df_1d[df_1d["timestamp"] > cut2]
        
        sd = datetime.fromtimestamp(train["timestamp"].iloc[0]/1000, tz=timezone.utc)
        ed = datetime.fromtimestamp(test["timestamp"].iloc[-1]/1000, tz=timezone.utc)
        print(f"Data range: {sd.date()} to {ed.date()} ({(ed-sd).days/365:.1f} years)")
        print(f"Train size: {len(train):,} | Val size: {len(val):,} | Test size: {len(test):,}")
        
        # Get symbol specific merged config
        symbol_cfg = get_symbol_config(global_config, symbol)
        
        # For fair backtest comparison, use $10,000 capital and 20% position sizing
        # (matching the setup used in the original optimized tests that yielded Sharpe 2.71 for BTC)
        symbol_cfg["risk"]["initial_capital"] = 10000.0
        symbol_cfg["risk"]["max_position_pct"] = 0.20  # 20% per trade — standard backtest mode
        
        # CRITICAL: Override regime section with correct key names that the backtest engine reads.
        # settings.yaml uses different key names (adx_threshold, lookback_candles, atr_multiplier)
        # vs what the backtest regime classifier actually expects (adx_min, lookback_bars, atr_mult, etc.)
        # This mismatch caused 3x fewer trades (regime classifier fell back to wrong defaults).
        symbol_cfg["regime"] = {
            "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
            "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
            "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
            "hysteresis_bars": 2, "lookback_bars": 100,
        }
        # Check active MACD settings
        macd_opts = symbol_cfg["strategies"]["mtf_macd_elder"]
        print(f"Strategy: MTF_MACD_Elder with {macd_opts['macd']['fast']}/{macd_opts['macd']['slow']} MACD")
        print(f"Exits: trailing={macd_opts['exit'].get('trailing_stop_pct')}, atr_mult={macd_opts['exit'].get('atr_stop_mult')}")
        print(f"Shorts: {macd_opts.get('elder_filter', {}).get('allow_shorts', True)}")
        print()
        
        for name, data, d1, folds in [("TRAIN", train, train_1d, 15), ("VAL", val, val_1d, 8), ("TEST", test, test_1d, 8)]:
            symbol_cfg["backtest"] = {
                "walk_forward_folds": folds,
                "min_train_fraction": 0.33,
                "min_signal_exit_bars": 6  # Engine-level hold — always 6 bars (6h on H1 data)
            }
            
            engine = BacktestEngine(symbol_cfg)
            t0 = time.perf_counter()
            result = engine.run_ensemble_backtest(data, ensemble_strategies, data_1d=d1)
            m = result.metrics
            elapsed = time.perf_counter() - t0
            
            bh_start = data["close"].iloc[0]
            bh_end = data["close"].iloc[-1]
            bh_ret = (bh_end / bh_start - 1) * 100
            
            print(f"  {name:5s}: {m['total_trades']:>4} trades | WR={m['win_rate']:>5.1f}% | "
                  f"PnL=${m['total_pnl']:>8,.0f} ({m['total_return_pct']:>+5.1f}%) | "
                  f"Sharpe={m['sharpe_ratio']:>+6.2f} | DD={m['max_drawdown_pct']:>4.1f}% | "
                  f"BH={bh_ret:+.1f}% | {elapsed:.0f}s")
                  
            all_metrics.append({
                "Symbol": symbol.split("/")[0],
                "Split": name,
                "Trades": m["total_trades"],
                "WinRate": f"{m['win_rate']:.1f}%",
                "PnL": f"${m['total_pnl']:+,.0f}",
                "Return": f"{m['total_return_pct']:+.1f}%",
                "Sharpe": f"{m['sharpe_ratio']:+.2f}",
                "Drawdown": f"{m['max_drawdown_pct']:.1f}%",
                "BH": f"{bh_ret:+.1f}%"
            })
        print()
        
    # Print summary table
    df_metrics = pd.DataFrame(all_metrics)
    print("=========================================================================")
    print("                           SUMMARY TABLE                                 ")
    print("=========================================================================")
    print(df_metrics.to_string(index=False))
    print("=========================================================================")

if __name__ == "__main__":
    main()
