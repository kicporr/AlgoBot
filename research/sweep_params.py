"""Parameter sweep for MTF MACD strategy on cached BTC data"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import time, itertools
import pandas as pd

from backtest.engine import BacktestEngine
from strategies.mtf_macd import MTF_MACD_Elder

def _resample(df):
    df2 = df.copy()
    df2["dt"] = pd.to_datetime(df2["timestamp"], unit="ms")
    df2 = df2.set_index("dt")
    daily = df2.resample("1D", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    daily["bar_count"] = df2.resample("1D").size()
    daily = daily[daily["bar_count"] >= 12].dropna().reset_index()
    daily["timestamp"] = daily["dt"].astype("datetime64[ms]").astype("int64")
    daily.drop(columns=["dt"], inplace=True)
    return daily

# Load cached data
df_1h = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "btc_1h_2020_2026.parquet")
df_1d = _resample(df_1h)

base_config = {
    "exchange": {"name": "bitget", "symbols": ["BTC/USDT"],
                 "fees": {"taker": 0.0006, "maker": 0.0002, "slippage": 0.0005}},
    "risk": {"initial_capital": 10000, "max_position_pct": 0.95},
    "backtest": {"walk_forward_folds": 9, "min_train_fraction": 0.33},
    "features": {"max_window_bars": 500, "min_bars_required": 50},
    "regime": {
        "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
        "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
        "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
        "hysteresis_bars": 2, "lookback_bars": 100,
    },
    "strategies": {"mean_reversion": {
        "rsi": {"period": 14, "oversold": 30, "overbought": 70},
        "bollinger": {"period": 20, "std_dev": 2}, "require_both_signals": True,
    }},
}

# Parameters to sweep
trail_stops = [0.02, 0.03, 0.04, 0.05, 0.06]
atr_stops = [1.5, 2.0, 2.5, 3.0]
shorts_mode = [False]
volume_filter = [False]
fast_periods = [8, 12]
slow_periods = [21, 26]

combos = list(itertools.product(trail_stops, atr_stops, shorts_mode, volume_filter, fast_periods, slow_periods))
# Filter: fast < slow
combos = [(ts, atr, sh, vf, fast, slow) for ts, atr, sh, vf, fast, slow in combos if fast < slow]

print(f"Testing {len(combos)} parameter combinations...")
print(f"{'Trail':>5} {'ATR':>5} {'Fast':>5} {'Slow':>5} | {'Trades':>7} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'Return':>7}")
print("-" * 75)

best_sharpe = -999
best_params = None
results = []

for ts, atr, sh, vf, fast, slow in combos:
    config = base_config.copy()
    config["strategies"] = {
        **base_config["strategies"],
        "mtf_macd_elder": {
            "macd": {"fast": fast, "slow": slow, "signal": 9},
            "exit": {"trailing_stop_pct": ts, "atr_stop_mult": atr, "min_hold_bars": 1},
            "elder_filter": {"require_volume_confirm": vf, "volume_mult": 1.2, "allow_shorts": sh},
        },
    }
    
    try:
        engine = BacktestEngine(config)
        result = engine.run_walk_forward(df_1h, MTF_MACD_Elder, data_1d=df_1d)
        m = result.metrics
        trades = m.get("total_trades", 0)
        wr = m.get("win_rate", 0)
        pnl = m.get("total_pnl", 0)
        sharpe = m.get("sharpe_ratio", 0)
        ret = m.get("total_return_pct", 0)
        
        # Only show meaningful results
        if trades >= 3:
            results.append((ts, atr, fast, slow, trades, wr, pnl, sharpe, ret))
            marker = " ***" if sharpe > best_sharpe else ""
            print(f"{ts:>5.0%} {atr:>5.1f} {fast:>5} {slow:>5} | {trades:>7} {wr:>5.1f}% ${pnl:>9.2f} {sharpe:>7.2f} {ret:>6.1f}%{marker}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (ts, atr, fast, slow)
    except Exception as e:
        pass

print("-" * 75)
if best_params:
    print(f"\nBest: trail={best_params[0]:.0%}, ATR={best_params[1]:.1f}x, "
          f"MACD({best_params[2]},{best_params[3]},9)")
    print(f"Sharpe: {best_sharpe:.2f}")

# Top 5 by Sharpe
results.sort(key=lambda x: -x[7])
print("\nTop 5 by Sharpe:")
for r in results[:5]:
    print(f"  trail={r[0]:.0%} atr={r[1]:.1f}x fast={r[2]} slow={r[3]} | "
          f"{r[4]} trades, WR={r[5]:.1f}%, PnL=${r[6]:.0f}, Sharpe={r[7]:.2f}")
