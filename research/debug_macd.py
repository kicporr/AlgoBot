from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine
from strategies.mtf_macd import MTF_MACD_Elder
from features.indicators import IndicatorCalculator

# Load data
df_1h = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "btc_1h_2020_2026.parquet")
def _resample(df2):
    df = df2.copy()
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

df_1d = _resample(df_1h)

config = {
    "exchange": {
        "name": "bitget", "symbols": ["BTC/USDT"],
        "fees": {"taker": 0.001, "maker": 0.001, "slippage": 0.0005},
    },
    "risk": {"initial_capital": 10000, "max_position_pct": 0.95},
    "backtest": {"walk_forward_folds": 8, "min_train_fraction": 0.33},
    "features": {"max_window_bars": 500, "min_bars_required": 50},
    "strategies": {
        "mtf_macd_elder": {
            "macd": {"fast": 8, "slow": 21, "signal": 9},
            "exit": {"trailing_stop_pct": 0.02, "atr_stop_mult": 1.5, "min_hold_bars": 1},
            "elder_filter": {"require_volume_confirm": False, "allow_shorts": True},
        },
    },
}

n = len(df_1h)
n_train = int(n * 0.60)
n_val = int(n * 0.20)
val_data = df_1h.iloc[n_train + n_val:]
cut2 = df_1h.iloc[:n_train + n_val]["timestamp"].iloc[-1]
val_1d = df_1d[df_1d["timestamp"] > cut2]

# Compute both D1 trend series
ic = IndicatorCalculator()
macd_fast, macd_slow, macd_sig = 8, 21, 9
macd_line, signal_line, macd_hist = ic.macd(val_1d["close"], macd_fast, macd_slow, macd_sig)

# 1) Current: MACD > Signal
trend_current = pd.Series("FLAT", index=val_1d.index)
trend_current[macd_line > signal_line] = "UP"
trend_current[macd_line < signal_line] = "DOWN"

# 2) Elder: slope of MACD histogram
trend_elder = pd.Series("FLAT", index=val_1d.index)
hist_slope = macd_hist.diff(1)
trend_elder[hist_slope > 0] = "UP"
trend_elder[hist_slope < 0] = "DOWN"

# Helper to expand to 1H resolution
def expand_to_1h(daily_trend, n_hourly_bars):
    trends_1h = pd.Series("FLAT", index=pd.RangeIndex(n_hourly_bars))
    for i in range(len(daily_trend)):
        start_h = i * 24
        end_h = min((i + 1) * 24, n_hourly_bars)
        trends_1h.iloc[start_h:end_h] = daily_trend.iloc[i]
    return trends_1h

d1_current = expand_to_1h(trend_current, len(val_data))
d1_elder = expand_to_1h(trend_elder, len(val_data))

# Run backtests
engine = BacktestEngine(config)

print("Running Backtest with current D1 filter (MACD > Signal)...")
res_current = engine.run_walk_forward(val_data, MTF_MACD_Elder, data_1d=val_1d)
mc = res_current.metrics
print(f"  Current: {mc['total_trades']} trades | WR={mc['win_rate']:.1f}% | PnL=${mc['total_pnl']:.2f} | Sharpe={mc['sharpe_ratio']:.2f}")

# Hack engine to use Elder D1 trends
# Let's override _compute_d1_trend_series method on this engine instance
def new_compute_d1(self, data_1d, n_hourly_bars):
    # Same logic but with histogram slope
    from features.indicators import IndicatorCalculator
    ic = IndicatorCalculator()
    macd_line, signal_line, macd_hist = ic.macd(data_1d["close"], 8, 21, 9)
    hist_slope = macd_hist.diff(1)
    daily_trend = pd.Series("FLAT", index=data_1d.index)
    daily_trend[hist_slope > 0] = "UP"
    daily_trend[hist_slope < 0] = "DOWN"
    
    trends_1h = pd.Series("FLAT", index=pd.RangeIndex(n_hourly_bars))
    for i in range(len(data_1d)):
        start_h = i * 24
        end_h = min((i + 1) * 24, n_hourly_bars)
        trends_1h.iloc[start_h:end_h] = daily_trend.iloc[i]
    return trends_1h

import types
engine._compute_d1_trend_series = types.MethodType(new_compute_d1, engine)

print("Running Backtest with Elder D1 filter (Histogram Slope)...")
res_elder = engine.run_walk_forward(val_data, MTF_MACD_Elder, data_1d=val_1d)
me = res_elder.metrics
print(f"  Elder: {me['total_trades']} trades | WR={me['win_rate']:.1f}% | PnL=${me['total_pnl']:.2f} | Sharpe={me['sharpe_ratio']:.2f}")
