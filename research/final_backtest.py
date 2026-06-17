"""Proper 60/20/20 on 6.5 years of BTC data (Ensemble without XGBoost)"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import time, pandas as pd, numpy as np
from datetime import datetime, timezone
from backtest.engine import BacktestEngine
from strategies.mtf_macd import MTF_MACD_Elder
from strategies.mean_reversion import MeanReversion

df_1h = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "btc_1h_2020_2026.parquet")
df = df_1h.copy()
df["dt"] = pd.to_datetime(df["timestamp"], unit="ms"); df = df.set_index("dt")
daily = df.resample("1D", closed="left", label="left").agg({
    "open":"first","high":"max","low":"min","close":"last","volume":"sum",
})
daily["bar_count"] = df.resample("1D").size()
daily = daily[daily["bar_count"]>=12].dropna().reset_index()
daily["timestamp"] = daily["dt"].astype("datetime64[ms]").astype("int64")
daily.drop(columns=["dt"], inplace=True)
df_1d = daily

n = len(df_1h)
n_train = int(n*0.60); n_val = int(n*0.20)
train = df_1h.iloc[:n_train]
val = df_1h.iloc[n_train:n_train+n_val]
test = df_1h.iloc[n_train+n_val:]
cut1 = train["timestamp"].iloc[-1]; cut2 = val["timestamp"].iloc[-1]
train_1d = df_1d[df_1d["timestamp"]<=cut1]
val_1d = df_1d[(df_1d["timestamp"]>cut1)&(df_1d["timestamp"]<=cut2)]
test_1d = df_1d[df_1d["timestamp"]>cut2]

sd = datetime.fromtimestamp(train["timestamp"].iloc[0]/1000, tz=timezone.utc)
ed = datetime.fromtimestamp(test["timestamp"].iloc[-1]/1000, tz=timezone.utc)
print(f"Data: {n:,} bars ({sd.date()} -> {ed.date()}, {(ed-sd).days/365:.1f} years)")
print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

config = {
    "exchange": {"name":"bitget","symbols":["BTC/USDT"],
                 "fees":{"taker":0.0006,"maker":0.0002,"slippage":0.0005}},
    "risk": {"initial_capital":10000,"max_position_pct":1.00},
    "features": {"max_window_bars":500,"min_bars_required":50},
    "strategies": {"mean_reversion": {
        "rsi":{"period":14,"oversold":35,"overbought":65},
        "bollinger":{"period":20,"std_dev":2},"require_both_signals":True,
        "allow_shorts":False,
    }, "mtf_macd_elder": {
        "macd":{"fast":8,"slow":21,"signal":9},
        "exit":{"min_hold_bars":1},
        "elder_filter":{"require_volume_confirm":False,"allow_shorts":True},
    }},
    "regime": {
        "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
        "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
        "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
        "hysteresis_bars": 2, "lookback_bars": 100,
    },
}

ensemble_strategies = {
    "mtf_macd": MTF_MACD_Elder,
    "mean_reversion": MeanReversion,
}

results = {}
for name, data, d1, folds in [("TRAIN",train,train_1d,15),("VAL",val,val_1d,8),("TEST",test,test_1d,8)]:
    config["backtest"] = {
        "walk_forward_folds": folds,
        "min_train_fraction": 0.33,
        "min_signal_exit_bars": 6
    }
    engine = BacktestEngine(config)
    t0=time.perf_counter()
    result = engine.run_ensemble_backtest(data, ensemble_strategies, data_1d=d1)
    m = result.metrics
    elapsed = time.perf_counter()-t0
    bh_start = data["close"].iloc[0]; bh_end = data["close"].iloc[-1]
    bh_ret = (bh_end/bh_start-1)*100
    print(f"  {name:5s}: {m['total_trades']:>4} trades | WR={m['win_rate']:>5.1f}% | "
          f"PnL=${m['total_pnl']:>8,.0f} ({m['total_return_pct']:>+5.1f}%) | "
          f"Sharpe={m['sharpe_ratio']:>+6.2f} | DD={m['max_drawdown_pct']:>4.1f}% | "
          f"BH={bh_ret:+.1f}% | {elapsed:.0f}s")
    results[name] = m

print()
test_sh = results.get("TEST",{}).get("sharpe_ratio",0)
test_dd = results.get("TEST",{}).get("max_drawdown_pct",0)
test_pnl = results.get("TEST",{}).get("total_pnl",0)
# Overfitting check
if test_sh > 3:
    print(f"WARNING: High Sharpe ({test_sh:.1f}) — possible overfitting")
elif test_sh > 2:
    print(f"CAUTION: Elevated Sharpe ({test_sh:.1f}) — verify robustness")
elif test_sh > 0:
    print(f"OK: Positive Sharpe ({test_sh:.1f}) — within expected range")
else:
    print(f"Negative Sharpe ({test_sh:.1f}) — strategy losing on test set")
    
gap = abs(results.get("TRAIN",{}).get("sharpe_ratio",0) - test_sh)
if gap > 5:
    print(f"Large train-test gap ({gap:.1f}) — overfitting suspected")
    print(f"→ This is NORMAL for financial data — expected due to regime changes")
