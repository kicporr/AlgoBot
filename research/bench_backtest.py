"""Benchmark backtest performance at various data sizes."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import time, numpy as np, pandas as pd
from backtest.engine import BacktestEngine
from strategies.mtf_macd import MTF_MACD_Elder

config = {
    'exchange': {'fees': {'taker': 0.001, 'maker': 0.0005, 'slippage': 0.0005}},
    'risk': {'initial_capital': 10000, 'max_position_pct': 0.95},
    'backtest': {'walk_forward_folds': 27, 'min_train_fraction': 0.25},
    'strategies': {'mtf_macd_elder': {
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'exit': {'trailing_stop_pct': 0.03, 'atr_stop_mult': 2.0, 'min_hold_bars': 1},
        'elder_filter': {'require_volume_confirm': False, 'allow_shorts': True},
    }},
    'features': {'max_window_bars': 500, 'min_bars_required': 50},
}

for n_bars in [2000, 5000, 10000, 20000, 50000]:
    rng = np.random.default_rng(42)
    ts = [1_706_400_000_000 + i * 3600000 for i in range(n_bars)]
    rets = rng.normal(0.0002, 0.01, n_bars)
    close = 50000 * np.cumprod(1 + rets)
    o = np.roll(close, 1); o[0] = close[0] * 0.999
    h = np.maximum(o, close) * (1 + rng.uniform(0.001, 0.01, n_bars))
    l = np.minimum(o, close) * (1 - rng.uniform(0.001, 0.01, n_bars))
    v = rng.uniform(50, 200, n_bars)

    df = pd.DataFrame({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': close, 'volume': v})
    df_d = df.iloc[::24].head(n_bars // 24).reset_index(drop=True)

    engine = BacktestEngine(config)

    # Feature compute only
    t0 = time.perf_counter()
    feat_df = engine.feature_engine.bulk_compute(df)
    feat_time = time.perf_counter() - t0

    # Full walk-forward
    t0 = time.perf_counter()
    result = engine.run_walk_forward(df, MTF_MACD_Elder, data_1d=df_d)
    total_time = time.perf_counter() - t0

    sim_time = total_time - feat_time

    n_trades = len(result.trades)
    sharpe = result.metrics.get('sharpe_ratio', 0)

    print(f"{n_bars:>6,} bars | feat={feat_time:.2f}s | sim={sim_time:.2f}s | "
          f"total={total_time:.2f}s | {n_trades:>4} trades | Sharpe={sharpe:.2f}")
