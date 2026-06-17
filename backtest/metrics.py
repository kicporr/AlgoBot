"""Backtest performance metrics.

Computes: total return, annualized return, Sharpe ratio, Sortino ratio,
Calmar ratio, max drawdown, max drawdown duration, win rate, profit factor,
expectancy, average trade duration, consecutive losses, and fold-level stats.

All metrics assume 1H candles (8,760 bars/year for crypto).
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from loguru import logger


# Crypto = 24/7/365: 365.25 * 24 = 8,766 hours/year
HOURS_PER_YEAR = 365.25 * 24


def calculate_metrics(
    trades: List[dict],
    initial_capital: float = 10000,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> dict:
    """Calculate comprehensive performance metrics from a trade list.

    Args:
        trades: List of trade dicts with keys: pnl, entry_price, exit_price,
                side, exit_reason, bars_held, entry_time, exit_time
        initial_capital: Starting capital used for return calculations
        start_time: Optional start timestamp of the period in ms
        end_time: Optional end timestamp of the period in ms

    Returns:
        Dict of metric name → value
    """
    if not trades:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_trade_pnl": 0.0,
        }

    # Extract PnLs
    pnls = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total_pnl = pnls.sum()

    # Win/Loss metrics
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Equity curve and drawdown
    equity = np.full(len(pnls) + 1, initial_capital, dtype=float)
    equity[1:] = initial_capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = drawdowns.max()
    max_dd_duration = _max_drawdown_duration(equity)

    total_return_pct = (total_pnl / initial_capital) * 100

    # Determine time span in years
    entry_times = [t.get("entry_time") for t in trades if t.get("entry_time") is not None]
    exit_times = [t.get("exit_time") for t in trades if t.get("exit_time") is not None]
    
    t_start = start_time if start_time is not None else (min(entry_times) if entry_times else None)
    t_end = end_time if end_time is not None else (max(exit_times) if exit_times else None)

    if t_start is not None and t_end is not None:
        duration_ms = t_end - t_start
        years = max(duration_ms / (365.25 * 24 * 3600 * 1000), 0.001)
    else:
        years = 1.0

    # Annualized return
    annualized_return = ((1 + total_pnl / initial_capital) ** (1 / years)) - 1 if (years > 0 and (1 + total_pnl / initial_capital) > 0) else 0.0

    # Sharpe ratio (Traditional Daily Sharpe using daily reconstructed equity)
    sharpe = 0.0
    sortino = 0.0
    
    if t_start is not None and t_end is not None:
        start_date = pd.to_datetime(t_start, unit="ms").date()
        end_date = pd.to_datetime(t_end, unit="ms").date()
        if start_date == end_date:
            end_date = start_date + pd.Timedelta(days=1)
            
        daily_dates = pd.date_range(start_date, end_date, freq="1D")
        daily_equity = pd.Series(float(initial_capital), index=daily_dates, dtype=float)
        
        trades_sorted = sorted(trades, key=lambda x: x.get("exit_time", 0))
        current_eq = initial_capital
        for trade in trades_sorted:
            exit_ts = trade.get("exit_time")
            if exit_ts is not None:
                exit_date = pd.to_datetime(exit_ts, unit="ms").date()
                current_eq += trade.get("pnl", 0.0)
                exit_timestamp = pd.Timestamp(exit_date)
                daily_equity.loc[exit_timestamp:] = current_eq
                
        daily_returns = daily_equity.pct_change().fillna(0.0)
        mean_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        
        if std_daily_ret > 0:
            sharpe = (mean_daily_ret / std_daily_ret) * np.sqrt(365.25)
            
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino = (mean_daily_ret / downside_std) * np.sqrt(365.25)
                
    # Fallback to trade-based calculations if dates are not available
    if sharpe == 0.0 and len(pnls) > 1:
        trade_returns = pnls / initial_capital
        if trade_returns.std() > 0:
            trades_per_year = len(pnls) / years if years > 0 else 80.0
            sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year)

    if sortino == 0.0 and len(losses) > 1:
        trade_returns = pnls / initial_capital
        downside_std = np.std([r for r in trade_returns if r < 0])
        if downside_std > 0:
            trades_per_year = len(pnls) / years if years > 0 else 80.0
            sortino = (trade_returns.mean() / downside_std) * np.sqrt(trades_per_year)

    # Calmar ratio (annualized return / max drawdown)
    calmar = annualized_return / max_dd if max_dd > 0 else float("inf")

    # Trade statistics
    exit_reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Consecutive losses
    max_consecutive_losses = 0
    current_streak = 0
    for pnl in pnls:
        if pnl < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # Bars held stats
    bars_held = [t.get("bars_held", 0) for t in trades if t.get("bars_held")]
    avg_bars = np.mean(bars_held) if bars_held else 0

    return {
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4) if not np.isinf(sortino) else "inf",
        "calmar_ratio": round(calmar, 4) if not np.isinf(calmar) else "inf",
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_duration_bars": max_dd_duration,
        "win_rate": round(win_rate * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(avg_win / avg_loss, 2) if avg_loss > 0 else float("inf"),
        "profit_factor": round(profit_factor, 2) if not np.isinf(profit_factor) else "inf",
        "expectancy": round(expectancy, 2),
        "avg_trade_pnl": round(pnls.mean(), 2),
        "max_consecutive_losses": max_consecutive_losses,
        "avg_bars_held": round(avg_bars, 1),
        "exit_reasons": exit_reasons,
    }


def calculate_fold_metrics(trades_per_fold: List[List[dict]], initial_capital: float = 10000) -> List[dict]:
    """Calculate metrics for each fold in walk-forward CV."""
    fold_results = []
    for fold_idx, trades in enumerate(trades_per_fold):
        metrics = calculate_metrics(trades, initial_capital)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)
    return fold_results


def fold_stability_report(fold_metrics: List[dict]) -> dict:
    """Compute mean ± std of key metrics across folds.

    High std → strategy is unstable across market regimes.
    """
    if not fold_metrics:
        return {}

    metrics_of_interest = [
        "sharpe_ratio", "total_return_pct", "win_rate",
        "profit_factor", "max_drawdown_pct",
    ]

    report = {}
    for key in metrics_of_interest:
        values = [fm.get(key, 0) for fm in fold_metrics]
        # Filter out "inf" strings
        numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isinf(v)]
        if numeric_values:
            report[key] = {
                "mean": round(np.mean(numeric_values), 4),
                "std": round(np.std(numeric_values), 4),
                "min": round(min(numeric_values), 4),
                "max": round(max(numeric_values), 4),
            }

    return report


def _max_drawdown_duration(equity: np.ndarray) -> int:
    """Compute longest drawdown duration (in bars).

    Duration is the number of consecutive bars the equity stays below
    its previous peak.
    """
    if len(equity) < 2:
        return 0

    peak = np.maximum.accumulate(equity)
    in_drawdown = equity < peak

    max_duration = 0
    current = 0
    for below in in_drawdown:
        if below:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0

    return max_duration
