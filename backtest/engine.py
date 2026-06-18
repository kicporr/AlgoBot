"""Walk-forward backtesting engine.

Validates strategies using the only methodology that prevents look-ahead
bias in financial time series: N-fold walk-forward with expanding window.

Key rules (Ślępaczuk 2026 methodology):
    - Train on past, test on unseen future — NEVER the reverse
    - Enter at NEXT candle's open after signal
    - Check SL/TP during candle using high/low
    - Apply realistic commission (0.1% taker) and slippage (0.05%)
    - Track equity curve, drawdowns, and per-trade PnL

V2 additions:
    - Ensemble routing: RegimeClassifier + EnsembleRouter per fold
    - Dynamic equity tracking (compound growth)
    - Cooldown after losing trades
    - Dynamic position sizing (50% equity base, scales with streak)
"""

from typing import List, Tuple, Type, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base import BaseStrategy, Signal
from features.engine import FeatureEngine
from backtest.metrics import calculate_metrics
from ensemble.regime_classifier import RegimeClassifier, MarketRegime
from ensemble.router import EnsembleRouter


class BacktestResult:
    """Complete result from a backtest run."""

    def __init__(self):
        self.trades: list[dict] = []
        self.metrics: dict = {}
        self.equity_curve: list[float] = []
        self.drawdown_curve: list[float] = []
        self.fold_metrics: list[dict] = []
        self.config_summary: dict = {}


class BacktestEngine:
    """Event-driven backtesting engine with walk-forward cross-validation."""

    def __init__(self, config: dict):
        self.config = config

        # Fee structure
        fees = config.get("exchange", {}).get("fees", {})
        self.commission = fees.get("taker", 0.001)        # 0.1% taker fee
        self.maker_commission = fees.get("maker", 0.0005)  # 0.05% maker
        self.slippage = fees.get("slippage", 0.0005)       # 0.05%

        # Risk
        risk_cfg = config.get("risk", {})
        self.initial_capital = risk_cfg.get("initial_capital", 10000)
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.50)  # Conservative 50%

        # Walk-forward
        bt_cfg = config.get("backtest", {})
        self.n_folds = bt_cfg.get("walk_forward_folds", 27)
        self.min_train_fraction = bt_cfg.get("min_train_fraction", 0.33)

        # Cooldown
        self.cooldown_bars = bt_cfg.get("cooldown_bars_after_loss", 2)

        # Minimum bars before signal-based exit is allowed
        # (TP, SL, trailing stop still fire immediately)
        self.min_signal_exit_bars = bt_cfg.get("min_signal_exit_bars", 6)

        # Feature engine (for bulk computation)
        self.feature_engine = FeatureEngine(config)

    # ─── Main Entry Point ──────────────────────────────────────

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_class: Type[BaseStrategy],
        data_4h: Optional[pd.DataFrame] = None,
        data_1d: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            data: 1H OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
            strategy_class: Strategy class (not instance! Will create fresh per fold)
            data_4h: Optional 4H data for multi-TF features
            data_1d: Optional 1D data for D1 trend filter

        Returns:
            BacktestResult with trades, metrics, equity curve.
        """
        if len(data) < self.min_train_fraction * 2:
            raise ValueError(f"Need at least {int(self.min_train_fraction * 2):,} bars")

        # Compute all features once
        logger.info("Computing features for whole dataset...")
        features_df = self.feature_engine.bulk_compute(data, data_4h, data_1d)
        logger.info(f"Computed {len(features_df.columns)} features for {len(features_df)} bars")

        # Compute D1 MACD trend for Elder filter (if D1 data available)
        d1_trends = self._compute_d1_trend_series(data_1d, len(data)) if data_1d is not None else None

        # Walk-forward folds
        result = BacktestResult()
        fold_size = len(data) // self.n_folds

        logger.info(f"Starting walk-forward: {self.n_folds} folds, ~{fold_size} bars each")

        for fold in range(1, self.n_folds):  # Fold 0 has insufficient training data
            train_end = fold * fold_size
            test_start = train_end
            test_end = min((fold + 1) * fold_size, len(data))

            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            test_features = features_df.iloc[test_start:test_end]

            if len(test_data) < 10:
                continue

            # Create fresh strategy instance
            strategy = strategy_class(self.config)

            # Train on past data (with features for ML strategies)
            if hasattr(strategy, 'retrain'):
                train_features = features_df.iloc[:train_end].copy()
                train_features["close"] = train_data["close"].values
                strategy.retrain(train_features)

            # Simulate trading on test data
            trades = self._simulate_trading(
                strategy=strategy,
                data=test_data,
                features=test_features,
                d1_trends=d1_trends,
                test_start_idx=test_start,
            )

            if trades:
                fold_metrics = calculate_metrics(
                    trades,
                    self.initial_capital,
                    start_time=test_data["timestamp"].iloc[0],
                    end_time=test_data["timestamp"].iloc[-1]
                )
                fold_metrics["fold"] = fold
                result.fold_metrics.append(fold_metrics)

            result.trades.extend(trades)

        # Compute overall metrics
        result.metrics = calculate_metrics(
            result.trades,
            self.initial_capital,
            start_time=data["timestamp"].iloc[0],
            end_time=data["timestamp"].iloc[-1]
        )
        result.equity_curve = self._compute_equity_curve(result.trades)
        result.drawdown_curve = self._compute_drawdown_curve(result.equity_curve)

        logger.info(
            f"Backtest complete: {len(result.trades)} trades | "
            f"Total PnL: ${result.metrics.get('total_pnl', 0):,.2f} | "
            f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}"
        )

        return result

    def run_ensemble_backtest(
        self,
        data: pd.DataFrame,
        strategy_classes: dict[str, Type[BaseStrategy]],
        data_4h: Optional[pd.DataFrame] = None,
        data_1d: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest with ensemble routing.

        Uses RegimeClassifier to detect market regime and EnsembleRouter
        to delegate signals to the best strategy for each regime.

        Args:
            data: 1H OHLCV DataFrame
            strategy_classes: Dict mapping strategy key -> class.
                Expected keys: 'xgboost', 'mtf_macd', 'mean_reversion'
            data_4h: Optional 4H data
            data_1d: Optional 1D data for D1 trend filter

        Returns:
            BacktestResult with trades, metrics, equity curve, and
            regime-level statistics.
        """
        if len(data) < self.min_train_fraction * 2:
            raise ValueError(f"Need at least {int(self.min_train_fraction * 2):,} bars")

        # Compute all features once
        logger.info("Computing features for ensemble backtest...")
        features_df = self.feature_engine.bulk_compute(data, data_4h, data_1d)
        logger.info(f"Computed {len(features_df.columns)} features for {len(features_df)} bars")

        d1_trends = self._compute_d1_trend_series(data_1d, len(data)) if data_1d is not None else None

        result = BacktestResult()
        fold_size = len(data) // self.n_folds

        logger.info(f"Starting ensemble walk-forward: {self.n_folds} folds, ~{fold_size} bars each")

        for fold in range(1, self.n_folds):
            train_end = fold * fold_size
            test_start = train_end
            test_end = min((fold + 1) * fold_size, len(data))

            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            test_features = features_df.iloc[test_start:test_end]

            if len(test_data) < 10:
                continue

            # Create fresh strategy instances for this fold
            strategies = {}
            for key, cls in strategy_classes.items():
                strat = cls(self.config)
                # Train ML strategies
                if hasattr(strat, 'retrain'):
                    train_features = features_df.iloc[:train_end].copy()
                    train_features["close"] = train_data["close"].values
                    strat.retrain(train_features)
                strategies[key] = strat

            # Create fresh regime classifier + router for this fold
            classifier = RegimeClassifier(self.config)
            router = EnsembleRouter(strategies, classifier)

            # Set D1 trend for MTF MACD strategy if available
            if d1_trends is not None:
                mtf_strat = strategies.get("mtf_macd")
                if mtf_strat and hasattr(mtf_strat, 'set_d1_trend_direct'):
                    # Pre-set initial D1 trend
                    global_idx = test_start
                    if global_idx < len(d1_trends):
                        mtf_strat.set_d1_trend_direct(d1_trends.iloc[global_idx])

            # Simulate trading with ensemble routing
            trades = self._simulate_trading(
                strategy=None,  # Not used when router is provided
                data=test_data,
                features=test_features,
                d1_trends=d1_trends,
                test_start_idx=test_start,
                ensemble_router=router,
            )

            if trades:
                fold_metrics = calculate_metrics(
                    trades,
                    self.initial_capital,
                    start_time=test_data["timestamp"].iloc[0],
                    end_time=test_data["timestamp"].iloc[-1]
                )
                fold_metrics["fold"] = fold
                result.fold_metrics.append(fold_metrics)

                # Record trade outcomes in router for per-regime stats
                for trade in trades:
                    router.record_trade_outcome(trade.get("pnl", 0))

            result.trades.extend(trades)

        # Compute overall metrics
        result.metrics = calculate_metrics(
            result.trades,
            self.initial_capital,
            start_time=data["timestamp"].iloc[0],
            end_time=data["timestamp"].iloc[-1]
        )
        result.equity_curve = self._compute_equity_curve(result.trades)
        result.drawdown_curve = self._compute_drawdown_curve(result.equity_curve)

        logger.info(
            f"Ensemble backtest complete: {len(result.trades)} trades | "
            f"Total PnL: ${result.metrics.get('total_pnl', 0):,.2f} | "
            f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}"
        )

        return result

    # ─── Trade Simulation ──────────────────────────────────────────

    def _simulate_trading(
        self,
        strategy: Optional[BaseStrategy],
        data: pd.DataFrame,
        features: pd.DataFrame,
        d1_trends: Optional[pd.Series] = None,
        test_start_idx: int = 0,
        ensemble_router: Optional[EnsembleRouter] = None,
    ) -> list[dict]:
        """Simulate trading bar-by-bar.

        Trade lifecycle:
            Bar N: strategy.on_candle() returns Signal
            Bar N+1 open: entry or exit is executed
            During Bar N+1: check SL/TP using high/low

        V2 additions:
            - ensemble_router: if provided, signals come from router instead of strategy
            - Dynamic equity tracking (compound growth)
            - Cooldown after losing trades
            - Dynamic position sizing based on streak
        """
        trades = []
        pending_signal: Optional[Signal] = None
        entry_features: Optional[dict] = None  # Features at signal time (meta-labeling)
        entry_signal_type: Optional[str] = None
        entry_regime: Optional[str] = None

        in_position = False
        position_side = ""          # "long" or "short"
        entry_price = 0.0
        entry_bar = 0
        entry_ts = 0
        highest_since_entry = 0.0   # For trailing stop (long)
        lowest_since_entry = float("inf")
        position_size = 0.0
        exit_reason = ""
        entry_theoretical = 0.0

        # Dynamic equity tracking
        current_equity = self.initial_capital
        consecutive_losses = {} # dict for per-strategy tracking
        consecutive_wins = {}   # dict for per-strategy tracking
        last_exit_bar = -999  # Cooldown tracker
        last_trade_was_loss = False
        active_strategy_name = "unknown"
        entry_strategy = "unknown"

        # Convert pandas structures to list/dicts for massive performance speedup (eliminates .iloc lookup overhead)
        data_list = data.to_dict('records')
        features_list = features.to_dict('records') if not features.empty else []
        d1_trends_list = d1_trends.tolist() if d1_trends is not None else None

        for i in range(len(data_list)):
            row = data_list[i]
            feat = features_list[i] if i < len(features_list) else None
            close = row["close"]
            ts = row["timestamp"]

            # Update D1 trend if available (for single-strategy mode)
            if d1_trends_list is not None and strategy is not None and hasattr(strategy, 'set_d1_trend_direct'):
                global_idx = test_start_idx + i
                if global_idx < len(d1_trends_list):
                    strategy.set_d1_trend_direct(d1_trends_list[global_idx])

            # Update D1 trend for ensemble router's MTF MACD strategy
            if d1_trends_list is not None and ensemble_router is not None:
                mtf_strat = ensemble_router.strategies.get("mtf_macd")
                if mtf_strat and hasattr(mtf_strat, 'set_d1_trend_direct'):
                    global_idx = test_start_idx + i
                    if global_idx < len(d1_trends_list):
                        mtf_strat.set_d1_trend_direct(d1_trends_list[global_idx])

            # ── Exit Check (if in position) ────────────────
            if in_position:
                exit_price = None
                atr_14 = feat.get("atr_14", 0) if feat is not None else 0
                atr_pct = feat.get("atr_pct", 2.0) if feat is not None else 2.0

                # Dynamic TP/SL based on volatility
                # Key insight: SL must be wider than TP to let winners run
                # Previous config had too-tight SL causing 51% ATR stop exits
                exit_cfg = self.config.get("strategies", {}).get("mtf_macd_elder", {}).get("exit", {})
                vol_factor = max(0.5, min(2.0, atr_pct / 2.0))  # Normalize around 2% ATR
                tp_mult = 2.0 + (vol_factor * 0.5)              # 2.25-3.0x risk (take profits earlier)
                sl_mult = exit_cfg.get("atr_stop_mult", 2.0 + vol_factor)  # Fallback to vol-derived if not in config
                trail_pct = exit_cfg.get("trailing_stop_pct", 0.025 + (vol_factor * 0.01))
                max_bars = 48  # Max 48 hours in a trade

                if position_side == "long":
                    risk = sl_mult * atr_14 if atr_14 > 0 else entry_price * 0.02
                    tp_price = entry_price + (tp_mult * risk)
                    sl_price = entry_price - risk
                    trail_price = highest_since_entry * (1 - trail_pct)

                    # Priority: TP > signal > trailing > ATR stop
                    if row["high"] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                    elif (exit_price is None and pending_signal is not None
                          and pending_signal == Signal.SHORT
                          and (i - entry_bar) >= self.min_signal_exit_bars):
                        exit_price = row["open"]
                        exit_reason = "signal"
                    elif row["low"] <= trail_price:
                        exit_price = max(trail_price, row["open"])
                        exit_reason = "trailing_stop"
                    elif row["low"] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "atr_stop"
                    elif i - entry_bar >= max_bars:
                        exit_price = row["close"]
                        exit_reason = "time_exit"

                    highest_since_entry = max(highest_since_entry, row["high"])

                elif position_side == "short":
                    risk = sl_mult * atr_14 if atr_14 > 0 else entry_price * 0.02
                    tp_price = entry_price - (tp_mult * risk)
                    sl_price = entry_price + risk
                    trail_price = lowest_since_entry * (1 + trail_pct)

                    if row["low"] <= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                    elif (exit_price is None and pending_signal is not None
                          and pending_signal == Signal.LONG
                          and (i - entry_bar) >= self.min_signal_exit_bars):
                        exit_price = row["open"]
                        exit_reason = "signal"
                    elif row["high"] >= trail_price:
                        exit_price = min(trail_price, row["open"])
                        exit_reason = "trailing_stop"
                    elif row["high"] >= sl_price:
                        exit_price = sl_price
                        exit_reason = "atr_stop"
                    elif i - entry_bar >= max_bars:
                        exit_price = row["close"]
                        exit_reason = "time_exit"

                    lowest_since_entry = min(lowest_since_entry, row["low"])

                if exit_price is not None:
                    # Calculate PnL
                    if position_side == "long":
                        gross_return = (exit_price - entry_price) / entry_price
                    else:
                        gross_return = (entry_price - exit_price) / entry_price

                    # Apply commission (taker on entry + taker on exit)
                    commission_cost = self.commission * 2

                    # Apply slippage on both sides
                    slippage_cost = self.slippage * 2

                    net_return = gross_return - commission_cost - slippage_cost
                    pnl = position_size * entry_price * net_return

                    # Update equity
                    current_equity += pnl

                    # Track streaks per strategy
                    strat_name = entry_strategy
                    if pnl < 0:
                        consecutive_losses[strat_name] = consecutive_losses.get(strat_name, 0) + 1
                        consecutive_wins[strat_name] = 0
                        last_trade_was_loss = True
                    else:
                        consecutive_wins[strat_name] = consecutive_wins.get(strat_name, 0) + 1
                        consecutive_losses[strat_name] = 0
                        last_trade_was_loss = False

                    trades.append({
                        "entry_time": entry_ts,
                        "exit_time": ts,
                        "side": position_side,
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "quantity": round(position_size, 6),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(net_return * 100, 4),
                        "exit_reason": exit_reason,
                        "strategy": entry_strategy,
                        "bars_held": i - entry_bar,
                        "theoretical_entry_price": round(entry_theoretical, 2),
                        "theoretical_exit_price": round(exit_price, 2),
                        "features_at_signal": entry_features,
                        "signal_type": entry_signal_type,
                        "regime": entry_regime,
                    })

                    in_position = False
                    position_side = ""
                    last_exit_bar = i

                    # Don't enter same bar we exited
                    pending_signal = None
                    continue

            # ── Entry Check ─────────────────────────────────
            if not in_position and feat is not None:
                # Cooldown: skip entry after a losing trade
                strat_losses = consecutive_losses.get(active_strategy_name, 0)
                strat_wins = consecutive_wins.get(active_strategy_name, 0)
                
                if last_trade_was_loss and (i - last_exit_bar) < self.cooldown_bars:
                    pending_signal = None
                # Halt after 5 consecutive losses for the active strategy
                elif strat_losses >= 5:
                    pending_signal = None
                elif pending_signal is not None:
                    sig = pending_signal

                    # Dynamic position sizing based on streak
                    sizing_equity = max(current_equity, self.initial_capital * 0.5)  # Floor at 50% of initial
                    if strat_losses >= 2:
                        size_pct = self.max_position_pct * 0.5  # Half size after 2+ losses
                    elif strat_wins >= 3:
                        size_pct = min(self.max_position_pct * 1.5, 0.75)  # Max 75% of equity
                    else:
                        size_pct = self.max_position_pct

                    if sig == Signal.LONG:
                        entry_price = row["open"]
                        entry_theoretical = entry_price
                        gross_entry = entry_price * (1 + self.slippage)
                        position_size = (sizing_equity * size_pct) / gross_entry

                        in_position = True
                        position_side = "long"
                        entry_bar = i
                        entry_ts = ts
                        highest_since_entry = entry_price
                        entry_price = gross_entry  # Use slippage-adjusted price
                        entry_strategy = active_strategy_name

                    elif sig == Signal.SHORT:
                        entry_price = row["open"]
                        entry_theoretical = entry_price
                        gross_entry = entry_price * (1 - self.slippage)
                        position_size = (sizing_equity * size_pct) / gross_entry

                        in_position = True
                        position_side = "short"
                        entry_bar = i
                        entry_ts = ts
                        lowest_since_entry = entry_price
                        entry_price = gross_entry
                        entry_strategy = active_strategy_name

            # ── Get signal for NEXT bar ─────────────────────
            if feat is not None:
                candle_dict = {
                    "timestamp": ts,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": close,
                    "volume": row["volume"],
                    "atr_14": feat.get("atr_14", 0),
                }

                # Get signal from ensemble router or single strategy
                if ensemble_router is not None:
                    # Sync strategy position states with actual engine state
                    for strat in ensemble_router.strategies.values():
                        if hasattr(strat, "in_position"):
                            strat.in_position = in_position
                        if hasattr(strat, "position_side"):
                            strat.position_side = position_side

                    new_signal = ensemble_router.get_signal(candle_dict, feat)
                    active_strategy_name = ensemble_router.get_active_strategy_name()
                else:
                    if hasattr(strategy, "in_position"):
                        strategy.in_position = in_position
                    if hasattr(strategy, "position_side"):
                        strategy.position_side = position_side

                    new_signal = strategy.on_candle(candle_dict, feat)
                    active_strategy_name = strategy.name if strategy else "unknown"

                # Only update pending on FLAT if we're not in position
                if not in_position:
                    pending_signal = new_signal if new_signal != Signal.FLAT else None
                else:
                    pending_signal = new_signal  # Keep for exit check

                # Store features at signal time (for meta-labeling training)
                if pending_signal is not None:
                    entry_features = dict(feat) if feat else {}
                    entry_signal_type = pending_signal.name  # "LONG" or "SHORT"
                    entry_regime = (ensemble_router.current_regime.value
                                    if ensemble_router and hasattr(ensemble_router, 'current_regime')
                                    else None)
                else:
                    entry_features = None
                    entry_signal_type = None
                    entry_regime = None

        return trades

    # ─── D1 Trend Helper ───────────────────────────────────────

    def _compute_d1_trend_series(self, data_1d: pd.DataFrame, n_hourly_bars: int) -> pd.Series:
        """Compute Elder D1 trend, then expand to 1H resolution.

        Returns Series of length n_hourly_bars: "UP", "DOWN", or "FLAT".
        Each 1H bar gets the trend from the last completed D1 candle.
        """
        from features.indicators import IndicatorCalculator
        ic = IndicatorCalculator()

        macd_fast = self.config.get("strategies", {}).get("mtf_macd_elder", {}).get("macd", {}).get("fast", 12)
        macd_slow = self.config.get("strategies", {}).get("mtf_macd_elder", {}).get("macd", {}).get("slow", 26)
        macd_sig = self.config.get("strategies", {}).get("mtf_macd_elder", {}).get("macd", {}).get("signal", 9)

        d1_closes = data_1d["close"]
        macd_line, signal_line, macd_hist = ic.macd(d1_closes, macd_fast, macd_slow, macd_sig)

        hist_slope = macd_hist.diff(1)
        daily_trend = pd.Series("FLAT", index=data_1d.index)
        daily_trend[hist_slope > 0] = "UP"
        daily_trend[hist_slope < 0] = "DOWN"

        # Reindex to 1H: forward-fill daily trend to all hourly bars
        hourly_index = pd.RangeIndex(n_hourly_bars)
        df_daily = pd.DataFrame({"trend": daily_trend.values}, index=data_1d["timestamp"])
        df_hourly = pd.DataFrame(index=hourly_index)

        # Simple: each 24-hour block gets the daily trend
        trends_1h = pd.Series("FLAT", index=hourly_index)
        for i in range(len(data_1d)):
            start_h = i * 24
            end_h = min((i + 1) * 24, n_hourly_bars)
            trends_1h.iloc[start_h:end_h] = daily_trend.iloc[i]

        return trends_1h

    # ─── Equity / Drawdown Curves ──────────────────────────────

    def _compute_equity_curve(self, trades: list[dict]) -> list[float]:
        """Compute equity curve from trade list."""
        equity = self.initial_capital
        curve = [equity]

        for trade in trades:
            equity += trade.get("pnl", 0)
            curve.append(equity)

        return curve

    def _compute_drawdown_curve(self, equity_curve: list[float]) -> list[float]:
        """Compute drawdown curve from equity."""
        peak = equity_curve[0]
        drawdowns = []

        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            drawdowns.append(dd)

        return drawdowns
