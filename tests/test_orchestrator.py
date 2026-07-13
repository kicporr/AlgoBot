"""Integration tests for the main trading orchestrator.

Tests the full signal pipeline: FeatureEngine → Strategy → Ensemble →
CircuitBreaker → PositionSizer → Execution (paper mode).

All external dependencies (exchange, WebSocket, Telegram) are mocked.
Uses a minimal config to keep tests fast and deterministic.
"""

import sys, os, time, threading, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, PropertyMock, patch

from orchestrator import TradingBot, TradingPipeline
from strategies.base import Signal
from risk.circuit_breaker import BreakerState


# ─── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Minimal config for testing — no network, no DB, single symbol."""
    return {
        "bot": {"name": "test", "version": "0.1.0", "mode": "paper", "log_level": "WARNING"},
        "exchange": {
            "name": "bitget", "type": "spot", "ws_inst_type": "SPOT",
            "symbols": ["BTC/USDT:USDT"],
            "fees": {"maker": 0.0002, "taker": 0.0006, "slippage": 0.0002},
            "rate_limit": {"max_requests_per_second": 10},
        },
        "risk": {
            "initial_capital": 10000.0,
            "max_position_pct": 0.50,
            "max_concurrent_positions": 3,
            "position_sizing": {
                "method": "fixed_fraction",
                "max_risk_per_trade_pct": 2.0,
                "max_position_size_btc": 1.0,
                "max_total_exposure_pct": 80,
            },
            "circuit_breaker": {
                "max_drawdown_pct": 20,
                "daily_loss_limit_pct": 5,
                "weekly_loss_limit_pct": 10,
                "consecutive_loss_halt": 5,
                "consecutive_loss_warn": 3,
                "volatility_circuit_mult": 5.0,
                "loss_reference": "peak",
            },
            "per_trade": {"max_duration_hours": 48},
        },
        "data": {
            "database": {"type": "sqlite", "path": ":memory:"},
            "validation": {"max_price_jump_pct": 30},
            "resampler": {"min_bars_1h": 30, "min_bars_4h": 120, "min_bars_1d": 720},
        },
        "features": {"max_window_bars": 500, "min_bars_required": 50},
        "strategies": {
            "mtf_macd_elder": {
                "enabled": True,
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "exit": {"trailing_stop_pct": 0.03, "atr_stop_mult": 2.0, "min_hold_bars": 1},
                "elder_filter": {"require_volume_confirm": False, "allow_shorts": True, "volume_mult": 1.2},
            },
            "mean_reversion": {"enabled": True, "rsi": {"period": 14, "oversold": 30, "overbought": 70},
                               "bollinger": {"period": 20, "std_dev": 2}, "require_both_signals": True, "allow_shorts": False},
        },
        "regime": {
            "trending": {"adx_min": 25, "di_ratio_strong": 1.3, "di_ratio_reverse": 0.77},
            "ranging": {"adx_max": 20, "bb_width_max": 0.04, "vol_max": 0.50},
            "volatile": {"atr_mult": 2.0, "vol_absolute": 1.0, "bb_width_min": 0.08},
            "hysteresis_bars": 2, "lookback_bars": 100,
        },
        "meta_labeling": {"enabled": True, "min_confidence": 0.55, "training_samples": 1000},
        "execution": {"order_type": "limit", "order_timeout_seconds": 30, "max_retries": 3},
        "monitoring": {
            "telegram": {"enabled": False, "reports": {"enabled": False}},
            "snapshot_interval_minutes": 60,
        },
        "paths": {"data_dir": "./data", "models_dir": "./models", "logs_dir": "./logs"},
        "dashboard": {"enabled": False},
        "backtest": {"min_signal_exit_bars": 6},
        "symbols": {},
    }


@pytest.fixture
def make_candle():
    """Factory for creating test candle dicts."""
    base = 50000.0
    def _make(**overrides):
        return {
            "timestamp": int(time.time() * 1000),
            "open": base,
            "high": base + 100,
            "low": base - 100,
            "close": base,
            "volume": 100.0,
            **overrides,
        }
    return _make


@pytest.fixture
def make_features():
    """Factory for creating feature dicts with realistic values for TRENDING regime."""
    def _make(**overrides):
        return {
            "close": 50000.0, "price": 50000.0,
            "adx_14": 30.0, "di_plus": 30.0, "di_minus": 10.0, "di_ratio": 3.0,
            "atr_14": 500.0, "atr_pct": 1.0,
            "macd": 100.0, "macd_signal": 90.0, "macd_hist": 10.0, "macd_cross": 1,
            "bb_width": 0.03, "volatility_20": 0.30, "garman_klass": 0.02,
            "rsi_14": 55.0, "volume_sma_ratio": 1.5,
            "dist_sma_50": 0.02, "trend_strength": 0.6,
            **overrides,
        }
    return _make


def make_bot(config, with_mocks=True):
    """Create a TradingBot with all external dependencies mocked."""
    with patch("orchestrator.load_dotenv"), \
         patch("orchestrator.setup_logger", return_value=MagicMock()), \
         patch("orchestrator.TelegramAlerter"), \
         patch("orchestrator.has_websocket", return_value=False), \
         patch("orchestrator.DatabaseManager"), \
         patch("orchestrator.CandleRepository"), \
         patch("orchestrator.TradeRepository"), \
         patch("orchestrator.SignalRepository"), \
         patch("orchestrator.ExchangeAdapter"), \
         patch("orchestrator.BitgetRESTClient"), \
         patch("ccxt.bitget", return_value=MagicMock(load_markets=MagicMock())):
        # Override config path
        import yaml, tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        bot = TradingBot(config_path)
        os.unlink(config_path)

        # Prime regime classifier past hysteresis for all symbols (2 bars of TRENDING)
        for sym in config["exchange"]["symbols"]:
            for p in bot.pipelines.values():
                rc = p.symbol_states[sym]["regime_classifier"]
                feats = {"adx_14": 30, "di_plus": 30, "di_minus": 10, "atr_14": 500,
                         "bb_width": 0.03, "volatility_20": 0.30, "garman_klass": 0.02,
                         "price": 50000, "dist_sma_50": 0.02}
                rc.classify(feats)  # bar 1
                rc.classify(feats)  # bar 2 → switches to TRENDING

        return bot


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end tests for the 1H candle trading pipeline."""

    def test_flat_when_no_features(self, mock_config, make_candle):
        """Bot should return early when FeatureEngine has insufficient history."""
        bot = make_bot(mock_config)
        state = bot.symbol_states["BTC/USDT:USDT"]
        # Override process_candle to return empty (insufficient bars)
        state["feature_engine"].process_candle = MagicMock(return_value={})
        candle = make_candle()

        bot._on_1h_candle("BTC/USDT:USDT", candle)
        # Nothing should happen — no positions opened, no signals
        assert len(state["open_positions"]) == 0

    def test_long_signal_executes_paper_trade(self, mock_config, make_candle, make_features):
        """A LONG signal should open a paper position."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]
        features = make_features()

        # Mock strategy to return LONG
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=features)

        candle = make_candle(close=50000.0)
        bot._on_1h_candle(sym, candle)

        # Position should be opened
        assert len(state["open_positions"]) == 1
        pos = state["open_positions"].get("long")
        assert pos is not None
        assert pos["side"] == "long"
        assert pos["entry_price"] > 0
        assert pos["size"] > 0

    def test_short_signal_executes_paper_trade(self, mock_config, make_candle, make_features):
        """A SHORT signal should open a paper short position."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]
        features = make_features(macd_cross=-1)

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.SHORT)
        state["feature_engine"].process_candle = MagicMock(return_value=features)

        candle = make_candle(close=50000.0)
        bot._on_1h_candle(sym, candle)

        assert len(state["open_positions"]) == 1
        pos = state["open_positions"].get("short")
        assert pos is not None
        assert pos["side"] == "short"

    def test_flat_signal_does_nothing(self, mock_config, make_candle, make_features):
        """FLAT signal should not open any position."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.FLAT)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle())
        assert len(state["open_positions"]) == 0

    def test_no_entry_when_in_position(self, mock_config, make_candle, make_features):
        """Bot should not open a second position when already in one."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        # First: open a long
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(sym, make_candle(close=50000))

        # Second: signal LONG again — should stay in position, not add
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        bot._on_1h_candle(sym, make_candle(close=51000))
        assert len(state["open_positions"]) == 1  # still only one


# ═══════════════════════════════════════════════════════════════
# EXIT LOGIC TESTS
# ═══════════════════════════════════════════════════════════════

class TestExitLogic:
    """Tests for position exit (take profit, trailing stop, stop loss)."""

    def test_opposite_signal_exits_position(self, mock_config, make_candle, make_features):
        """When in LONG, a SHORT signal should close the position."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        # Open long at 50000
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(sym, make_candle(close=50000))
        assert len(state["open_positions"]) == 1

        # Mock tracker to say "hold" AND set bars_held >= 1
        state["position_tracker"].update = MagicMock(return_value="hold")
        state["position_tracker"].bars_held = 6  # >= min_hold_bars
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.SHORT)
        bot._on_1h_candle(sym, make_candle(close=51000))

        # Position should be closed
        assert len(state["open_positions"]) == 0
        assert bot.balance > 10000.0

    def test_trailing_stop_exits_long(self, mock_config, make_candle, make_features):
        """PositionTracker should trigger exit when trailing stop is hit."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        # Open long at 50000
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(sym, make_candle(close=50000))

        # PositionTracker returns "trailing_stop"
        state["position_tracker"].update = MagicMock(return_value="trailing_stop")
        bot._on_1h_candle(sym, make_candle(close=49500))

        assert len(state["open_positions"]) == 0

    def test_take_profit_exits_long(self, mock_config, make_candle, make_features):
        """PositionTracker should exit on take profit."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(sym, make_candle(close=50000))

        state["position_tracker"].update = MagicMock(return_value="take_profit")
        bot._on_1h_candle(sym, make_candle(close=53000))

        assert len(state["open_positions"]) == 0


# ═══════════════════════════════════════════════════════════════
# CIRCUIT BREAKER TESTS
# ═══════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Tests for circuit breaker integration in the trading loop."""

    def test_halted_breaker_blocks_entry(self, mock_config, make_candle, make_features):
        """When circuit breaker is HALTED, no new positions are opened."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        # Force halt via emergency_stop (proper way)
        bot.circuit_breaker.emergency_stop("test halt")
        assert bot.circuit_breaker.state == BreakerState.HALTED

        bot.symbol_states[sym]["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        bot.symbol_states[sym]["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle())
        assert len(bot.symbol_states[sym]["open_positions"]) == 0

    def test_warning_skips_signal(self, mock_config, make_candle, make_features):
        """WARNING state should skip one bar's signals."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        # Trigger 3 consecutive losses to enter WARNING state
        bot.recent_trades_pnl = [-100, -50, -75]
        bot.circuit_breaker.check(
            equity=10000, balance=10000,
            recent_trades_pnl=[-100, -50, -75],
            current_atr=500, avg_atr=500,
        )
        assert bot.circuit_breaker.state == BreakerState.WARNING

        bot.symbol_states[sym]["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        bot.symbol_states[sym]["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle())
        # Signal should be blocked by WARNING state
        assert len(bot.symbol_states[sym]["open_positions"]) == 0

    def test_consecutive_losses_trigger_warning(self, mock_config, make_candle, make_features):
        """3 consecutive losses should trigger WARNING on next check, blocking next entry."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        # Pre-populate 3 consecutive losses
        bot.recent_trades_pnl = [-100, -50, -75]
        bot.circuit_breaker.peak_equity = 10000

        # The first call to _on_1h_candle will check CB, detect 3 losses, and block entry
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle(close=50000))

        # Position should NOT be opened (blocked by WARNING)
        assert len(state["open_positions"]) == 0
        assert bot.circuit_breaker.state == BreakerState.WARNING


# ═══════════════════════════════════════════════════════════════
# RISK LIMITS TESTS
# ═══════════════════════════════════════════════════════════════

class TestRiskLimits:
    """Tests for position sizing and exposure limits."""

    def test_position_sizer_zero_blocks_trade(self, mock_config, make_candle, make_features):
        """When position sizer returns 0, no trade should execute."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        # Force sizer to return 0
        state["position_sizer"].calculate = MagicMock(return_value=0.0)
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle())
        assert len(state["open_positions"]) == 0

    def test_max_concurrent_positions_blocks(self, mock_config, make_candle, make_features):
        """Should block entry when max concurrent positions limit is reached."""
        # Use config with 2 symbols and max_concurrent=1
        cfg = dict(mock_config)
        cfg["exchange"]["symbols"] = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        cfg["risk"]["max_concurrent_positions"] = 1

        bot = make_bot(cfg)
        btc = "BTC/USDT:USDT"
        eth = "ETH/USDT:USDT"

        # Open position on BTC
        bot.symbol_states[btc]["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        bot.symbol_states[btc]["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(btc, make_candle(close=50000))
        assert len(bot.symbol_states[btc]["open_positions"]) == 1

        # Try to open on ETH — should be blocked
        bot.symbol_states[eth]["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        bot.symbol_states[eth]["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(eth, make_candle(close=3000))

        assert len(bot.symbol_states[eth]["open_positions"]) == 0

    def test_max_exposure_blocks_entry(self, mock_config, make_candle, make_features):
        """Entry blocked when total exposure would exceed max."""
        cfg = dict(mock_config)
        cfg["exchange"]["symbols"] = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        cfg["risk"]["position_sizing"]["max_total_exposure_pct"] = 1  # 1% limit = $100

        bot = make_bot(cfg)
        btc = "BTC/USDT:USDT"
        eth = "ETH/USDT:USDT"

        # First position: $50000 * 0.5 = $25000 in exposure (with max_position_pct=0.50)
        # But 1% exposure limit = $100. The sizer should respect this.
        # Actually the sizer uses max_exposure_pct: position_value <= capital * 1%
        # With capital $10000, max exposure = $100. Position at 50k = 0.002 BTC max.

        bot.symbol_states[btc]["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        bot.symbol_states[btc]["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(btc, make_candle(close=50000))
        assert len(bot.symbol_states[btc]["open_positions"]) == 1

        # Second position should be blocked (already $100 exposure)
        bot.symbol_states[eth]["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        bot.symbol_states[eth]["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(eth, make_candle(close=3000))
        assert len(bot.symbol_states[eth]["open_positions"]) == 0


# ═══════════════════════════════════════════════════════════════
# DUAL PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestDualPipeline:
    """Tests for Pure vs ML pipeline isolation."""

    def test_pipelines_are_independent(self, mock_config, make_candle, make_features):
        """Pure and ML pipelines should have separate capital and positions."""
        bot = make_bot(mock_config)

        pure = bot.pipelines["pure"]
        ml = bot.pipelines["ml"]

        # Both start with same capital
        assert pure.balance == 10000.0
        assert ml.balance == 10000.0
        assert pure.balance == ml.balance

        # ML pipeline should have MetaLabeler
        assert ml.meta_labeler is not None
        assert pure.meta_labeler is None

    def test_pure_pipeline_skips_metalabeler(self, mock_config, make_candle, make_features):
        """Pure pipeline should execute without MetaLabeler filtering."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        # Set active pipeline to pure
        bot._active_pipeline = bot.pipelines["pure"]
        state = bot.symbol_states[sym]

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle(close=50000))
        assert len(state["open_positions"]) == 1

        # Pure pipeline — MetaLabeler was never called
        assert bot._active().meta_labeler is None

    def test_ml_pipeline_filters_signals(self, mock_config, make_candle, make_features):
        """ML pipeline should filter signals through MetaLabeler."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        # Set active pipeline to ml
        bot._active_pipeline = bot.pipelines["ml"]
        state = bot.symbol_states[sym]

        # Make MetaLabeler reject the signal
        bot.pipelines["ml"].meta_labeler.evaluate = MagicMock(return_value=False)
        bot.pipelines["ml"].meta_labeler.is_ready = MagicMock(return_value=True)

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle(close=50000))
        # Position should NOT be opened (rejected by MetaLabeler)
        assert len(state["open_positions"]) == 0

    def test_ml_pipeline_approves_good_signals(self, mock_config, make_candle, make_features):
        """ML pipeline should execute when MetaLabeler approves."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        bot._active_pipeline = bot.pipelines["ml"]
        state = bot.symbol_states[sym]

        # MetaLabeler approves
        bot.pipelines["ml"].meta_labeler.evaluate = MagicMock(return_value=True)
        bot.pipelines["ml"].meta_labeler.is_ready = MagicMock(return_value=True)

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle(close=50000))
        assert len(state["open_positions"]) == 1

    def test_pipeline_capital_isolation(self, mock_config, make_candle, make_features):
        """Closing a trade in one pipeline should not affect the other's balance."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        pure = bot.pipelines["pure"]
        ml = bot.pipelines["ml"]

        # Open and close in pure pipeline
        bot._active_pipeline = pure
        pstate = pure.symbol_states[sym]
        pstate["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        pstate["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(sym, make_candle(close=50000))
        pnl = bot._close_position(sym, "long", make_candle(close=51000), Signal.SHORT, reason="signal")

        # Pure balance changed, ML unchanged
        assert pure.balance != 10000.0
        assert ml.balance == 10000.0


# ═══════════════════════════════════════════════════════════════
# POSITION TRACKING TESTS
# ═══════════════════════════════════════════════════════════════

class TestPositionTracking:
    """Tests for correct PnL calculation and position bookkeeping."""

    def test_long_position_pnl_calculation(self, mock_config, make_candle, make_features):
        """A long position should calculate profit correctly on close."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        # Enter long at $50,000
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())
        bot._on_1h_candle(sym, make_candle(close=50000))
        entry_balance = bot.balance

        # Exit at $51,000 (+2%)
        state["position_tracker"].update = MagicMock(return_value="signal")
        # Override the close mechanism — call _close_position directly with known prices
        pos_size = state["open_positions"]["long"]["size"]
        bot._close_position(sym, "long", make_candle(close=51000), Signal.SHORT, reason="take_profit")

        expected_pnl = pos_size * 50000 * (0.02 - 0.0012 - 0.001)  # gross - commission - slippage
        assert bot.balance > entry_balance
        assert len(state["open_positions"]) == 0

    def test_short_position_pnl_calculation(self, mock_config, make_candle, make_features):
        """A short position should profit when price drops."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.SHORT)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features(macd_cross=-1))
        bot._on_1h_candle(sym, make_candle(close=50000))
        entry_balance = bot.balance

        # Exit at $49,000 (-2%)
        state["position_tracker"].update = MagicMock(return_value="signal")
        bot._close_position(sym, "short", make_candle(close=49000), Signal.LONG, reason="take_profit")

        assert bot.balance > entry_balance

    def test_position_not_opened_when_size_is_zero(self, mock_config, make_candle, make_features):
        """If position sizer returns 0, no position should be tracked."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        state["position_sizer"].calculate = MagicMock(return_value=0.0)
        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle())
        assert len(state["open_positions"]) == 0


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_handler_handles_exception_gracefully(self, mock_config, make_candle):
        """If FeatureEngine throws, the handler should catch it and not crash."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        bot.symbol_states[sym]["feature_engine"].process_candle = MagicMock(side_effect=RuntimeError("simulated error"))

        # Should not raise
        bot._on_1h_candle(sym, make_candle())
        # Bot should still be alive
        assert bot.running is False  # default

    def test_close_nonexistent_position_noop(self, mock_config, make_candle):
        """Closing a non-existent position should not crash."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        # Should not raise
        bot._close_position(sym, "long", make_candle(), Signal.SHORT, reason="manual")

    def test_empty_balance_prevents_trade(self, mock_config, make_candle, make_features):
        """With 0 balance, no position should open."""
        bot = make_bot(mock_config)
        bot.balance = 0.0
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        bot._on_1h_candle(sym, make_candle(close=50000))
        assert len(state["open_positions"]) == 0

    def test_dual_pipeline_fan_out_calls_both(self, mock_config, make_candle, make_features):
        """_on_1h_candle_dual should call _on_1h_candle for both pipelines."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"

        # Setup both pipelines for LONG
        for p in bot.pipelines.values():
            p.symbol_states[sym]["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
            p.symbol_states[sym]["feature_engine"].process_candle = MagicMock(return_value=make_features())

        # Call the dual fan-out
        bot._active_pipeline = bot.pipelines["pure"]
        bot._on_1h_candle_dual(sym, make_candle(close=50000))

        # Both pipelines should have positions
        assert len(bot.pipelines["pure"].symbol_states[sym]["open_positions"]) == 1
        assert len(bot.pipelines["ml"].symbol_states[sym]["open_positions"]) == 1

    def test_negative_price_does_not_crash(self, mock_config, make_candle, make_features):
        """Negative close price should be handled gracefully."""
        bot = make_bot(mock_config)
        sym = "BTC/USDT:USDT"
        state = bot.symbol_states[sym]

        state["strategies"]["mtf_macd"].on_candle = MagicMock(return_value=Signal.LONG)
        state["feature_engine"].process_candle = MagicMock(return_value=make_features())

        # Negative price — should not crash
        bot._on_1h_candle(sym, make_candle(close=-1))
        # Position sizer should handle this gracefully
