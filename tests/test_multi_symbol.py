"""Tests for Multi-Symbol Trading functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
import yaml

from strategies.base import Signal
from orchestrator import TradingBot


@pytest.fixture
def mock_dependencies():
    """Mock database and external API clients to avoid real connections or side effects."""
    with patch("orchestrator.DatabaseManager"), \
         patch("orchestrator.CandleRepository"), \
         patch("orchestrator.TradeRepository"), \
         patch("orchestrator.SignalRepository"), \
         patch("orchestrator.BitgetRESTClient"), \
         patch("orchestrator.BitgetWSClient"), \
         patch("orchestrator.ExchangeAdapter"), \
         patch("orchestrator.OrderManager"), \
         patch("orchestrator.TelegramAlerter"):
        yield


@pytest.fixture
def base_config_yaml():
    yaml_content = """
bot:
  name: "bocik"
  version: "0.1.0"
  mode: "paper"
  log_level: "INFO"

exchange:
  name: "bitget"
  type: "swap"
  ws_inst_type: "USDT-FUTURES"
  symbols:
    - "BTC/USDT:USDT"
    - "ETH/USDT:USDT"
    - "XRP/USDT:USDT"
    - "SOL/USDT:USDT"
  timeframes:
    primary: "1h"
    secondary: "4h"
    higher_tf: "1d"
  fees:
    maker: 0.0002
    taker: 0.0006
    slippage: 0.0005

data:
  database:
    type: "sqlite"
    path: ":memory:"
  cache:
    max_candles_1m: 100
    max_candles_1h: 100
    max_candles_4h: 100
    max_candles_1d: 100
  validation:
    max_price_jump_pct: 30
    timestamp_tolerance_s: 5

features:
  enabled_groups:
    - "trend"
    - "volatility"

strategies:
  mtf_macd_elder:
    enabled: true
    macd:
      fast: 12
      slow: 26
      signal: 9
    elder_filter:
      enabled: true
      higher_tf: "1d"
    exit:
      min_hold_bars: 6

risk:
  max_position_pct: 1.00
  position_sizing:
    method: "fixed_fraction"
    max_risk_per_trade_pct: 2.0
    max_position_size_btc: 0.5
    max_total_exposure_pct: 150
  circuit_breaker:
    max_drawdown_pct: 20
    daily_loss_limit_pct: 5
    weekly_loss_limit_pct: 10
    consecutive_loss_halt: 5
    consecutive_loss_warn: 3
    volatility_circuit_mult: 5.0
  per_trade:
    stop_loss_pct: 3.0
    take_profit_pct: 8.0
    trailing_stop_activation: 3.0
    trailing_stop_distance: 1.5
    volatility_stop_mult: 2.0
    max_duration_hours: 48
    cooldown_minutes: 60

symbols:
  "BTC/USDT:USDT":
    enabled: true
    risk:
      max_position_pct: 1.00
  "ETH/USDT:USDT":
    enabled: true
    risk:
      max_position_pct: 0.80
    strategies:
      mtf_macd_elder:
        macd:
          fast: 8
          slow: 21
          signal: 9
        exit:
          trailing_stop_pct: 0.04
          atr_stop_mult: 2.5
          min_hold_bars: 6
  "XRP/USDT:USDT":
    enabled: true
    risk:
      max_position_pct: 0.70
    strategies:
      mtf_macd_elder:
        exit:
          trailing_stop_pct: 0.04
          atr_stop_mult: 2.5
          min_hold_bars: 6
  "SOL/USDT:USDT":
    enabled: true
    risk:
      max_position_pct: 0.75
    strategies:
      mtf_macd_elder:
        macd:
          fast: 8
          slow: 21
          signal: 9
        elder_filter:
          allow_shorts: false
        exit:
          trailing_stop_pct: 0.03
          atr_stop_mult: 2.0
          min_hold_bars: 6
"""
    return yaml.safe_load(yaml_content)


def test_config_merging(mock_dependencies, base_config_yaml):
    """Verify that global configuration merges correctly with symbol overrides."""
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=base_config_yaml):
        bot = TradingBot()
        
        # Verify symbol configs are created
        assert "BTC/USDT:USDT" in bot.symbol_states
        assert "ETH/USDT:USDT" in bot.symbol_states
        assert "XRP/USDT:USDT" in bot.symbol_states
        assert "SOL/USDT:USDT" in bot.symbol_states
        
        # Check BTC config (should inherit global MACD)
        btc_cfg = bot.symbol_states["BTC/USDT:USDT"]["config"]
        assert btc_cfg["strategies"]["mtf_macd_elder"]["macd"]["fast"] == 12
        assert btc_cfg["risk"]["max_position_pct"] == 1.00
        
        # Check ETH config (should have overridden MACD and risk values)
        eth_cfg = bot.symbol_states["ETH/USDT:USDT"]["config"]
        assert eth_cfg["strategies"]["mtf_macd_elder"]["macd"]["fast"] == 8
        assert eth_cfg["strategies"]["mtf_macd_elder"]["macd"]["slow"] == 21
        assert eth_cfg["strategies"]["mtf_macd_elder"]["exit"]["trailing_stop_pct"] == 0.04
        assert eth_cfg["strategies"]["mtf_macd_elder"]["exit"]["atr_stop_mult"] == 2.5
        assert eth_cfg["risk"]["max_position_pct"] == 0.80
        
        # Check XRP config (should have global MACD but overridden exit values)
        xrp_cfg = bot.symbol_states["XRP/USDT:USDT"]["config"]
        assert xrp_cfg["strategies"]["mtf_macd_elder"]["macd"]["fast"] == 12
        assert xrp_cfg["strategies"]["mtf_macd_elder"]["exit"]["trailing_stop_pct"] == 0.04
        assert xrp_cfg["risk"]["max_position_pct"] == 0.70
        
        # Check SOL config (should have custom MACD, allow_shorts=False, and custom exits)
        sol_cfg = bot.symbol_states["SOL/USDT:USDT"]["config"]
        assert sol_cfg["strategies"]["mtf_macd_elder"]["macd"]["fast"] == 8
        assert sol_cfg["strategies"]["mtf_macd_elder"]["macd"]["slow"] == 21
        assert sol_cfg["strategies"]["mtf_macd_elder"]["elder_filter"]["allow_shorts"] is False
        assert sol_cfg["strategies"]["mtf_macd_elder"]["exit"]["trailing_stop_pct"] == 0.03
        assert sol_cfg["strategies"]["mtf_macd_elder"]["exit"]["atr_stop_mult"] == 2.0
        assert sol_cfg["risk"]["max_position_pct"] == 0.75


def test_independent_symbol_states(mock_dependencies, base_config_yaml):
    """Verify that strategies and trackers keep independent states per symbol."""
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=base_config_yaml):
        bot = TradingBot()
        
        btc_state = bot.symbol_states["BTC/USDT:USDT"]
        eth_state = bot.symbol_states["ETH/USDT:USDT"]
        
        # Strategies must be separate objects
        assert btc_state["strategies"]["mtf_macd"] is not eth_state["strategies"]["mtf_macd"]
        assert btc_state["position_tracker"] is not eth_state["position_tracker"]
        
        # Simulating active position on BTC but not ETH
        btc_state["open_positions"]["long"] = {"entry_price": 50000.0, "size": 0.1, "ts": 12345}
        assert len(btc_state["open_positions"]) == 1
        assert len(eth_state["open_positions"]) == 0


def test_execute_and_close_position(mock_dependencies, base_config_yaml):
    """Verify that _execute_signal and _close_position handle state & compatibility dicts correctly."""
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=base_config_yaml):
        bot = TradingBot()
        bot.paper_trading = True
        
        symbol = "ETH/USDT:USDT"
        candle = {"timestamp": 1600000000000, "close": 3000.0, "open": 2990.0, "high": 3010.0, "low": 2980.0}
        
        # 1. Execute Long Signal
        bot._execute_signal(symbol, Signal.LONG, candle, size_coin=2.0, atr=50.0, atr_pct=1.5)
        
        state = bot.symbol_states[symbol]
        assert "long" in state["open_positions"]
        assert state["open_positions"]["long"]["size"] == 2.0
        assert state["open_positions"]["long"]["entry_price"] > 3000.0  # including slippage
        
        # Verify compatibility dict syncing (symbol-prefixed key only)
        assert f"{symbol}:long" in bot.open_positions
        
        # Verify tracker enter and SL/TP overrides
        tracker = state["position_tracker"]
        assert tracker.position is not None
        assert tracker.position.symbol == symbol
        assert tracker.position.trailing_distance == 0.04  # Custom override
        assert tracker.position.stop_loss == tracker.position.entry_price - (2.5 * 50.0)  # 2.5x ATR override
        
        # 2. Close Position
        exit_candle = {"timestamp": 1600003600000, "close": 3100.0, "open": 3000.0, "high": 3110.0, "low": 2990.0}
        bot._close_position(symbol, "long", exit_candle, Signal.SHORT, reason="take_profit")
        
        assert "long" not in state["open_positions"]
        assert f"{symbol}:long" not in bot.open_positions
        assert "long" not in bot.open_positions
        assert tracker.position is None
        
        # Verify trade was inserted in DB with dynamic strategy field
        bot.trade_repo.insert.assert_called_once()
        inserted_trade = bot.trade_repo.insert.call_args[0][0]
        assert inserted_trade["strategy"] == f"mtf_macd:{symbol}"
        assert inserted_trade["quantity"] == 2.0


def test_max_total_exposure_filter(mock_dependencies, base_config_yaml):
    """Verify that new entries are blocked if total exposure exceeds max_total_exposure_pct."""
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=base_config_yaml):
        bot = TradingBot()
        bot.paper_trading = True
        bot.balance = 10000.0
        bot.equity = 10000.0
        
        # Current exposure limit: 150% of 10000 = $15000
        # Let's add a large position on BTC: size 0.25 @ 50,000 = $12500 exposure
        bot.symbol_states["BTC/USDT:USDT"]["open_positions"]["long"] = {
            "entry_price": 50000.0, "size": 0.25, "ts": 12345
        }
        
        # Try to enter ETH: size 2.0 @ 3,000 = $6000 exposure.
        # Total exposure would be $12500 + $6000 = $18500, which exceeds $15000 limit.
        candle = {"timestamp": 1600000000000, "close": 3000.0, "open": 2990.0, "high": 3010.0, "low": 2980.0}
        
        # We need to simulate the _on_1h_candle call
        # Mock feature engine to return dummy features
        bot.symbol_states["ETH/USDT:USDT"]["feature_engine"].process_candle = MagicMock(return_value={"atr_14": 50.0, "atr_pct": 1.5})
        bot.symbol_states["ETH/USDT:USDT"]["ensemble"].get_signal = MagicMock(return_value=Signal.LONG)
        bot.symbol_states["ETH/USDT:USDT"]["position_sizer"].calculate = MagicMock(return_value=2.0)
        
        with patch.object(bot, "_execute_signal") as mock_execute:
            bot._on_1h_candle("ETH/USDT:USDT", candle)
            # Should have blocked entry
            mock_execute.assert_not_called()


def test_emergency_stop_all_symbols(mock_dependencies, base_config_yaml):
    """Verify that emergency_stop closes positions and cancels orders across all symbols."""
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=base_config_yaml):
        bot = TradingBot()
        bot.paper_trading = False  # Set to False to verify exchange calls
        
        # Populate positions
        bot.symbol_states["BTC/USDT:USDT"]["open_positions"]["long"] = {"entry_price": 50000.0, "size": 0.1, "ts": 12345}
        bot.symbol_states["ETH/USDT:USDT"]["open_positions"]["short"] = {"entry_price": 3000.0, "size": 1.0, "ts": 12345}
        
        # Mock exchange API calls
        bot.exchange.fetch_open_orders = MagicMock(return_value=[{"id": "order1"}])
        bot.exchange.cancel_order = MagicMock()
        bot.exchange.create_market_sell_order = MagicMock()
        bot.exchange.create_market_buy_order = MagicMock()
        
        bot.emergency_stop()
        
        # Verify cancel order was called for all 4 configured symbols
        assert bot.exchange.cancel_order.call_count == 4
        # Verify market orders were created to close positions
        bot.exchange.create_market_sell_order.assert_called_with("BTC/USDT:USDT", 0.1)
        bot.exchange.create_market_buy_order.assert_called_with("ETH/USDT:USDT", 1.0)
        
        # Verify state is cleared
        assert len(bot.symbol_states["BTC/USDT:USDT"]["open_positions"]) == 0
        assert len(bot.symbol_states["ETH/USDT:USDT"]["open_positions"]) == 0
        assert len(bot.open_positions) == 0
        assert bot.running is False
