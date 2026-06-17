"""Main trading bot orchestrator.

Coordinates all layers:
1. Data ingestion (WebSocket + REST) — validates, stores, resamples
2. Feature computation
3. Strategy signal generation (ensemble routing)
4. Risk management
5. Order execution
6. Monitoring and logging

Lifecycle:
    init()     → Wire all components
    start()    → Connect WS, begin consuming candles
    _on_candle → Per-1H-candle trading logic
    stop()     → Graceful shutdown
"""

import os
import time
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from monitoring.logger import setup_logger
from monitoring.telegram_bot import TelegramAlerter

# Data pipeline
from data import (
    DataValidator, BitgetWSClient, BitgetRESTClient, OHLCVResampler,
    DatabaseManager, CandleRepository, TradeRepository, SignalRepository,
    has_websocket,
)

# Execution
from execution.exchange_adapter import ExchangeAdapter
from execution.order_manager import OrderManager
from execution.position_tracker import PositionTracker

# Feature engineering
from features.engine import FeatureEngine

# Strategies
from strategies.base import Signal
from strategies.mtf_macd import MTF_MACD_Elder
from strategies.mean_reversion import MeanReversion

# Ensemble
from ensemble.regime_classifier import RegimeClassifier, MarketRegime
from ensemble.router import EnsembleRouter

# Risk
from risk.position_sizer import KellyPositionSizer
from risk.circuit_breaker import CircuitBreaker, BreakerState
from risk.risk_monitor import RiskMonitor


class TradingBot:
    """Main trading bot — wires everything together."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        load_dotenv("config/.env")
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Update config with environment variables
        import os
        for key in ["BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
            val = os.getenv(key)
            if val:
                self.config[key] = val
        
        self.logger = setup_logger(self.config)
        self.telegram = TelegramAlerter(self.config)
        
        # ── Data Pipeline (Phase 1) ─────────────────────────
        self.db = DatabaseManager(self.config)
        self.candle_repo = CandleRepository(self.db)
        self.trade_repo = TradeRepository(self.db)
        self.signal_repo = SignalRepository(self.db)
        
        # REST client for backfilling
        self.rest_client: Optional[BitgetRESTClient] = None
        try:
            self.rest_client = BitgetRESTClient(self.config)
            self.logger.info("Bitget REST client initialized")
        except Exception as e:
            self.logger.warning(f"REST client unavailable: {e}")
        
        # ── Multi-Symbol Setup ────────────────────────────────
        self.symbols = self.config.get("exchange", {}).get("symbols", ["BTC/USDT:USDT"])
        self.symbol_states = {}
        
        for symbol in self.symbols:
            symbol_cfg = self._get_symbol_config(symbol)
            
            # Strategies
            strategies = {
                "mtf_macd": MTF_MACD_Elder(symbol_cfg),
                "mean_reversion": MeanReversion(symbol_cfg),
            }
            
            # Regime Classifier and Ensemble Router
            regime_classifier = RegimeClassifier(symbol_cfg)
            ensemble = EnsembleRouter(strategies, regime_classifier)
            
            # Feature Engine
            feature_engine = FeatureEngine(symbol_cfg)
            
            # Position Tracker
            position_tracker = PositionTracker(symbol_cfg)
            
            # Position Sizer
            position_sizer = KellyPositionSizer(symbol_cfg)
            
            self.symbol_states[symbol] = {
                "config": symbol_cfg,
                "feature_engine": feature_engine,
                "strategies": strategies,
                "regime_classifier": regime_classifier,
                "ensemble": ensemble,
                "position_tracker": position_tracker,
                "position_sizer": position_sizer,
                "open_positions": {},
                "latest_features": {},
            }

        # WebSocket clients (one per symbol)
        self.ws_clients = {}
        if has_websocket():
            for symbol in self.symbols:
                symbol_cfg = self.symbol_states[symbol]["config"]
                self.ws_clients[symbol] = BitgetWSClient(symbol_cfg)
        else:
            self.logger.warning("websocket-client not installed — live mode disabled")
        
        # ── Execution layer ──────────────────────────────────
        self.exchange = ExchangeAdapter(self.config)
        self.order_manager = OrderManager(self.config, self.exchange)
        
        # Single instances for fallback/backward compatibility
        first_symbol = self.symbols[0]
        self.position_tracker = self.symbol_states[first_symbol]["position_tracker"]
        self.feature_engine = self.symbol_states[first_symbol]["feature_engine"]
        self.strategies = self.symbol_states[first_symbol]["strategies"]
        self.regime_classifier = self.symbol_states[first_symbol]["regime_classifier"]
        self.ensemble = self.symbol_states[first_symbol]["ensemble"]
        self.position_sizer = self.symbol_states[first_symbol]["position_sizer"]
        
        # Risk (global)
        self.circuit_breaker = CircuitBreaker(self.config)
        self.risk_monitor = RiskMonitor(self.config)
        
        # State
        self.running = False
        self.paper_trading = self.config.get("bot", {}).get("mode", "paper") == "paper"
        self.initial_capital = self.config.get("risk", {}).get("initial_capital", 10000.0)
        self.equity = self.initial_capital
        self.balance = self.initial_capital
        self.recent_trades_pnl: list[float] = []  # Last 50 trades PnL
        self.open_positions: dict = {}  # Track paper positions: {side: {entry_price, size, ts}} (backward compatibility)
        self.circuit_breaker_alert_sent = False
        self.dashboard_server = None
        self.latest_features = {}
        self.scheduler_thread = None

    @property
    def ws_client(self):
        """Backward compatibility for ws_client."""
        if self.ws_clients:
            first_sym = list(self.ws_clients.keys())[0]
            return self.ws_clients[first_sym]
        return None

    def _get_symbol_config(self, symbol: str) -> dict:
        """Get merged config for a specific symbol."""
        import copy
        # Deep copy global config
        cfg = copy.deepcopy(self.config)
        
        # Set the symbol as the active exchange symbol
        cfg["exchange"]["symbols"] = [symbol]
        
        # Merge symbol-specific overrides
        overrides = self.config.get("symbols", {}).get(symbol, {})
        if not overrides:
            return cfg
            
        # Recursive merge of nested structures
        def merge_dicts(dict1, dict2):
            for k, v in dict2.items():
                if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
                    merge_dicts(dict1[k], v)
                else:
                    dict1[k] = v
        
        merge_dicts(cfg, overrides)
        return cfg
    
    def start(self):
        """Start the trading bot."""
        self.logger.info(f"🤖 Starting bocik v{self.config['bot']['version']}")
        self.logger.info(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        self.logger.info(f"Exchange: {self.config['exchange']['name']} "
                         f"({'testnet' if self.config['exchange'].get('testnet', False) else 'mainnet'})")
        
        self.running = True
        
        # Start Dashboard Server
        dash_cfg = self.config.get("dashboard", {})
        if dash_cfg.get("enabled", True):
            from dashboard.server import run_dashboard_server
            host = dash_cfg.get("host", "127.0.0.1")
            port = dash_cfg.get("port", 8080)
            try:
                self.dashboard_server = run_dashboard_server(self, host=host, port=port)
            except Exception as e:
                self.logger.error(f"Failed to start dashboard server: {e}")

        self.risk_monitor.set_initial_capital(self.initial_capital)
        
        # Prime FeatureEngine and strategy caches from REST API on startup
        if self.rest_client:
            try:
                self.logger.info("Priming feature engines and strategy caches from REST API...")
                for symbol in self.symbols:
                    self.logger.info(f"Priming caches for {symbol}...")
                    state = self.symbol_states[symbol]
                    df_1h = self.rest_client.fetch_days(timeframe="1h", days=30, symbol=symbol)
                    df_4h = self.rest_client.fetch_days(timeframe="4h", days=30, symbol=symbol)
                    df_1d = self.rest_client.fetch_days(timeframe="1d", days=45, symbol=symbol)
                    
                    if not df_1h.empty:
                        state["feature_engine"].prime_cache(
                            df_1h,
                            df_4h if not df_4h.empty else None,
                            df_1d if not df_1d.empty else None
                        )
                        # Compute initial features to populate self.latest_features
                        df_feats = state["feature_engine"].bulk_compute(
                            df_1h,
                            df_4h if not df_4h.empty else None,
                            df_1d if not df_1d.empty else None
                        )
                        if not df_feats.empty:
                            state["latest_features"] = df_feats.iloc[-1].to_dict()
                            self.latest_features = state["latest_features"]  # compatibility
                            self.logger.info(f"Feature engine primed for {symbol}. Calculated features count: {len(state['latest_features'])}")
                    
                    if not df_1d.empty:
                        mtf_macd = state["strategies"].get("mtf_macd")
                        if mtf_macd:
                            mtf_macd._d1_closes = []
                            for _, row in df_1d.iterrows():
                                mtf_macd.on_higher_tf_candle(row.to_dict(), "1d")
                            self.logger.info(
                                f"MTF MACD strategy primed for {symbol}. D1 closes: {len(mtf_macd._d1_closes)}, "
                                f"Trend: {mtf_macd.d1_trend}, MACD: {mtf_macd.d1_macd:.2f}, Signal: {mtf_macd.d1_signal:.2f}"
                            )
            except Exception as e:
                self.logger.error(f"Error priming caches on startup: {e}", exc_info=True)

        self.telegram.alert("🚀 bocik started")

        # ── Wire callbacks ───────────────────────────────────────
        if self.ws_clients and has_websocket():
            for symbol, ws in self.ws_clients.items():
                # Bind callbacks using lambda with symbol as default argument to capture value
                ws.on_candle("1m", lambda candle, s=symbol: self._on_1m_candle(s, candle))
                ws.on_candle("1h", lambda candle, s=symbol: self._on_1h_candle(s, candle))
                ws.on_candle("4h", lambda candle, s=symbol: self._on_4h_candle(s, candle))
                ws.on_candle("1d", lambda candle, s=symbol: self._on_1d_candle(s, candle))
                
                # Start WS (auto-primes resampler with historical data, then streams)
                backfill_hours = self.config.get("data", {}).get("backfill", {}).get("prime_hours", 24)
                ws.start(backfill_hours=backfill_hours)
                self.logger.info(f"WebSocket client started for {symbol} — listening for candles")
        else:
            self.logger.warning("Running without WebSocket — historical/backtest mode only")
            
        # Start periodic report scheduler
        reports_cfg = self.config.get("monitoring", {}).get("telegram", {}).get("reports", {})
        if reports_cfg.get("enabled", False):
            import threading
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, name="bocik-scheduler", daemon=True)
            self.scheduler_thread.start()
            self.logger.info("Telegram periodic report scheduler thread started.")
        
        self.logger.info("Bot initialized and running.")
    
    def stop(self, reason: str = "Manual stop"):
        """Gracefully stop the bot."""
        self.running = False
        
        # Stop Dashboard Server
        if hasattr(self, 'dashboard_server') and self.dashboard_server:
            try:
                self.dashboard_server.shutdown()
                self.dashboard_server.server_close()
                self.logger.info("Dashboard server stopped.")
            except Exception as e:
                self.logger.warning(f"Error stopping dashboard server: {e}")

        # Stop WebSocket clients
        for symbol, ws in self.ws_clients.items():
            try:
                ws.stop()
                self.logger.info(f"WebSocket client stopped for {symbol}")
            except Exception as e:
                self.logger.warning(f"Error stopping WebSocket for {symbol}: {e}")
        
        self.logger.warning(f"Bot stopped: {reason}")
        self.telegram.alert(f"⏹️ Bot stopped: {reason}")

    
    def _on_1m_candle(self, symbol: str, candle: dict):
        """Called on every validated 1m candle from WebSocket.
        
        Validation and resampling are handled by the WS client internally.
        Our job: persist to the database for backtesting and analysis.
        """
        if symbol == self.symbols[0]:
            self.candle_repo.insert(candle)
    
    def _on_1h_candle(self, symbol: str, candle: dict):
        """Main trading logic — called when a new 1H candle completes.

        Full pipeline:
            1. Compute features
            2. Circuit breaker check (with real equity)
            3. Check exits on existing positions
            4. Generate strategy signal
            5. Risk: position sizing
            6. Execute (paper or live)
            7. Update risk monitor
        """
        try:
            state = self.symbol_states[symbol]
            close = candle["close"]
            features = state["feature_engine"].process_candle(candle)
            
            if not features:
                return  # Not enough history yet
            
            state["latest_features"] = features
            self.latest_features = features  # compatibility
            
            # ── Check symbol specific positions ────────────
            open_pos = state["open_positions"]
            in_position = len(open_pos) > 0
            position_side = list(open_pos.keys())[0] if in_position else ""
            position_size = sum(p["size"] for p in open_pos.values())
            entry_price = open_pos.get(position_side, {}).get("entry_price", 0.0)
            
            # ── Update risk monitor with global equity ───────────
            total_unrealized_pnl = 0.0
            total_position_value = 0.0
            
            # Actually, let's calculate total unrealized PnL from self.symbol_states
            for sym, sym_state in self.symbol_states.items():
                for side, p in sym_state["open_positions"].items():
                    try:
                        sym_close = self.symbol_states[sym]["feature_engine"]._cache["1h"]["close"].iloc[-1]
                    except Exception:
                        sym_close = p["entry_price"]
                        
                    if side == "long":
                        unrealized = p["size"] * (sym_close - p["entry_price"])
                    else:
                        unrealized = p["size"] * (p["entry_price"] - sym_close)
                    total_unrealized_pnl += unrealized
                    total_position_value += p["size"] * sym_close
            
            # Update equity to include unrealized PnL (for circuit breaker check)
            self.equity = self.balance + total_unrealized_pnl
            
            self.risk_monitor.update(
                equity=self.equity,
                balance=self.balance,
                btc_price=close,
                position_size=position_size, # compatibility: this symbol's size
                position_entry=entry_price, # compatibility: this symbol's entry
            )
            # Override with the actual portfolio-wide unrealized PnL
            self.risk_monitor.unrealized_pnl = total_unrealized_pnl
            
            # ── Circuit breaker check (global) ───────────────────
            avg_atr = features.get("atr_14", 0)
            state_cb, reason = self.circuit_breaker.check(
                equity=self.equity,
                balance=self.balance,
                recent_trades_pnl=self.recent_trades_pnl,
                current_atr=avg_atr,
                avg_atr=avg_atr,  # Use current as proxy
            )
            
            if state_cb == BreakerState.HALTED:
                self.logger.warning(f"⛔ GLOBAL CIRCUIT BREAKER: {reason}")
                if not self.circuit_breaker_alert_sent:
                    from datetime import datetime
                    halt_reason = getattr(self.circuit_breaker, "halt_reason", "") or reason
                    halt_time = getattr(self.circuit_breaker, "halt_timestamp", None) or datetime.now()
                    msg = (
                        f"🚨 *CIRCUIT BREAKER ACTIVATED*\n"
                        f"Reason: `{halt_reason}`\n"
                        f"Time: `{halt_time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
                        f"Action required: Verify strategy and call `reset_circuit_breaker()`"
                    )
                    self.telegram.circuit_breaker(msg)
                    self.circuit_breaker_alert_sent = True
                return
            
            if state_cb == BreakerState.WARNING:
                self.logger.warning(f"⚠️ CIRCUIT WARNING: {reason} — skipping signal for {symbol}")
                return
            
            # ── Check existing positions for this symbol (exit logic) ────
            if in_position:
                exit_reason = state["position_tracker"].update(candle)
                if exit_reason != "hold":
                    self.logger.info(f"[{symbol}] Position exit triggered by tracker: {exit_reason}")
                    exit_signal = Signal.SHORT if position_side == "long" else Signal.LONG
                    self._close_position(symbol, position_side, candle, exit_signal, reason=exit_reason)
                    return  # Don't enter same bar
            
            # ── Generate strategy signal via ensemble router ─────────
            signal = state["ensemble"].get_signal(candle, features)
            
            # ── Check opposite strategy signal exit ───────────────
            if in_position:
                is_opposite = (position_side == "long" and signal == Signal.SHORT) or \
                              (position_side == "short" and signal == Signal.LONG)
                if is_opposite:
                    bars_held = state["position_tracker"].bars_held
                    min_hold = state["config"].get("strategies", {}).get("mtf_macd_elder", {}).get("exit", {}).get("min_hold_bars", 6)
                    if bars_held >= min_hold:
                        self.logger.info(f"[{symbol}] Position exit triggered by opposite strategy signal after {bars_held} bars")
                        exit_signal = Signal.SHORT if position_side == "long" else Signal.LONG
                        self._close_position(symbol, position_side, candle, exit_signal, reason="signal")
                        return
                
                # If we are in position and no exit occurred, we stay flat/hold
                return
            
            if signal == Signal.FLAT:
                return
            
            # ── Global exposure & concurrency checks ────────────
            total_exposure_val = 0.0
            total_active_positions = 0
            for sym, sym_state in self.symbol_states.items():
                for side, p in sym_state["open_positions"].items():
                    total_exposure_val += p["size"] * p["entry_price"]
                    total_active_positions += 1

            # Max concurrent positions (aligned with multi-asset backtest)
            max_concurrent = self.config.get("risk", {}).get("max_concurrent_positions", 3)
            if total_active_positions >= max_concurrent:
                self.logger.debug(f"[{symbol}] Entry blocked by max concurrent positions ({total_active_positions}/{max_concurrent})")
                self.signal_repo.insert({
                    "timestamp": candle["timestamp"],
                    "strategy": f"mtf_macd:{symbol}",
                    "signal": signal.value,
                    "executed": False,
                    "reject_reason": f"max_concurrent({total_active_positions}/{max_concurrent})",
                })
                return

            # ── Correlation risk check ──────────────────────
            # Warn when high-correlation pairs are simultaneously in position.
            # Conditional correlations from multi-asset backtest (OOS, 2023-2026):
            HIGH_CORR_PAIRS = {
                ("BTC/USDT:USDT", "ETH/USDT:USDT"): 0.48,
                ("BTC/USDT:USDT", "LTC/USDT:USDT"): 0.48,
                ("BTC/USDT:USDT", "SOL/USDT:USDT"): 0.44,
            }
            symbols_in_position = set()
            for sym, sym_state in self.symbol_states.items():
                if sym_state["open_positions"]:
                    symbols_in_position.add(sym)

            if symbol not in symbols_in_position:  # Only check when about to enter
                for (s1, s2), corr in HIGH_CORR_PAIRS.items():
                    if symbol in (s1, s2):
                        other = s2 if symbol == s1 else s1
                        if other in symbols_in_position:
                            self.logger.warning(
                                f"[{symbol}] HIGH CORRELATION: entering while {other.split('/')[0]} "
                                f"already in position (conditional corr={corr:.2f}). "
                                f"Consider reducing position size."
                            )

            max_exposure_pct = self.config.get("risk", {}).get("position_sizing", {}).get("max_total_exposure_pct", 60)
            max_exposure_limit = (max_exposure_pct / 100.0) * self.balance
            
            # Calculate streaks from recent trades PnL
            consecutive_losses = 0
            consecutive_wins = 0
            for pnl in reversed(self.recent_trades_pnl):
                if pnl < 0:
                    if consecutive_wins > 0:
                        break
                    consecutive_losses += 1
                else:  # pnl >= 0
                    if consecutive_losses > 0:
                        break
                    consecutive_wins += 1

            # ── Risk: Position sizing ────────────────────────────
            coin_size = state["position_sizer"].calculate(
                capital=self.balance,
                btc_price=close,
                current_atr=avg_atr,
                avg_atr=avg_atr,
                consecutive_losses=consecutive_losses,
                consecutive_wins=consecutive_wins,
            )
            
            if coin_size <= 0:
                self.logger.debug(f"[{symbol}] Position sizer returned 0 — skipping trade")
                self.signal_repo.insert({
                    "timestamp": candle["timestamp"],
                    "strategy": f"mtf_macd:{symbol}",
                    "signal": signal.value,
                    "executed": False,
                    "reject_reason": "position_sizer_zero",
                })
                return
            
            # Enforce max total exposure check
            new_position_value = coin_size * close
            if total_exposure_val + new_position_value > max_exposure_limit:
                self.logger.warning(
                    f"[{symbol}] Entry blocked by Max Total Exposure filter! "
                    f"Current Exposure: ${total_exposure_val:.2f}, Proj Exposure: ${total_exposure_val + new_position_value:.2f}, Limit: ${max_exposure_limit:.2f}"
                )
                self.signal_repo.insert({
                    "timestamp": candle["timestamp"],
                    "strategy": f"mtf_macd:{symbol}",
                    "signal": signal.value,
                    "executed": False,
                    "reject_reason": f"max_exposure({total_exposure_val:.0f}/{max_exposure_limit:.0f})",
                })
                return
            
            # ── Execute ──────────────────────────────────────────
            self._execute_signal(symbol, signal, candle, coin_size, atr=avg_atr, atr_pct=features.get("atr_pct", 2.0))
            
            # ── Periodic risk snapshot ───────────────────────────
            self._log_risk_snapshot()
            
        except Exception as e:
            self.logger.error(f"[{symbol}] 1H handler error: {e}", exc_info=True)
            self.telegram.error(f"[{symbol}] 1H handler error: {e}")
    
    def _on_4h_candle(self, symbol: str, candle: dict):
        """Called when a new 4H candle completes — reassess market regime."""
        try:
            state = self.symbol_states[symbol]
            features = state["latest_features"]
            if features:
                regime = state["regime_classifier"].current_regime
                self.logger.info(
                    f"[{symbol}] 4H Regime: {regime.value.upper()} | "
                    f"ADX={features.get('adx_14',0):.1f} | "
                    f"BBw={features.get('bb_width',0):.3f} | "
                    f"ATR={features.get('atr_14',0):.1f}"
                )
        except Exception as e:
            self.logger.error(f"[{symbol}] 4H handler error: {e}", exc_info=True)
            
    def _on_1d_candle(self, symbol: str, candle: dict):
        """Called when a new 1D candle completes."""
        try:
            self.logger.info(f"📅 [{symbol}] 1D candle completed: close={candle['close']}. Updating MTF MACD Elder D1 trend...")
            state = self.symbol_states[symbol]
            mtf_macd = state["strategies"].get("mtf_macd")
            if mtf_macd:
                mtf_macd.on_higher_tf_candle(candle, "1d")
                self.logger.info(f"[{symbol}] MTF MACD D1 trend updated: {mtf_macd.d1_trend}")
        except Exception as e:
            self.logger.error(f"[{symbol}] 1D candle handler error: {e}", exc_info=True)
    
    # ─── Position Management ────────────────────────────────
    
    def _close_position(self, symbol: str, side: str, candle: dict, exit_signal: Signal, reason: str = "trailing_stop"):
        """Close an open position and record the trade."""
        state = self.symbol_states[symbol]
        if side not in state["open_positions"]:
            return
        pos = state["open_positions"].pop(side)
        state["position_tracker"].exit()
        
        # Sync with compatibility dict
        self.open_positions.pop(f"{symbol}:{side}", None)
        self.open_positions.pop(side, None)
        
        entry_price = pos["entry_price"]
        size = pos["size"]
        
        if self.paper_trading:
            exit_price = candle["close"]
            commission = 0.0006 * 2  # standard taker fee simulated
            slippage = 0.0005 * 2
            if side == "long":
                gross_return = (exit_price - entry_price) / entry_price
            else:
                gross_return = (entry_price - exit_price) / entry_price
            net_return = gross_return - commission - slippage
            pnl = size * entry_price * net_return
        else:
            # LIVE / REAL execution
            opposite_side = "sell" if side == "long" else "buy"
            self.logger.info(f"LIVE: Executing exit order for {symbol} {side.upper()} position...")
            order_res = self.order_manager.place_order_maker_only(
                symbol=symbol,
                side=opposite_side,
                amount=size,
                fallback_to_market=True,  # Exits must fill!
            )
            
            if order_res["status"] == "failed" or order_res["filled"] <= 0:
                self.logger.critical(f"LIVE: Exit order failed completely! Manual intervention required.")
                self.telegram.alert(f"🚨 LIVE EXIT FAILED for {symbol} {side.upper()}! Status: {order_res['status']}", force=True)
                
                # Restore state on failure
                state["open_positions"][side] = pos
                self.open_positions[f"{symbol}:{side}"] = pos
                self.open_positions[side] = pos
                return

            exit_price = order_res["average"]
            actual_filled = order_res["filled"]
            
            if actual_filled < size:
                self.logger.warning(f"LIVE: Position only partially closed! Filled: {actual_filled}/{size}")
            
            # Calculate actual PnL (maker fee 0.02% * 2)
            maker_fee = self.config.get("exchange", {}).get("fees", {}).get("maker", 0.0002)
            commission_cost = maker_fee * 2
            
            if side == "long":
                gross_return = (exit_price - entry_price) / entry_price
            else:
                gross_return = (entry_price - exit_price) / entry_price
                
            net_return = gross_return - commission_cost
            pnl = actual_filled * entry_price * net_return
            size = actual_filled

        self.balance += pnl
        self.equity = self.balance
        self.recent_trades_pnl.append(pnl)
        if len(self.recent_trades_pnl) > 50:
            self.recent_trades_pnl = self.recent_trades_pnl[-50:]
        self.risk_monitor.record_trade(pnl, pnl > 0)
        self.circuit_breaker.record_trade(pnl, is_closing=True)
        pnl_pct = (pnl / (size * entry_price)) * 100 if size > 0 else 0.0
        emoji = "✅" if pnl > 0 else "❌"
        
        # Calculate slippage in basis points (bps)
        theoretical_entry = pos.get("theoretical_entry_price", entry_price)
        theoretical_exit = candle["close"]
        
        if side == "long":
            entry_slippage_bps = ((entry_price - theoretical_entry) / theoretical_entry) * 10000
            exit_slippage_bps = ((theoretical_exit - exit_price) / theoretical_exit) * 10000
        else:  # short
            entry_slippage_bps = ((theoretical_entry - entry_price) / theoretical_entry) * 10000
            exit_slippage_bps = ((exit_price - theoretical_exit) / theoretical_exit) * 10000
        total_slippage_bps = entry_slippage_bps + exit_slippage_bps
        
        self.logger.info(
            f"{emoji} Trade: {symbol} {side.upper()} | Reason={reason} | "
            f"Entry={entry_price:.2f} Exit={exit_price:.2f} | "
            f"Size={size:.6f} | PnL=${pnl:.2f} ({pnl_pct:.2f}%) | "
            f"Slippage: Entry={entry_slippage_bps:+.1f} bps, Exit={exit_slippage_bps:+.1f} bps, Total={total_slippage_bps:+.1f} bps | "
            f"Equity=${self.equity:.2f}"
        )
        if abs(pnl) > 50 or not self.paper_trading:
            self.telegram.alert(
                f"{emoji} {symbol} {side.upper()} closed ({reason}): ${pnl:.2f}\n"
                f"Entry: {entry_price:.2f} → Exit: {exit_price:.2f}\n"
                f"Slippage: Entry={entry_slippage_bps:+.1f} bps, Exit={exit_slippage_bps:+.1f} bps, Total={total_slippage_bps:+.1f} bps\n"
                f"Equity: ${self.equity:.2f}"
            )
        self.trade_repo.insert({
            "entry_time": pos["ts"],
            "exit_time": candle["timestamp"],
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": size,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "strategy": f"mtf_macd:{symbol}",
            "exit_reason": reason,
            "theoretical_entry_price": theoretical_entry,
            "theoretical_exit_price": theoretical_exit,
        })
    
    def _execute_signal(self, symbol: str, signal: Signal, candle: dict, size_coin: float, atr: float = 0.0, atr_pct: float = 2.0):
        """Execute a trading signal (paper or live)."""
        close = candle["close"]
        side = "long" if signal == Signal.LONG else "short"
        state = self.symbol_states[symbol]

        if self.paper_trading:
            slippage = 0.0005
            if side == "long":
                entry_price = close * (1 + slippage)
            else:
                entry_price = close * (1 - slippage)
            executed_size = size_coin
        else:
            # LIVE / REAL execution
            order_side = "buy" if side == "long" else "sell"
            self.logger.info(f"LIVE: Executing entry order for {symbol} {side.upper()}...")
            order_res = self.order_manager.place_order_maker_only(
                symbol=symbol,
                side=order_side,
                amount=size_coin,
                fallback_to_market=False,  # Don't chase entries with Market!
            )
            
            if order_res["status"] == "failed" or order_res["filled"] <= 0:
                self.logger.warning(f"LIVE: Entry order failed/unfilled. Skipping signal.")
                return

            entry_price = order_res["average"]
            executed_size = order_res["filled"]

        pos_data = {
            "entry_price": entry_price, "size": executed_size,
            "ts": candle["timestamp"], "side": side,
            "highest": close, "lowest": close,
            "symbol": symbol,
            "theoretical_entry_price": close,
        }
        state["open_positions"][side] = pos_data
        
        # Sync with compatibility dict
        self.open_positions[f"{symbol}:{side}"] = pos_data
        self.open_positions[side] = pos_data
        
        # Initialize position tracker matching the backtester exit chain
        state["position_tracker"].enter(
            side=side,
            entry_price=entry_price,
            quantity=executed_size,
            atr=atr,
            atr_pct=atr_pct,
            timestamp=candle["timestamp"],
            symbol=symbol
        )
        
        self.logger.info(
            f"📈 {symbol} {side.upper()}: {executed_size:.6f} @ ${entry_price:.2f} (Tracker levels: SL=${state['position_tracker'].position.stop_loss:.2f}, TP=${state['position_tracker'].position.take_profit:.2f}, trail={state['position_tracker'].position.trailing_distance*100:.1f}%)"
        )
        self.signal_repo.insert({
            "timestamp": candle["timestamp"],
            "strategy": f"mtf_macd:{symbol}",
            "signal": signal.value,
            "executed": True,
        })
    
    def _log_risk_snapshot(self):
        """Periodically log risk metrics (every hour)."""
        now = time.time()
        if not hasattr(self, '_last_risk_log'):
            self._last_risk_log = 0.0
        if now - self._last_risk_log >= 3600:
            self._last_risk_log = now
            snap = self.risk_monitor.snapshot()
            self.logger.info(
                f"📊 Risk: Equity=${snap['equity']:,.2f} | "
                f"Return={snap['total_return_pct']}% | "
                f"DD={snap['current_drawdown_pct']}% | "
                f"Trades={snap['trade_count']} WR={snap['win_rate']}%"
            )
    
    def emergency_stop(self):
        """Emergency stop — close all positions immediately."""
        self.logger.critical("EMERGENCY STOP TRIGGERED")
        self.circuit_breaker.emergency_stop("Manual emergency")
        
        for symbol in self.symbols:
            state = self.symbol_states[symbol]
            if not self.paper_trading:
                # Cancel all open orders on exchange for this symbol
                try:
                    open_orders = self.exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        order_id = order.get("id")
                        self.logger.warning(f"LIVE: Emergency cancel order {order_id} for {symbol}")
                        self.exchange.cancel_order(order_id, symbol)
                except Exception as e:
                    self.logger.error(f"LIVE: Failed to cancel open orders for {symbol} in emergency: {e}")

                # Close all active positions for this symbol at market price
                for side, pos in list(state["open_positions"].items()):
                    size = pos["size"]
                    opposite_side = "sell" if side == "long" else "buy"
                    self.logger.warning(f"LIVE: Emergency market close for {symbol} {side.upper()} position of size {size}")
                    try:
                        if opposite_side == "sell":
                            self.exchange.create_market_sell_order(symbol, size)
                        else:
                            self.exchange.create_market_buy_order(symbol, size)
                    except Exception as e:
                        self.logger.critical(f"LIVE: Failed to emergency market close {symbol} {side.upper()}: {e}")
            
            # Clear symbol-specific positions
            state["open_positions"].clear()
            state["position_tracker"].exit()
            
        # For paper or live, clear local state and stop bot
        self.open_positions.clear()
        self.stop("Emergency stop")

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker and clear recent losses to resume trading."""
        self.logger.info("Manually resetting circuit breaker...")
        self.recent_trades_pnl = []  # Clear trade history to clear consecutive losses
        self.circuit_breaker.reset_manual_halt()
        self.circuit_breaker_alert_sent = False
        self.telegram.alert("🔄 Circuit breaker manually reset. Bot resumed trading.", force=True)

    def _scheduler_loop(self):
        """Background thread loop to run daily/weekly reports."""
        import datetime
        self.logger.info("Periodic report scheduler loop started.")
        last_sent_date = None
        
        while self.running:
            try:
                reports_cfg = self.config.get("monitoring", {}).get("telegram", {}).get("reports", {})
                if not reports_cfg.get("enabled", False):
                    time.sleep(30)
                    continue
                
                now = datetime.datetime.now()
                today_str = now.strftime("%Y-%m-%d")
                
                # Check target time
                target_time_str = reports_cfg.get("time", "22:00")
                try:
                    target_hour, target_minute = map(int, target_time_str.split(":"))
                except Exception:
                    target_hour, target_minute = 22, 0
                
                if now.hour == target_hour and now.minute == target_minute:
                    if last_sent_date != today_str:
                        interval = reports_cfg.get("interval", "both")
                        weekly_day = reports_cfg.get("weekly_day", "Sunday")
                        
                        should_send_daily = False
                        should_send_weekly = False
                        
                        if interval == "daily":
                            should_send_daily = True
                        elif interval == "weekly":
                            current_day = now.strftime("%A")  # e.g., "Sunday"
                            if current_day.lower() == weekly_day.lower():
                                should_send_weekly = True
                        elif interval == "both":
                            current_day = now.strftime("%A")
                            if current_day.lower() == weekly_day.lower():
                                should_send_weekly = True
                            else:
                                should_send_daily = True
                        
                        if should_send_weekly:
                            self.logger.info("Triggering weekly Telegram report...")
                            self._send_periodic_report("weekly")
                            last_sent_date = today_str
                        elif should_send_daily:
                            self.logger.info("Triggering daily Telegram report...")
                            self._send_periodic_report("daily")
                            last_sent_date = today_str
            except Exception as e:
                self.logger.error(f"Error in periodic report scheduler loop: {e}", exc_info=True)
                
            time.sleep(30)

    def _send_periodic_report(self, interval: str):
        """Generate and send a periodic performance report to Telegram with an equity curve chart."""
        import sqlite3
        import datetime
        from monitoring.chart_generator import generate_equity_chart
        
        db_path = self.config.get("data", {}).get("database", {}).get("path", "./data/trading.db")
        trades = []
        
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy, exit_reason "
                    "FROM trades WHERE exit_time IS NOT NULL ORDER BY exit_time ASC"
                )
                rows = cursor.fetchall()
                trades = [dict(row) for row in rows]
                conn.close()
            except Exception as e:
                self.logger.error(f"Error reading database for periodic report: {e}")

        now = datetime.datetime.now()
        now_ms = int(now.timestamp() * 1000)
        
        if interval == "daily":
            since_ms = now_ms - (24 * 60 * 60 * 1000)
            interval_name = "RAPORT DOBOWY"
        else:
            since_ms = now_ms - (7 * 24 * 60 * 60 * 1000)
            interval_name = "RAPORT TYGODNIOWY"

        # Filter trades closed in the period
        period_trades = [t for t in trades if t.get("exit_time", 0) >= since_ms]
        period_pnl = sum(t["pnl"] for t in period_trades if t.get("pnl") is not None)
        period_trade_count = len(period_trades)
        
        period_wins = [t for t in period_trades if t.get("pnl", 0.0) > 0]
        period_win_rate = (len(period_wins) / period_trade_count * 100) if period_trade_count > 0 else 0.0
        
        # Per symbol statistics in the period
        symbol_stats = {}
        for t in period_trades:
            strat = t.get("strategy", "")
            # Extract symbol: e.g. mtf_macd:BTC/USDT:USDT -> BTC/USDT
            symbol = strat.split(":", 1)[1].split(":")[0] if ":" in strat else "BTC/USDT"
            
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"pnl": 0.0, "count": 0}
            symbol_stats[symbol]["pnl"] += t.get("pnl", 0.0)
            symbol_stats[symbol]["count"] += 1
            
        # Format per-symbol breakdown
        symbols_summary = ""
        if symbol_stats:
            symbols_summary = "\n*Wyniki per instrument:*\n"
            for sym, stat in symbol_stats.items():
                pnl_val = stat["pnl"]
                sign = "+" if pnl_val >= 0 else ""
                symbols_summary += f"• `{sym}`: *{sign}${pnl_val:,.2f}* ({stat['count']} trans.)\n"
        else:
            symbols_summary = "\n*Brak zamkniętych transakcji w tym okresie.*\n"
            
        # Get overall stats from risk monitor
        current_equity = self.equity
        current_balance = self.balance
        
        current_dd = 0.0
        if hasattr(self, "risk_monitor") and self.risk_monitor:
            snap = self.risk_monitor.snapshot()
            current_dd = snap.get("current_drawdown_pct", 0.0)
            
        starting_capital = current_equity - period_pnl
        if starting_capital <= 0:
            starting_capital = self.initial_capital
        period_return_pct = (period_pnl / starting_capital * 100) if starting_capital > 0 else 0.0
        
        pnl_sign = "+" if period_pnl >= 0 else ""
        emoji = "📈" if period_pnl >= 0 else "📉"
        
        message = (
            f"{emoji} *{interval_name} ({now.strftime('%d.%m.%Y %H:%M')})*\n\n"
            f"💰 *Stan konta (Equity):* `${current_equity:,.2f}`\n"
            f"💵 *Saldo (Balance):* `${current_balance:,.2f}`\n"
            f"📉 *Aktualne obsunięcie (DD):* `{current_dd:.2f}%`\n\n"
            f"📊 *Wynik okresu:* *{pnl_sign}${period_pnl:,.2f}* ({period_return_pct:+.2f}%)\n"
            f"🔄 *Zamknięte transakcje:* `{period_trade_count}`\n"
            f"🎯 *Win Rate okresu:* `{period_win_rate:.1f}%`\n"
            f"{symbols_summary}"
        )
        
        # Build full equity series from all trades to show overall history
        equity_series = [self.initial_capital]
        for t in trades:
            pnl_val = t.get("pnl", 0.0)
            equity_series.append(equity_series[-1] + pnl_val)
            
        if len(equity_series) == 1:
            equity_series.append(equity_series[0])
            
        chart_dir = self.config.get("paths", {}).get("logs_dir", "./logs")
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = os.path.join(chart_dir, "equity_curve.png")
        
        chart_generated = generate_equity_chart(equity_series, chart_path)
        
        if chart_generated and os.path.exists(chart_path):
            try:
                with open(chart_path, "rb") as f:
                    photo_bytes = f.read()
                self.telegram.send_photo(photo_bytes, caption=message)
                self.logger.info(f"Periodic report with chart sent to Telegram.")
            except Exception as e:
                self.logger.error(f"Failed to send periodic report photo to Telegram: {e}")
                self.telegram.alert(message, force=True)
        else:
            self.telegram.alert(message, force=True)



def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="bocik — BTC Trading Bot")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], help="Override mode")
    args = parser.parse_args()
    
    bot = TradingBot(args.config)
    
    try:
        bot.start()
        # Keep alive (in production, WebSocket callbacks handle this)
        while bot.running:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop("Keyboard interrupt")
    except Exception as e:
        bot.logger.critical(f"Fatal error: {e}")
        bot.emergency_stop()


if __name__ == "__main__":
    main()
