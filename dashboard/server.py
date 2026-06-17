import json
import os
import re
import glob
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer
import threading
from loguru import logger
import yaml

class DashboardHandler(BaseHTTPRequestHandler):
    bot = None  # Reference to TradingBot instance (set from outside)

    def log_message(self, format, *args):
        # Prevent spamming the main logger with HTTP requests unless it's a warning/error
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_static_file('index.html', 'text/html')
        elif self.path == '/api/status':
            self._send_json(self._get_status())
        elif self.path == '/api/trades':
            self._send_json(self._get_trades())
        elif self.path == '/api/signals':
            self._send_json(self._get_signals())
        elif self.path == '/api/logs':
            self._send_json(self._get_logs())
        elif self.path == '/api/analytics':
            self._send_json(self._get_analytics())
        elif self.path == '/api/export/csv':
            self._export_csv()
        elif self.path == '/api/export/json':
            self._export_json()
        else:
            self.send_error(404, 'File Not Found')

    def do_POST(self):
        if self.path == '/api/reset':
            if self.bot:
                self.bot.reset_circuit_breaker()
                self._send_json({"status": "ok", "message": "Circuit breaker reset successfully"})
            else:
                self.send_error(500, "Bot instance not loaded")
        elif self.path == '/api/emergency':
            if self.bot:
                self.bot.emergency_stop()
                self._send_json({"status": "ok", "message": "Emergency stop triggered"})
            else:
                self.send_error(500, "Bot instance not loaded")
        elif self.path == '/api/settings':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                settings_data = json.loads(post_data)
                self._save_settings(settings_data)
                self._send_json({"status": "ok", "message": "Settings updated successfully"})
            except Exception as e:
                logger.error(f"Error saving settings: {e}")
                self.send_error(400, f"Invalid request payload: {e}")
        else:
            self.send_error(404, 'Endpoint Not Found')

    def _serve_static_file(self, filename, content_type):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            self.send_error(404, f"File {filename} not found")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error serving file: {e}")

    def _send_json(self, data):
        try:
            response_body = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as e:
            logger.error(f"Error serialization JSON: {e}")
            self.send_error(500, "Internal Server Error")

    def _get_status(self):
        if not self.bot:
            return {"error": "Bot not initialized"}

        # Get active positions (across all symbols)
        active_positions = []
        for symbol in self.bot.symbols:
            sym_state = self.bot.symbol_states.get(symbol, {})
            open_pos = sym_state.get("open_positions", {})
            for side, pos_data in open_pos.items():
                active_positions.append({
                    "symbol": symbol,
                    "side": side.upper(),
                    "size": pos_data.get("size", 0.0),
                    "entry_price": pos_data.get("entry_price", 0.0),
                    "ts": pos_data.get("ts", 0),
                    "highest": pos_data.get("highest", 0.0),
                    "lowest": pos_data.get("lowest", 0.0),
                })

        active_pos = active_positions[0] if active_positions else None

        # Calculate estimated win rate and stats
        total_trades = 0
        win_rate = 0.0
        max_dd = 0.0
        if hasattr(self.bot, 'risk_monitor'):
            snap = self.bot.risk_monitor.snapshot()
            total_trades = snap.get('trade_count', 0)
            win_rate = snap.get('win_rate', 0.0)
            max_dd = snap.get('current_drawdown_pct', 0.0)

        # Get circuit breaker details
        cb_state = "NORMAL"
        cb_reason = ""
        cb_manual = False
        if hasattr(self.bot, 'circuit_breaker'):
            cb_state = self.bot.circuit_breaker.state.name if hasattr(self.bot.circuit_breaker.state, 'name') else str(self.bot.circuit_breaker.state)
            cb_reason = getattr(self.bot.circuit_breaker, "halt_reason", "") or ""
            cb_manual = getattr(self.bot.circuit_breaker, "manual_halted", False)

        # Get real-time price & 24h change for all symbols
        tickers = {}
        for symbol in self.bot.symbols:
            ws = self.bot.ws_clients.get(symbol)
            last_price = 0.0
            price_24h = 0.0
            change_pct = 0.0
            if ws:
                last_price = ws.last_price
                price_24h = ws.price_24h_ago
                
            # If price_24h not available via WS, try candle repository (only for first symbol)
            if price_24h <= 0 and symbol == self.bot.symbols[0] and hasattr(self.bot, "candle_repo") and self.bot.candle_repo:
                try:
                    import time
                    now_ms = int(time.time() * 1000)
                    target_ts = now_ms - 24 * 60 * 60 * 1000
                    candles = self.bot.candle_repo.get_range(target_ts - 150_000, target_ts + 150_000, limit=1)
                    if candles:
                        price_24h = candles[0]["close"]
                except Exception as e:
                    logger.error(f"Error getting 24h ago price from DB: {e}")

            if price_24h > 0 and last_price > 0:
                change_pct = ((last_price - price_24h) / price_24h) * 100

            tickers[symbol] = {
                "last_price": last_price,
                "price_24h": price_24h,
                "change_24h_pct": change_pct
            }

        # Keep legacy compatibility values for first symbol
        first_symbol = self.bot.symbols[0] if self.bot.symbols else "BTC/USDT:USDT"
        first_ticker = tickers.get(first_symbol, {"last_price": 0.0, "price_24h": 0.0, "change_24h_pct": 0.0})
        last_price = first_ticker["last_price"]
        change_24h_pct = first_ticker["change_24h_pct"]

        # Get indicators / feature values and regime classification details per symbol
        proximity = {}
        regimes = {}
        for symbol in self.bot.symbols:
            sym_state = self.bot.symbol_states.get(symbol)
            if not sym_state:
                continue

            sym_prox = {
                "mtf_macd": {
                    "d1_trend": "FLAT",
                    "d1_macd": 0.0,
                    "d1_signal": 0.0,
                    "d1_hist": 0.0,
                    "macd": 0.0,
                    "macd_signal": 0.0,
                    "macd_hist": 0.0,
                    "macd_cross": 0.0,
                    "volume_sma_ratio": 1.0,
                    "volume_mult": 1.2,
                    "require_volume": True,
                }
            }

            # Populate from MTF MACD Elder
            mtf_macd = sym_state["strategies"].get("mtf_macd") if "strategies" in sym_state else None
            if mtf_macd:
                sym_prox["mtf_macd"].update({
                    "d1_trend": getattr(mtf_macd, "d1_trend", "FLAT"),
                    "d1_macd": getattr(mtf_macd, "d1_macd", 0.0),
                    "d1_signal": getattr(mtf_macd, "d1_signal", 0.0),
                    "d1_hist": getattr(mtf_macd, "d1_macd", 0.0) - getattr(mtf_macd, "d1_signal", 0.0),
                    "volume_mult": getattr(mtf_macd, "volume_mult", 1.2),
                    "require_volume": getattr(mtf_macd, "require_volume", True),
                })

            # Update indicators from latest features if available
            lf = sym_state.get("latest_features")
            if lf:
                sym_prox["mtf_macd"].update({
                    "macd": lf.get("macd", 0.0),
                    "macd_signal": lf.get("macd_signal", 0.0),
                    "macd_hist": lf.get("macd_hist", 0.0),
                    "macd_cross": lf.get("macd_cross", 0.0),
                    "volume_sma_ratio": lf.get("volume_sma_ratio", 1.0),
                })

            proximity[symbol] = sym_prox

            # Get market regime classification details
            if "regime_classifier" in sym_state:
                regimes[symbol] = sym_state["regime_classifier"].get_regime_metadata()

        compat_proximity = proximity.get(first_symbol, {})
        compat_regime = regimes.get(first_symbol, {})

        return {
            "bot_name": self.bot.config.get("bot", {}).get("name", "bocik"),
            "version": self.bot.config.get("bot", {}).get("version", "0.1.0"),
            "mode": self.bot.config.get("bot", {}).get("mode", "paper").upper(),
            "running": self.bot.running,
            "exchange": self.bot.config.get("exchange", {}).get("name", "bitget"),
            "ws_inst_type": self.bot.config.get("exchange", {}).get("ws_inst_type", "USDT-FUTURES"),
            "ping": 42, # Mock latency
            "equity": self.bot.equity,
            "balance": self.bot.balance,
            "initial_capital": self.bot.initial_capital,
            "active_position": active_pos,
            "active_positions": active_positions,
            "btc_price": last_price,
            "btc_change_24h": change_24h_pct,
            "tickers": tickers,
            "circuit_breaker": {
                "state": cb_state,
                "reason": cb_reason,
                "manual_halted": cb_manual,
            },
            "stats": {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "max_drawdown": max_dd,
            },
            "proximity": compat_proximity,
            "proximities": proximity,
            "regime": compat_regime,
            "regimes": regimes,
            "config": {
                "risk": self.bot.config.get("risk", {}),
                "strategies": self.bot.config.get("strategies", {}),
                "meta_labeling": self.bot.config.get("meta_labeling", {}),
                "telegram": {
                    "chat_id": self.bot.config.get("TELEGRAM_CHAT_ID", ""),
                    "bot_token": self.bot.config.get("TELEGRAM_BOT_TOKEN", ""),
                }
            }
        }

    def _get_trades(self):
        import sqlite3
        db_path = self.bot.config.get("data", {}).get("database", {}).get("path", "./data/trading.db")
        if not os.path.exists(db_path):
            return []
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy, exit_reason, theoretical_entry_price, theoretical_exit_price FROM trades ORDER BY exit_time DESC LIMIT 50")
            rows = cursor.fetchall()
            trades = [dict(row) for row in rows]
            conn.close()
            return trades
        except Exception as e:
            logger.error(f"Error querying trades database: {e}")
            return []

    def _get_signals(self):
        import sqlite3
        db_path = self.bot.config.get("data", {}).get("database", {}).get("path", "./data/trading.db")
        if not os.path.exists(db_path):
            return []
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, strategy, signal, confidence, regime, executed, reject_reason FROM signals ORDER BY timestamp DESC LIMIT 50")
            rows = cursor.fetchall()
            signals = [dict(row) for row in rows]
            conn.close()
            return signals
        except Exception as e:
            logger.error(f"Error querying signals database: {e}")
            return []

    def _get_logs(self):
        logs_dir = self.bot.config.get("paths", {}).get("logs_dir", "./logs")
        if not os.path.exists(logs_dir):
            return []
        
        log_files = glob.glob(os.path.join(logs_dir, "*.log"))
        if not log_files:
            return []
        
        # Sort log files by modified time and read the latest
        latest_log = max(log_files, key=os.path.getmtime)
        try:
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Extract last 100 lines and format them
            last_lines = lines[-100:]
            formatted_logs = []
            for line in last_lines:
                # Regex to parse loguru output format: YYYY-MM-DD HH:MM:SS.ms | LEVEL | msg
                match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| (.*)$", line.strip())
                if match:
                    timestamp, level, message = match.groups()
                    # Strip details from message if it's too long
                    formatted_logs.append({
                        "timestamp": timestamp.split(" ")[1], # Only show HH:MM:SS.ms
                        "level": level,
                        "message": message
                    })
                else:
                    # Fallback for plain lines
                    parts = line.strip().split(" | ")
                    if len(parts) >= 3:
                        formatted_logs.append({
                            "timestamp": parts[0].split(" ")[-1],
                            "level": parts[1].strip(),
                            "message": " | ".join(parts[2:])
                        })
                    else:
                        formatted_logs.append({
                            "timestamp": "",
                            "level": "INFO",
                            "message": line.strip()
                        })
            return formatted_logs
        except Exception as e:
            logger.error(f"Error reading logs file: {e}")
            return [{"timestamp": "", "level": "ERROR", "message": f"Failed to read logs: {e}"}]

    def _get_all_trades_from_db(self):
        import sqlite3
        db_path = self.bot.config.get("data", {}).get("database", {}).get("path", "./data/trading.db")
        if not os.path.exists(db_path):
            return []
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy, regime, exit_reason, theoretical_entry_price, theoretical_exit_price FROM trades ORDER BY exit_time ASC")
            rows = cursor.fetchall()
            trades = [dict(row) for row in rows]
            conn.close()
            return trades
        except Exception as e:
            logger.error(f"Error querying trades database: {e}")
            return []

    def _get_analytics(self):
        import math
        import numpy as np
        try:
            from backtest.metrics import calculate_metrics
        except ImportError as e:
            logger.error(f"Failed to import calculate_metrics: {e}")
            return {"error": "Failed to import metrics engine"}
            
        trades = self._get_all_trades_from_db()
        closed_trades = [t for t in trades if t.get("exit_time") is not None]
        
        initial_capital = 10000.0
        if self.bot and hasattr(self.bot, 'initial_capital'):
            initial_capital = self.bot.initial_capital
            
        metrics = calculate_metrics(closed_trades, initial_capital=initial_capital)
        
        max_dd_pct = metrics.get("max_drawdown_pct", 0.0)
        total_pnl = metrics.get("total_pnl", 0.0)
        
        if max_dd_pct > 0:
            recovery_factor = metrics.get("total_return_pct", 0.0) / max_dd_pct
        else:
            recovery_factor = float("inf") if total_pnl > 0 else 0.0
            
        wins = [t for t in closed_trades if t.get("pnl", 0.0) > 0]
        losses = [t for t in closed_trades if t.get("pnl", 0.0) < 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_loss_count_ratio = win_count / loss_count if loss_count > 0 else float("inf")
        
        durations = [t["exit_time"] - t["entry_time"] for t in closed_trades if t.get("exit_time") is not None and t.get("entry_time") is not None]
        avg_duration_sec = (sum(durations) / len(durations) / 1000.0) if durations else 0.0
        
        days = int(avg_duration_sec // 86400)
        hours = int((avg_duration_sec % 86400) // 3600)
        minutes = int((avg_duration_sec % 3600) // 60)
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or not parts:
            parts.append(f"{minutes}m")
        formatted_duration = " ".join(parts) if avg_duration_sec > 0 else "-"
        
        max_win_trade = None
        max_loss_trade = None
        if closed_trades:
            max_win_trade = max(closed_trades, key=lambda x: x.get("pnl", 0.0))
            max_loss_trade = min(closed_trades, key=lambda x: x.get("pnl", 0.0))
            if max_win_trade:
                max_win_trade = dict(max_win_trade)
            if max_loss_trade:
                max_loss_trade = dict(max_loss_trade)

        def clean_val(v):
            if isinstance(v, float):
                if math.isinf(v) or math.isnan(v):
                    return "inf" if v > 0 else ("-inf" if v < 0 else "nan")
            return v

        def get_symbol_from_trade(t):
            strat = t.get("strategy", "")
            if ":" in strat:
                # Extract everything after the first colon, e.g. mtf_macd:BTC/USDT:USDT -> BTC/USDT:USDT
                sym = strat.split(":", 1)[1]
                # Strip exchange suffixes (like :USDT) to get the clean ticker symbol (e.g. BTC/USDT)
                return sym.split(":")[0]
            return "BTC/USDT"

        def compute_trade_slippage(t):
            t_entry = t.get("theoretical_entry_price")
            t_exit = t.get("theoretical_exit_price")
            a_entry = t.get("entry_price")
            a_exit = t.get("exit_price")
            side = t.get("side", "long").lower()
            
            if t_entry is None or t_exit is None or a_entry is None or a_exit is None:
                return None
            if t_entry <= 0 or t_exit <= 0:
                return None
                
            if side == "long":
                entry_slip = ((a_entry - t_entry) / t_entry) * 10000
                exit_slip = ((t_exit - a_exit) / t_exit) * 10000
            else: # short
                entry_slip = ((t_entry - a_entry) / t_entry) * 10000
                exit_slip = ((a_exit - t_exit) / t_exit) * 10000
            return entry_slip, exit_slip

        slippage_by_symbol = {}
        global_entry_slips = []
        global_exit_slips = []
        
        active_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        if self.bot and hasattr(self.bot, 'symbol_states'):
            # Strip exchange suffixes from keys (e.g. BTC/USDT:USDT -> BTC/USDT)
            active_symbols = [s.split(":")[0] for s in self.bot.symbol_states.keys()]
            
        for sym in active_symbols:
            slippage_by_symbol[sym] = {
                "symbol": sym,
                "entry_slip_bps": 0.0,
                "exit_slip_bps": 0.0,
                "total_slip_bps": 0.0,
                "tracked_count": 0,
                "entry_slips_list": [],
                "exit_slips_list": []
            }
            
        for t in closed_trades:
            sym = get_symbol_from_trade(t)
            slip = compute_trade_slippage(t)
            if slip is not None:
                entry_slip, exit_slip = slip
                global_entry_slips.append(entry_slip)
                global_exit_slips.append(exit_slip)
                
                if sym not in slippage_by_symbol:
                    slippage_by_symbol[sym] = {
                        "symbol": sym,
                        "entry_slip_bps": 0.0,
                        "exit_slip_bps": 0.0,
                        "total_slip_bps": 0.0,
                        "tracked_count": 0,
                        "entry_slips_list": [],
                        "exit_slips_list": []
                    }
                slippage_by_symbol[sym]["entry_slips_list"].append(entry_slip)
                slippage_by_symbol[sym]["exit_slips_list"].append(exit_slip)
                slippage_by_symbol[sym]["tracked_count"] += 1
                
        for sym, stats in slippage_by_symbol.items():
            entry_list = stats.pop("entry_slips_list", [])
            exit_list = stats.pop("exit_slips_list", [])
            if entry_list:
                stats["entry_slip_bps"] = clean_val(sum(entry_list) / len(entry_list))
                stats["exit_slip_bps"] = clean_val(sum(exit_list) / len(exit_list))
                stats["total_slip_bps"] = clean_val((sum(entry_list) + sum(exit_list)) / len(entry_list))
            else:
                stats["entry_slip_bps"] = "-"
                stats["exit_slip_bps"] = "-"
                stats["total_slip_bps"] = "-"
            
        global_avg_entry_slip = clean_val(sum(global_entry_slips) / len(global_entry_slips)) if global_entry_slips else "-"
        global_avg_exit_slip = clean_val(sum(global_exit_slips) / len(global_exit_slips)) if global_exit_slips else "-"
        global_avg_total_slip = clean_val((sum(global_entry_slips) + sum(global_exit_slips)) / len(global_entry_slips)) if global_entry_slips else "-"

        return {
            "total_trades": len(closed_trades),
            "win_rate": clean_val(metrics.get("win_rate", 0.0)),
            "total_pnl": clean_val(metrics.get("total_pnl", 0.0)),
            "total_return_pct": clean_val(metrics.get("total_return_pct", 0.0)),
            "annualized_return_pct": clean_val(metrics.get("annualized_return_pct", 0.0)),
            "sharpe_ratio": clean_val(metrics.get("sharpe_ratio", 0.0)),
            "sortino_ratio": clean_val(metrics.get("sortino_ratio", 0.0)),
            "calmar_ratio": clean_val(metrics.get("calmar_ratio", 0.0)),
            "max_drawdown_pct": clean_val(metrics.get("max_drawdown_pct", 0.0)),
            "profit_factor": clean_val(metrics.get("profit_factor", 0.0)),
            "recovery_factor": clean_val(recovery_factor),
            "win_count": win_count,
            "loss_count": loss_count,
            "win_loss_count_ratio": clean_val(win_loss_count_ratio),
            "avg_win": clean_val(metrics.get("avg_win", 0.0)),
            "avg_loss": clean_val(metrics.get("avg_loss", 0.0)),
            "win_loss_ratio": clean_val(metrics.get("win_loss_ratio", 0.0)),
            "avg_duration_seconds": avg_duration_sec,
            "formatted_duration": formatted_duration,
            "max_win_trade": max_win_trade,
            "max_loss_trade": max_loss_trade,
            "slippage_summary": {
                "global_avg_entry_slip": global_avg_entry_slip,
                "global_avg_exit_slip": global_avg_exit_slip,
                "global_avg_total_slip": global_avg_total_slip,
                "global_tracked_count": len(global_entry_slips),
                "by_symbol": list(slippage_by_symbol.values())
            }
        }

    def _export_json(self):
        trades = self._get_all_trades_from_db()
        response_body = json.dumps(trades, indent=2).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Disposition', 'attachment; filename=trade_history.json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response_body)

    def _export_csv(self):
        import csv
        import io
        trades = self._get_all_trades_from_db()
        
        output = io.StringIO()
        if trades:
            headers = list(trades[0].keys())
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            for t in trades:
                writer.writerow(t)
        else:
            headers = ["id", "entry_time", "exit_time", "side", "entry_price", "exit_price", "quantity", "pnl", "pnl_pct", "strategy", "regime", "exit_reason"]
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            
        response_body = output.getvalue().encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv')
        self.send_header('Content-Disposition', 'attachment; filename=trade_history.csv')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response_body)

    def _save_settings(self, settings_data):
        config_path = "config/settings.yaml"
        # 1. Read the current settings.yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)
            
        # 2. Update config sections in-place based on settings_data
        if "risk" in settings_data:
            current_config["risk"] = settings_data["risk"]
        if "strategies" in settings_data:
            current_config["strategies"] = settings_data["strategies"]
        if "meta_labeling" in settings_data:
            current_config["meta_labeling"] = settings_data["meta_labeling"]

        # 3. Write back to config/settings.yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(current_config, f, default_flow_style=False)
            
        # 4. Update the bot's runtime config
        self.bot.config.update(current_config)
        
        # 5. Save credentials in .env if provided
        if "telegram" in settings_data:
            env_path = "config/.env"
            tg_data = settings_data["telegram"]
            lines = []
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            # Update values
            new_lines = []
            keys_updated = {"TELEGRAM_BOT_TOKEN": False, "TELEGRAM_CHAT_ID": False}
            for line in lines:
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    new_lines.append(f"TELEGRAM_BOT_TOKEN={tg_data['bot_token']}\n")
                    keys_updated["TELEGRAM_BOT_TOKEN"] = True
                elif line.startswith("TELEGRAM_CHAT_ID="):
                    new_lines.append(f"TELEGRAM_CHAT_ID={tg_data['chat_id']}\n")
                    keys_updated["TELEGRAM_CHAT_ID"] = True
                else:
                    new_lines.append(line)
            
            for key, updated in keys_updated.items():
                if not updated:
                    new_lines.append(f"{key}={tg_data['bot_token'] if 'TOKEN' in key else tg_data['chat_id']}\n")
                    
            with open(env_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
                
            # Update bot credentials in config as well
            self.bot.config["TELEGRAM_BOT_TOKEN"] = tg_data["bot_token"]
            self.bot.config["TELEGRAM_CHAT_ID"] = tg_data["chat_id"]
            if hasattr(self.bot, 'telegram'):
                self.bot.telegram.token = tg_data["bot_token"]
                self.bot.telegram.chat_id = tg_data["chat_id"]
                self.bot.telegram.enabled = bool(tg_data["bot_token"] and tg_data["chat_id"])

        # 6. Reinitialize the modules in TradingBot for each symbol
        logger.info("Reinitializing trading bot modules with new settings...")
        from risk.position_sizer import KellyPositionSizer
        from risk.circuit_breaker import CircuitBreaker
        from risk.risk_monitor import RiskMonitor
        from strategies.mtf_macd import MTF_MACD_Elder
        from ensemble.regime_classifier import RegimeClassifier
        from ensemble.router import EnsembleRouter
        from execution.position_tracker import PositionTracker
        
        # Global breakers and monitors
        self.bot.circuit_breaker = CircuitBreaker(self.bot.config)
        self.bot.risk_monitor = RiskMonitor(self.bot.config)
        
        # Per-symbol state reinitialization
        for symbol in self.bot.symbols:
            symbol_cfg = self.bot._get_symbol_config(symbol)
            state = self.bot.symbol_states.get(symbol)
            if state:
                # Update strategies
                strategies = {
                    "mtf_macd": MTF_MACD_Elder(symbol_cfg),
                }
                # Update regime classifier and router
                regime_classifier = RegimeClassifier(symbol_cfg)
                ensemble = EnsembleRouter(strategies, regime_classifier)
                
                # Update position tracker preserving active status
                old_tracker = state.get("position_tracker")
                position_tracker = PositionTracker(symbol_cfg)
                if old_tracker:
                    position_tracker.position = old_tracker.position
                    position_tracker.bars_held = old_tracker.bars_held
                    
                position_sizer = KellyPositionSizer(symbol_cfg)
                
                state.update({
                    "config": symbol_cfg,
                    "strategies": strategies,
                    "regime_classifier": regime_classifier,
                    "ensemble": ensemble,
                    "position_tracker": position_tracker,
                    "position_sizer": position_sizer,
                })
                
        # Synchronize first symbol compatibility properties
        if self.bot.symbols:
            first_sym = self.bot.symbols[0]
            first_state = self.bot.symbol_states.get(first_sym, {})
            self.bot.position_tracker = first_state.get("position_tracker")
            self.bot.strategies = first_state.get("strategies")
            self.bot.regime_classifier = first_state.get("regime_classifier")
            self.bot.ensemble = first_state.get("ensemble")
            self.bot.position_sizer = first_state.get("position_sizer")
            
        logger.info("Trading bot settings updated and reloaded in memory.")

class ThreadingHTTPServer(ThreadingTCPServer):
    allow_reuse_address = True
    def __init__(self, server_address, RequestHandlerClass):
        HTTPServer.__init__(self, server_address, RequestHandlerClass)

def run_dashboard_server(bot, host='127.0.0.1', port=8080):
    DashboardHandler.bot = bot
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    logger.info(f"Dashboard server started on http://{host}:{port}")
    
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server
