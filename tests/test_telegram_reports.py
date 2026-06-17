import os
import time
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import datetime

from monitoring.chart_generator import generate_equity_chart
from orchestrator import TradingBot


class TestTelegramReports(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_trading.db")
        
        # Setup dummy database with trades table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time INTEGER,
                exit_time INTEGER,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                pnl_pct REAL,
                strategy TEXT,
                exit_reason TEXT,
                theoretical_entry_price REAL,
                theoretical_exit_price REAL
            )
        """)
        conn.commit()
        conn.close()

        # Mock Bot Config
        self.config = {
            "bot": {
                "name": "bocik_test",
                "version": "1.0.0",
                "mode": "paper"
            },
            "exchange": {
                "name": "bitget",
                "symbols": ["BTC/USDT:USDT"]
            },
            "data": {
                "database": {
                    "path": self.db_path
                }
            },
            "paths": {
                "logs_dir": self.temp_dir.name
            },
            "monitoring": {
                "telegram": {
                    "enabled": True,
                    "reports": {
                        "enabled": True,
                        "time": "22:00",
                        "interval": "both",
                        "weekly_day": "Sunday"
                    }
                }
            },
            "risk": {
                "initial_capital": 10000.0
            }
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_chart_generation_pillow_and_matplotlib(self):
        """Test that generate_equity_chart successfully creates a PNG file."""
        output_file = os.path.join(self.temp_dir.name, "test_chart.png")
        equity_series = [10000.0, 10100.0, 9950.0, 10300.0]
        
        res = generate_equity_chart(equity_series, output_file)
        self.assertTrue(res)
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)

    def test_chart_generation_single_point(self):
        """Test that single value equity series handles drawing properly."""
        output_file = os.path.join(self.temp_dir.name, "test_chart_single.png")
        equity_series = [10000.0, 10000.0]
        
        res = generate_equity_chart(equity_series, output_file)
        self.assertTrue(res)
        self.assertTrue(os.path.exists(output_file))

    @patch("monitoring.telegram_bot.requests.post")
    def test_send_periodic_report_daily(self, mock_post):
        """Test daily report calculation and Telegram sendPhoto call."""
        # Setup mock response for Telegram API
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        # Insert test trades (one inside 24h, one outside)
        now_ms = int(time.time() * 1000)
        yesterday_ms = now_ms - (25 * 60 * 60 * 1000)
        recent_ms = now_ms - (2 * 60 * 60 * 1000)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Trade 1: 25h ago (outside daily, inside weekly)
        cursor.execute("""
            INSERT INTO trades (entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy)
            VALUES (?, ?, 'long', 50000.0, 50500.0, 0.1, 50.0, 1.0, 'mtf_macd:BTC/USDT:USDT')
        """, (yesterday_ms - 3600000, yesterday_ms))
        
        # Trade 2: 2h ago (inside daily and weekly)
        cursor.execute("""
            INSERT INTO trades (entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy)
            VALUES (?, ?, 'short', 51000.0, 50000.0, 0.1, 100.0, 1.96, 'mtf_macd:ETH/USDT:USDT')
        """, (recent_ms - 3600000, recent_ms))
        conn.commit()
        conn.close()

        # Instantiate bot with mock config
        with patch("orchestrator.setup_logger"):
            bot = TradingBot()
            bot.config = self.config
            bot.equity = 10150.0
            bot.balance = 10150.0
            bot.initial_capital = 10000.0
            bot.telegram = MagicMock()

            # Execute periodic report
            bot._send_periodic_report("daily")

            # Check that send_photo was called on the telegram alerter
            bot.telegram.send_photo.assert_called_once()
            
            # Extract arguments passed to send_photo
            photo_bytes = bot.telegram.send_photo.call_args[0][0]
            caption = bot.telegram.send_photo.call_args[1]["caption"]

            # Verify that the correct trade was included (Trade 2 PnL was $100.0)
            self.assertIn("RAPORT DOBOWY", caption)
            self.assertIn("ETH/USDT", caption)
            self.assertIn("+$100.00", caption)
            # Trade 1 should NOT be in daily report
            self.assertNotIn("BTC/USDT", caption)
            self.assertGreater(len(photo_bytes), 0)

    @patch("monitoring.telegram_bot.requests.post")
    def test_send_periodic_report_weekly(self, mock_post):
        """Test weekly report calculation and Telegram sendPhoto call."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        now_ms = int(time.time() * 1000)
        two_days_ago_ms = now_ms - (2 * 24 * 60 * 60 * 1000)
        eight_days_ago_ms = now_ms - (8 * 24 * 60 * 60 * 1000)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Trade 1: 2 days ago (inside weekly)
        cursor.execute("""
            INSERT INTO trades (entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy)
            VALUES (?, ?, 'long', 50000.0, 50500.0, 0.1, 50.0, 1.0, 'mtf_macd:BTC/USDT:USDT')
        """, (two_days_ago_ms - 3600000, two_days_ago_ms))
        
        # Trade 2: 8 days ago (outside weekly)
        cursor.execute("""
            INSERT INTO trades (entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy)
            VALUES (?, ?, 'short', 51000.0, 52000.0, 0.1, -100.0, -1.96, 'mtf_macd:SOL/USDT:USDT')
        """, (eight_days_ago_ms - 3600000, eight_days_ago_ms))
        conn.commit()
        conn.close()

        with patch("orchestrator.setup_logger"):
            bot = TradingBot()
            bot.config = self.config
            bot.equity = 9950.0
            bot.balance = 9950.0
            bot.initial_capital = 10000.0
            bot.telegram = MagicMock()

            bot._send_periodic_report("weekly")

            bot.telegram.send_photo.assert_called_once()
            caption = bot.telegram.send_photo.call_args[1]["caption"]

            self.assertIn("RAPORT TYGODNIOWY", caption)
            self.assertIn("BTC/USDT", caption)
            self.assertIn("+$50.00", caption)
            # SOL trade should be excluded
            self.assertNotIn("SOL/USDT", caption)


if __name__ == "__main__":
    unittest.main()
