"""Tests for Slippage and Adverse Selection tracking.

Covers:
- Database migrations adding theoretical_entry_price and theoretical_exit_price.
- TradeRepository insert and get_recent with theoretical prices.
- Backend API analytics calculation of average slippage in bps (basis points) per symbol and globally.
"""

import sys
import os
import tempfile
import shutil
import sqlite3
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.storage.repositories import DatabaseManager, TradeRepository
from data.storage.models import Base, Trade
from dashboard.server import DashboardHandler


class TestSlippageTracking:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        DatabaseManager._instance = None
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_slippage.db")
        self.config = {
            "data": {
                "database": {
                    "type": "sqlite",
                    "path": self.db_path,
                }
            }
        }
        yield
        DatabaseManager._instance = None
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_migration_adds_columns(self):
        """Test that if trades table exists without theoretical price columns, they are automatically added."""
        # 1. Manually create db and trades table without the columns
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                strategy TEXT NOT NULL,
                regime TEXT,
                exit_reason TEXT,
                features_json TEXT
            )
        """)
        conn.commit()
        conn.close()

        # 2. Initialize DatabaseManager (this should run migrations)
        db_mgr = DatabaseManager(self.config)

        # 3. Check if columns exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(trades)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        assert "theoretical_entry_price" in columns
        assert "theoretical_exit_price" in columns

    def test_trade_repository_persists_and_retrieves_theoretical_prices(self):
        """Test that TradeRepository correctly inserts and retrieves theoretical entry/exit prices."""
        db_mgr = DatabaseManager(self.config)
        repo = TradeRepository(db_mgr)
        
        print("Trade table columns in SQLAlchemy:", [c.name for c in Trade.__table__.columns])

        trade_data = {
            "entry_time": 1700000000000,
            "exit_time": 1700003600000,
            "side": "long",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "quantity": 0.5,
            "pnl": 500.0,
            "pnl_pct": 1.0,
            "strategy": "mtf_macd:BTC/USDT",
            "exit_reason": "take_profit",
            "theoretical_entry_price": 49950.0,
            "theoretical_exit_price": 51050.0
        }

        trade_id = repo.insert(trade_data)
        assert trade_id > 0

        recent_trades = repo.get_recent(limit=5)
        assert len(recent_trades) == 1
        retrieved = recent_trades[0]
        assert retrieved["entry_price"] == 50000.0
        assert retrieved["exit_price"] == 51000.0
        assert retrieved["theoretical_entry_price"] == 49950.0
        assert retrieved["theoretical_exit_price"] == 51050.0

    def test_slippage_analytics_calculations(self):
        """Test slippage bps calculations (global and per-symbol) in API handler."""
        db_mgr = DatabaseManager(self.config)
        repo = TradeRepository(db_mgr)

        # Insert some trades with known slippages:
        # Trade 1: BTC/USDT Long
        # Actual entry = 50050.0, Theoretical entry = 50000.0 -> entry slip = (50050 - 50000)/50000 * 10000 = +10.0 bps
        # Actual exit = 50900.0, Theoretical exit = 51000.0 -> exit slip = (51000 - 50900)/51000 * 10000 = +19.6078 bps
        repo.insert({
            "entry_time": 1700000000000,
            "exit_time": 1700003600000,
            "side": "long",
            "entry_price": 50050.0,
            "exit_price": 50900.0,
            "quantity": 1.0,
            "pnl": 850.0,
            "pnl_pct": 1.7,
            "strategy": "mtf_macd:BTC/USDT",
            "exit_reason": "signal",
            "theoretical_entry_price": 50000.0,
            "theoretical_exit_price": 51000.0
        })

        # Trade 2: ETH/USDT Short
        # Actual entry = 3000.0, Theoretical entry = 3010.0 -> entry slip = (3010 - 3000)/3010 * 10000 = +33.2225 bps
        # Actual exit = 2960.0, Theoretical exit = 2950.0 -> exit slip = (2960 - 2950)/2950 * 10000 = +33.8983 bps
        repo.insert({
            "entry_time": 1700000000000,
            "exit_time": 1700003600000,
            "side": "short",
            "entry_price": 3000.0,
            "exit_price": 2960.0,
            "quantity": 10.0,
            "pnl": 400.0,
            "pnl_pct": 1.3,
            "strategy": "mtf_macd:ETH/USDT",
            "exit_reason": "trailing_stop",
            "theoretical_entry_price": 3010.0,
            "theoretical_exit_price": 2950.0
        })

        # Trade 3: Old trade without theoretical prices (should be ignored in slippage averages)
        repo.insert({
            "entry_time": 1700000000000,
            "exit_time": 1700003600000,
            "side": "long",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "quantity": 0.5,
            "pnl": 500.0,
            "pnl_pct": 1.0,
            "strategy": "mtf_macd:BTC/USDT",
            "exit_reason": "take_profit"
        })

        # Create mock bot
        mock_bot = MagicMock()
        mock_bot.config = self.config
        mock_bot.initial_capital = 10000.0
        # Mock active symbols
        mock_bot.symbol_states = {
            "BTC/USDT": {},
            "ETH/USDT": {}
        }

        # Mock DashboardHandler and call _get_analytics
        handler = MagicMock(spec=DashboardHandler)
        handler.bot = mock_bot
        
        # Bind methods we want to test to the mock handler
        handler._get_analytics = DashboardHandler._get_analytics.__get__(handler, DashboardHandler)
        handler._get_all_trades_from_db = DashboardHandler._get_all_trades_from_db.__get__(handler, DashboardHandler)
        
        all_trades = handler._get_all_trades_from_db()
        print(f"DIAGNOSTIC: db_path={self.db_path}, exists={os.path.exists(self.db_path)}")
        print(f"DIAGNOSTIC: bot.config path={mock_bot.config.get('data', {}).get('database', {}).get('path')}")
        print(f"DIAGNOSTIC: all_trades fetched = {all_trades}")

        analytics = handler._get_analytics()
        print(f"DIAGNOSTIC: analytics result = {analytics}")
        assert "slippage_summary" in analytics
        
        summary = analytics["slippage_summary"]
        assert summary["global_tracked_count"] == 2
        
        # Verify global averages
        # Entry average: (10.0 + 33.2225) / 2 = 21.61125
        # Exit average: (19.6078 + 33.8983) / 2 = 26.75305
        # Total average: (10.0 + 19.6078 + 33.2225 + 33.8983) / 2 = 48.3643
        assert pytest.approx(summary["global_avg_entry_slip"], 0.01) == 21.61
        assert pytest.approx(summary["global_avg_exit_slip"], 0.01) == 26.75
        assert pytest.approx(summary["global_avg_total_slip"], 0.01) == 48.36
        
        # Verify symbol breakdown
        by_symbol = {item["symbol"]: item for item in summary["by_symbol"]}
        assert "BTC/USDT" in by_symbol
        assert "ETH/USDT" in by_symbol
        
        btc_stats = by_symbol["BTC/USDT"]
        assert btc_stats["tracked_count"] == 1
        assert pytest.approx(btc_stats["entry_slip_bps"], 0.01) == 10.0
        assert pytest.approx(btc_stats["exit_slip_bps"], 0.01) == 19.61
        assert pytest.approx(btc_stats["total_slip_bps"], 0.01) == 29.61
        
        eth_stats = by_symbol["ETH/USDT"]
        assert eth_stats["tracked_count"] == 1
        assert pytest.approx(eth_stats["entry_slip_bps"], 0.01) == 33.22
        assert pytest.approx(eth_stats["exit_slip_bps"], 0.01) == 33.90
        assert pytest.approx(eth_stats["total_slip_bps"], 0.01) == 67.12

    def test_database_migration_fixes_bigint_primary_key(self):
        """Test that if trades or signals table has id BIGINT (old SQLite schema), it is migrated to INTEGER and data is kept."""
        # 1. Create a SQLite database with trades and signals having BIGINT ids
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE trades (
                id BIGINT NOT NULL,
                entry_time BIGINT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                strategy TEXT NOT NULL,
                PRIMARY KEY (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE signals (
                id BIGINT NOT NULL,
                timestamp BIGINT NOT NULL,
                strategy TEXT NOT NULL,
                signal TEXT NOT NULL,
                PRIMARY KEY (id)
            )
        """)
        
        # Insert test data into both tables
        cursor.execute("INSERT INTO trades (id, entry_time, side, entry_price, quantity, strategy) VALUES (10, 1700000000000, 'long', 50000.0, 1.0, 'test_strategy')")
        cursor.execute("INSERT INTO signals (id, timestamp, strategy, signal) VALUES (20, 1700000000000, 'test_strategy', 'buy')")
        
        conn.commit()
        conn.close()
        
        # 2. Initialize DatabaseManager (should automatically run type migration)
        db_mgr = DatabaseManager(self.config)
        
        # 3. Verify columns and types of trades
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(trades)")
        trades_cols = cursor.fetchall()
        id_col_trades = [row for row in trades_cols if row[1] == 'id'][0]
        # Should be INTEGER
        assert id_col_trades[2].upper() in ('INTEGER', 'INT')
        
        cursor.execute("PRAGMA table_info(signals)")
        signals_cols = cursor.fetchall()
        id_col_signals = [row for row in signals_cols if row[1] == 'id'][0]
        assert id_col_signals[2].upper() in ('INTEGER', 'INT')
        
        # Verify data was preserved
        cursor.execute("SELECT id, entry_time, side, entry_price FROM trades")
        trade_rows = cursor.fetchall()
        assert len(trade_rows) == 1
        assert trade_rows[0] == (10, 1700000000000, 'long', 50000.0)
        
        cursor.execute("SELECT id, timestamp, strategy, signal FROM signals")
        signal_rows = cursor.fetchall()
        assert len(signal_rows) == 1
        assert signal_rows[0] == (20, 1700000000000, 'test_strategy', 'buy')
        
        conn.close()
        
        # 4. Verify that we can insert new records using the TradeRepository
        repo = TradeRepository(db_mgr)
        new_trade_id = repo.insert({
            "entry_time": 1700001000000,
            "side": "short",
            "entry_price": 51000.0,
            "quantity": 0.5,
            "strategy": "test_strategy"
        })
        assert new_trade_id > 0
