"""Repository classes for database CRUD operations.

Thread-safe — uses SQLAlchemy sessions properly.
Supports both SQLite (dev) and PostgreSQL (production).
"""

from typing import Optional, List
from contextlib import contextmanager
import threading

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from .models import Base, Candle, Trade, Signal, PerformanceSnapshot


class DatabaseManager:
    """Manages database connections and schema.
    
    Singleton pattern — one instance per bot process.
    Thread-safe via session-per-thread pattern.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: dict = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._initialized = True
        db_cfg = config.get("data", {}).get("database", {}) if config else {}
        
        db_type = db_cfg.get("type", "sqlite")
        
        if db_type == "sqlite":
            db_path = db_cfg.get("path", "./data/trading.db")
            self.engine = create_engine(
                f"sqlite:///{db_path}",
                echo=False,
                connect_args={"check_same_thread": False},  # SQLite needs this for multi-thread
                pool_pre_ping=True,
            )
            # Enable WAL mode for better concurrency
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
                conn.commit()
        
        elif db_type == "postgresql":
            host = db_cfg.get("host", "localhost")
            port = db_cfg.get("port", 5432)
            dbname = db_cfg.get("name", "tradingbot")
            user = db_cfg.get("user", "bot")
            password = db_cfg.get("password", "")
            self.engine = create_engine(
                f"postgresql://{user}:{password}@{host}:{port}/{dbname}",
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        # Pre-creation check for SQLite: detect if trades or signals table has id of type BIGINT
        # and needs type migration to INTEGER for proper autoincrement.
        if db_type == "sqlite":
            try:
                with self.engine.connect() as conn:
                    # Check trades table
                    cursor = conn.exec_driver_sql("PRAGMA table_info(trades)")
                    trades_info = cursor.fetchall()
                    if trades_info:
                        id_col = [row for row in trades_info if row[1] == "id"]
                        if id_col and id_col[0][2].upper() == "BIGINT":
                            logger.warning("SQLite: trades table 'id' is BIGINT. Renaming to 'trades_old' for type migration.")
                            conn.exec_driver_sql("ALTER TABLE trades RENAME TO trades_old")
                    
                    # Check signals table
                    cursor = conn.exec_driver_sql("PRAGMA table_info(signals)")
                    signals_info = cursor.fetchall()
                    if signals_info:
                        id_col = [row for row in signals_info if row[1] == "id"]
                        if id_col and id_col[0][2].upper() == "BIGINT":
                            logger.warning("SQLite: signals table 'id' is BIGINT. Renaming to 'signals_old' for type migration.")
                            conn.exec_driver_sql("ALTER TABLE signals RENAME TO signals_old")
                    conn.commit()
            except Exception as e:
                logger.error(f"SQLite pre-migration table renaming failed: {e}")

        # Create all tables
        Base.metadata.create_all(self.engine)
        
        # Post-creation data copying for SQLite type migrations
        if db_type == "sqlite":
            try:
                with self.engine.connect() as conn:
                    # Migrate trades data
                    cursor = conn.exec_driver_sql("PRAGMA table_info(trades_old)")
                    trades_old_info = cursor.fetchall()
                    if trades_old_info:
                        logger.info("SQLite: migrating data from trades_old to trades...")
                        cursor_new = conn.exec_driver_sql("PRAGMA table_info(trades)")
                        new_cols = {row[1] for row in cursor_new.fetchall()}
                        old_cols = [row[1] for row in trades_old_info]
                        common_cols = [c for c in old_cols if c in new_cols]
                        cols_str = ", ".join(common_cols)
                        conn.exec_driver_sql(f"INSERT INTO trades ({cols_str}) SELECT {cols_str} FROM trades_old")
                        conn.exec_driver_sql("DROP TABLE trades_old")
                        logger.info("SQLite: trades table type migration complete.")
                        
                    # Migrate signals data
                    cursor = conn.exec_driver_sql("PRAGMA table_info(signals_old)")
                    signals_old_info = cursor.fetchall()
                    if signals_old_info:
                        logger.info("SQLite: migrating data from signals_old to signals...")
                        cursor_new = conn.exec_driver_sql("PRAGMA table_info(signals)")
                        new_cols = {row[1] for row in cursor_new.fetchall()}
                        old_cols = [row[1] for row in signals_old_info]
                        common_cols = [c for c in old_cols if c in new_cols]
                        cols_str = ", ".join(common_cols)
                        conn.exec_driver_sql(f"INSERT INTO signals ({cols_str}) SELECT {cols_str} FROM signals_old")
                        conn.exec_driver_sql("DROP TABLE signals_old")
                        logger.info("SQLite: signals table type migration complete.")
                    conn.commit()
            except Exception as e:
                logger.error(f"SQLite data migration from old tables failed: {e}")

        # Run automatic migrations to add new columns to existing tables
        try:
            with self.engine.connect() as conn:
                if db_type == "sqlite":
                    # --- candles_1m: add 'symbol' column if missing ---
                    cursor = conn.exec_driver_sql("PRAGMA table_info(candles_1m)")
                    candle_columns = [row[1] for row in cursor.fetchall()]
                    # Check for orphaned old table from a previous interrupted migration
                    cursor_old = conn.exec_driver_sql(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='candles_1m_old'"
                    )
                    has_orphaned = cursor_old.fetchone() is not None

                    if has_orphaned:
                        logger.warning("SQLite: found orphaned candles_1m_old table from previous migration — recovering...")
                        if not candle_columns:
                            # Table was renamed but new table wasn't created — recover
                            Base.metadata.create_all(self.engine, tables=[Base.metadata.tables["candles_1m"]])
                            conn.exec_driver_sql(
                                "INSERT INTO candles_1m (symbol, timestamp, open, high, low, close, volume) "
                                "SELECT 'BTC/USDT:USDT', timestamp, open, high, low, close, volume FROM candles_1m_old"
                            )
                        # Drop the old table regardless
                        conn.exec_driver_sql("DROP TABLE candles_1m_old")
                        logger.info("SQLite: orphaned candles_1m_old table dropped.")
                        conn.commit()

                    elif candle_columns and "symbol" not in candle_columns:
                        logger.info("SQLite: migrating candles_1m — adding 'symbol' column...")
                        conn.exec_driver_sql("ALTER TABLE candles_1m RENAME TO candles_1m_old")
                        conn.commit()
                        # Create new table with composite PK via SQLAlchemy
                        Base.metadata.create_all(self.engine, tables=[Base.metadata.tables["candles_1m"]])
                        # Copy old data, setting symbol to a default
                        conn.exec_driver_sql(
                            "INSERT INTO candles_1m (symbol, timestamp, open, high, low, close, volume) "
                            "SELECT 'BTC/USDT:USDT', timestamp, open, high, low, close, volume FROM candles_1m_old"
                        )
                        conn.exec_driver_sql("DROP TABLE candles_1m_old")
                        logger.info("SQLite: candles_1m migration complete (symbol column added).")
                        conn.commit()

                    # --- trades: add theoretical columns if missing ---
                    cursor = conn.exec_driver_sql("PRAGMA table_info(trades)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if "theoretical_entry_price" not in columns:
                        conn.exec_driver_sql("ALTER TABLE trades ADD COLUMN theoretical_entry_price FLOAT")
                        logger.info("Migrated SQLite: added 'theoretical_entry_price' column to 'trades' table")
                    if "theoretical_exit_price" not in columns:
                        conn.exec_driver_sql("ALTER TABLE trades ADD COLUMN theoretical_exit_price FLOAT")
                        logger.info("Migrated SQLite: added 'theoretical_exit_price' column to 'trades' table")
                elif db_type == "postgresql":
                    cursor = conn.execute(text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name='trades' AND column_name='theoretical_entry_price'"
                    ))
                    if not cursor.fetchone():
                        conn.execute(text("ALTER TABLE trades ADD COLUMN theoretical_entry_price DOUBLE PRECISION"))
                        logger.info("Migrated PostgreSQL: added 'theoretical_entry_price' column to 'trades' table")
                    cursor = conn.execute(text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name='trades' AND column_name='theoretical_exit_price'"
                    ))
                    if not cursor.fetchone():
                        conn.execute(text("ALTER TABLE trades ADD COLUMN theoretical_exit_price DOUBLE PRECISION"))
                        logger.info("Migrated PostgreSQL: added 'theoretical_exit_price' column to 'trades' table")
                conn.commit()
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
        
        # Session factory (create a new session per thread)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
        logger.info(f"Database initialized: {db_type}")
    
    @contextmanager
    def session(self) -> Session:
        """Get a database session. Use as context manager.
        
        Usage:
            with db.session() as session:
                session.add(candle)
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# ─── Candle Repository ────────────────────────────────────────


class CandleRepository:
    """CRUD operations for OHLCV candles.
    
    Stores 1m candles. Higher-timeframe candles are derived via resampling.
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def insert(self, candle: dict, symbol: str = "BTC/USDT:USDT") -> bool:
        """Insert a single candle. Returns True on success, False if duplicate."""
        try:
            sym = candle.get("symbol", symbol)
            ts = candle["timestamp"]
            with self.db.session() as session:
                existing = session.get(Candle, (sym, ts))
                if existing:
                    return False  # Duplicate — skip silently

                record = Candle(
                    symbol=sym,
                    timestamp=ts,
                    open=candle["open"],
                    high=candle["high"],
                    low=candle["low"],
                    close=candle["close"],
                    volume=candle["volume"],
                )
                session.add(record)
                return True
        except Exception as e:
            logger.error(f"Failed to insert candle ts={candle['timestamp']}: {e}")
            return False

    def insert_batch(self, candles: List[dict], symbol: str = "BTC/USDT:USDT") -> int:
        """Insert multiple candles efficiently. Returns count of inserted rows.

        Uses bulk insert for performance. Skips duplicates.
        """
        if not candles:
            return 0

        try:
            sym = candles[0].get("symbol", symbol) if candles else symbol
            records = [
                Candle(
                    symbol=c.get("symbol", sym),
                    timestamp=c["timestamp"],
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"],
                )
                for c in candles
            ]

            with self.db.session() as session:
                existing_keys = set(
                    session.query(Candle.symbol, Candle.timestamp)
                    .filter(Candle.symbol == sym)
                    .all()
                )
                new_records = [r for r in records if (r.symbol, r.timestamp) not in existing_keys]

                if new_records:
                    session.bulk_save_objects(new_records)

                return len(new_records)
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return 0
    
    def get_latest(self, symbol: Optional[str] = None) -> Optional[dict]:
        """Get the most recent candle, optionally filtered by symbol."""
        try:
            with self.db.session() as session:
                q = session.query(Candle)
                if symbol:
                    q = q.filter(Candle.symbol == symbol)
                candle = q.order_by(Candle.timestamp.desc()).first()
                if candle:
                    return self._to_dict(candle)
                return None
        except Exception as e:
            logger.error(f"Failed to get latest candle: {e}")
            return None

    def get_range(
        self,
        start_ts: int,
        end_ts: int,
        limit: int = 10000,
        symbol: Optional[str] = None,
    ) -> List[dict]:
        """Get candles in a timestamp range, sorted ascending. Optionally filter by symbol."""
        try:
            with self.db.session() as session:
                q = (
                    session.query(Candle)
                    .filter(Candle.timestamp >= start_ts)
                    .filter(Candle.timestamp <= end_ts)
                )
                if symbol:
                    q = q.filter(Candle.symbol == symbol)
                candles = q.order_by(Candle.timestamp.asc()).limit(limit).all()
                return [self._to_dict(c) for c in candles]
        except Exception as e:
            logger.error(f"Failed to get candle range: {e}")
            return []
    
    def get_count(self) -> int:
        """Get total number of stored candles."""
        try:
            with self.db.session() as session:
                return session.query(Candle).count()
        except Exception:
            return 0
    
    def get_all_timestamps(self) -> set[int]:
        """Get all stored timestamps (for gap detection)."""
        try:
            with self.db.session() as session:
                results = session.query(Candle.timestamp).all()
                return {r[0] for r in results}
        except Exception:
            return set()
    
    def get_oldest_timestamp(self) -> Optional[int]:
        """Get the oldest stored timestamp."""
        try:
            with self.db.session() as session:
                result = (
                    session.query(Candle.timestamp)
                    .order_by(Candle.timestamp.asc())
                    .first()
                )
                return result[0] if result else None
        except Exception:
            return None
    
    def delete_older_than(self, timestamp_ms: int) -> int:
        """Delete candles older than a timestamp. Returns count deleted."""
        try:
            with self.db.session() as session:
                count = (
                    session.query(Candle)
                    .filter(Candle.timestamp < timestamp_ms)
                    .delete()
                )
                return count
        except Exception as e:
            logger.error(f"Failed to delete old candles: {e}")
            return 0
    
    @staticmethod
    def _to_dict(candle: Candle) -> dict:
        return {
            "symbol": candle.symbol,
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }


# ─── Trade Repository ─────────────────────────────────────────


class TradeRepository:
    """CRUD operations for completed trades."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def insert(self, trade: dict) -> int:
        """Insert a completed trade. Returns the trade ID."""
        try:
            with self.db.session() as session:
                record = Trade(
                    entry_time=trade["entry_time"],
                    exit_time=trade.get("exit_time"),
                    side=trade["side"],
                    entry_price=trade["entry_price"],
                    exit_price=trade.get("exit_price"),
                    quantity=trade["quantity"],
                    pnl=trade.get("pnl"),
                    pnl_pct=trade.get("pnl_pct"),
                    strategy=trade["strategy"],
                    regime=trade.get("regime"),
                    exit_reason=trade.get("exit_reason"),
                    features_json=trade.get("features_json"),
                    theoretical_entry_price=trade.get("theoretical_entry_price"),
                    theoretical_exit_price=trade.get("theoretical_exit_price"),
                )
                session.add(record)
                session.flush()
                return record.id
        except Exception as e:
            logger.error(f"Failed to insert trade: {e}")
            return -1
    
    def get_recent(self, limit: int = 20) -> list:
        """Get the most recent trades."""
        try:
            with self.db.session() as session:
                trades = (
                    session.query(Trade)
                    .order_by(Trade.entry_time.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": t.id,
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                        "side": t.side,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct,
                        "strategy": t.strategy,
                        "exit_reason": t.exit_reason,
                        "theoretical_entry_price": t.theoretical_entry_price,
                        "theoretical_exit_price": t.theoretical_exit_price,
                    }
                    for t in trades
                ]
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
    
    def get_total_pnl(self) -> float:
        """Get cumulative PnL from all trades."""
        try:
            with self.db.session() as session:
                result = session.query(Trade.pnl).all()
                return sum(r[0] for r in result if r[0] is not None)
        except Exception:
            return 0.0


# ─── Signal Repository ────────────────────────────────────────


class SignalRepository:
    """CRUD for logged signals (debugging/analysis)."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def insert(self, signal: dict):
        """Log a trading signal."""
        try:
            with self.db.session() as session:
                record = Signal(
                    timestamp=signal["timestamp"],
                    strategy=signal["strategy"],
                    signal=signal["signal"],
                    confidence=signal.get("confidence"),
                    regime=signal.get("regime"),
                    executed=signal.get("executed", False),
                    reject_reason=signal.get("reject_reason"),
                    features_json=signal.get("features_json"),
                )
                session.add(record)
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")
