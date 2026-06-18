"""SQLAlchemy ORM models for candles, trades, signals, and performance snapshots."""

from sqlalchemy import Column, BigInteger, Integer, Float, Text, Boolean, create_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Candle(Base):
    """Raw 1-minute OHLCV candle — one per symbol per timestamp."""
    __tablename__ = "candles_1m"

    symbol = Column(Text, primary_key=True, default="BTC/USDT:USDT")
    timestamp = Column(BigInteger, primary_key=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class Trade(Base):
    """Completed trade record."""
    __tablename__ = "trades"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    entry_time = Column(BigInteger, nullable=False)
    exit_time = Column(BigInteger, nullable=True)
    side = Column(Text, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    strategy = Column(Text, nullable=False)
    regime = Column(Text, nullable=True)
    exit_reason = Column(Text, nullable=True)
    features_json = Column(Text, nullable=True)
    theoretical_entry_price = Column(Float, nullable=True)
    theoretical_exit_price = Column(Float, nullable=True)
    pipeline = Column(Text, nullable=True, default="pure")


class Signal(Base):
    """Logged signals for debugging and analysis."""
    __tablename__ = "signals"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False)
    strategy = Column(Text, nullable=False)
    signal = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    regime = Column(Text, nullable=True)
    executed = Column(Boolean, default=False)
    reject_reason = Column(Text, nullable=True)
    features_json = Column(Text, nullable=True)
    pipeline = Column(Text, nullable=True, default="pure")


class PerformanceSnapshot(Base):
    """Hourly performance snapshots."""
    __tablename__ = "performance_snapshots"

    timestamp = Column(BigInteger, primary_key=True)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    position_size = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    drawdown_pct = Column(Float, nullable=True)
    sharpe_rolling = Column(Float, nullable=True)
