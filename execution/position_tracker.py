"""Track open positions with SL/TP/trailing stops matching the backtest engine exit chain."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: float  # Unix timestamp
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_distance: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    unrealized_pnl: float = 0.0


class PositionTracker:
    """Monitors open position and manages stop-loss/take-profit logic."""
    
    def __init__(self, config: dict):
        self.config = config
        trade_cfg = config.get("risk", {}).get("per_trade", {})
        self.max_duration_h = trade_cfg.get("max_duration_hours", 48)
        self.position: Optional[Position] = None
        self.bars_held = 0
    
    def enter(self, side: str, entry_price: float, quantity: float, atr: float = 0.0, atr_pct: float = 2.0, timestamp: float = 0.0, symbol: str = "BTC/USDT") -> Position:
        """Open a new position with volatility-adjusted SL/TP levels."""
        # Check config for overrides
        exit_cfg = self.config.get("strategies", {}).get("mtf_macd_elder", {}).get("exit", {})
        
        # Dynamic multipliers exactly matching backtest/engine.py:
        vol_factor = max(0.5, min(2.0, atr_pct / 2.0))  # Normalize around 2% ATR
        tp_mult = 2.0 + (vol_factor * 0.5)              # 2.25-3.0x risk
        sl_mult = exit_cfg.get("atr_stop_mult", 2.0 + vol_factor)
        trail_pct = exit_cfg.get("trailing_stop_pct", 0.025 + (vol_factor * 0.01))
        
        risk = sl_mult * atr if atr > 0 else entry_price * 0.02
        
        if side == "long":
            sl = entry_price - risk
            tp = entry_price + (tp_mult * risk)
        else:
            sl = entry_price + risk
            tp = entry_price - (tp_mult * risk)
        
        self.position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=timestamp,
            stop_loss=sl,
            take_profit=tp,
            trailing_distance=trail_pct,
            highest_price=entry_price if side == "long" else 0.0,
            lowest_price=entry_price if side == "short" else float("inf"),
            unrealized_pnl=0.0
        )
        self.bars_held = 0
        return self.position
    
    def update(self, candle: dict) -> str:
        """Update position with completed candle. Returns exit reason ('take_profit', 'trailing_stop', 'atr_stop', 'time_exit') or 'hold'."""
        if not self.position:
            return "no_position"
        
        p = self.position
        close = candle["close"]
        high = candle["high"]
        low = candle["low"]
        
        # Calculate PnL
        if p.side == "long":
            p.unrealized_pnl = (close - p.entry_price) * p.quantity
        else:
            p.unrealized_pnl = (p.entry_price - close) * p.quantity
            
        self.bars_held += 1
        
        if p.side == "long":
            # 1. Take Profit check
            if high >= p.take_profit:
                return "take_profit"
            # 2. Trailing stop check
            trail_price = p.highest_price * (1 - p.trailing_distance)
            if low <= trail_price:
                return "trailing_stop"
            # 3. Stop Loss (ATR stop) check
            if low <= p.stop_loss:
                return "atr_stop"
            # 4. Timeout check
            if self.bars_held >= self.max_duration_h:
                return "time_exit"
                
            p.highest_price = max(p.highest_price, high)
            
        elif p.side == "short":
            # 1. Take Profit check
            if low <= p.take_profit:
                return "take_profit"
            # 2. Trailing stop check
            trail_price = p.lowest_price * (1 + p.trailing_distance)
            if high >= trail_price:
                return "trailing_stop"
            # 3. Stop Loss (ATR stop) check
            if high >= p.stop_loss:
                return "atr_stop"
            # 4. Timeout check
            if self.bars_held >= self.max_duration_h:
                return "time_exit"
                
            p.lowest_price = min(p.lowest_price, low)
            
        return "hold"
    
    def exit(self):
        """Close the position."""
        self.position = None
        self.bars_held = 0
