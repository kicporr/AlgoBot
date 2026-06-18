"""Position sizing using Kelly Criterion with volatility adjustment.

f* = (p * b - q) / b
where: p = win rate, q = loss rate, b = avg_win / avg_loss

Half-Kelly (f*/2) reduces volatility while maintaining ~75% of full-Kelly returns.
Volatility adjustment scales position inversely with ATR: higher volatility → smaller size.
"""

import numpy as np
from loguru import logger


class KellyPositionSizer:
    """Calculates position size using fractional Kelly Criterion.

    Sizing formula:
        half_kelly_pct = max(0, (p * b - q) / (2 * b))
        vol_adj = avg_atr / current_atr  (≤ 1.0 — only shrinks)
        risk_capital = capital * half_kelly_pct * vol_adj
        position_btc = risk_capital / btc_price
    """

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        sizing_cfg = risk_cfg.get("position_sizing", {})

        # Position sizing method
        self.method = sizing_cfg.get("method", "half_kelly")
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.20)

        # Kelly parameters (from backtest or config)
        kelly_cfg = sizing_cfg.get("kelly", {})
        self.default_win_rate = kelly_cfg.get("default_win_rate", 0.45)
        self.default_avg_win = kelly_cfg.get("default_avg_win_pct", 2.0) / 100
        self.default_avg_loss = kelly_cfg.get("default_avg_loss_pct", 1.5) / 100
        self.kelly_fraction = kelly_cfg.get("fraction", 0.5)  # Half-Kelly
        self.max_kelly_pct = kelly_cfg.get("max_kelly_pct", 0.25)  # Cap at 25%

        # Risk limits
        self.max_risk_per_trade_pct = sizing_cfg.get("max_risk_per_trade_pct", 2.0) / 100
        self.max_position_btc = sizing_cfg.get("max_position_size_btc", 0.1)
        self.max_exposure_pct = sizing_cfg.get("max_total_exposure_pct", 50) / 100
        self.min_position_btc = sizing_cfg.get("min_position_size_btc", 0.0001)

        # Volatility adjustment
        vol_cfg = sizing_cfg.get("volatility", {})
        self.vol_adjustment = vol_cfg.get("enabled", True)
        self.vol_adjustment_window = vol_cfg.get("window", 50)

        # Track sizing decisions for audit
        self.last_size: float = 0.0
        self.last_kelly_pct: float = 0.0
        self._last_reject: str = ""

    def calculate(
        self,
        capital: float,
        btc_price: float,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
        current_atr: float = 0.0,
        avg_atr: float = 0.0,
        consecutive_losses: int = 0,
        consecutive_wins: int = 0,
    ) -> float:
        """Calculate position size in BTC.

        Args:
            capital: Available capital in quote currency (USDT)
            btc_price: Current BTC price
            win_rate: Historical win rate (0..1), defaults to config value
            avg_win: Average winning trade return (as decimal, e.g., 0.02 = 2%)
            avg_loss: Average losing trade return (as decimal)
            current_atr: Current ATR(14) value
            avg_atr: Average ATR over lookback period
            consecutive_losses: Current consecutive losses streak
            consecutive_wins: Current consecutive wins streak

        Returns:
            Position size in BTC (0 if no position should be taken).
        """
        if btc_price <= 0 or capital <= 0:
            self._last_reject = f"invalid input: btc_price={btc_price}, capital={capital}"
            return 0.0

        # Volatility adjustment: shrink position when volatility is elevated
        vol_adj = 1.0
        if self.vol_adjustment and current_atr > 0 and avg_atr > 0:
            vol_adj = min(1.0, avg_atr / current_atr)
            vol_adj = max(0.25, vol_adj)  # Don't shrink below 25% of original

        if self.method == "fixed_fraction":
            if consecutive_losses >= 2:
                size_pct = self.max_position_pct * 0.5  # Half size after 2+ losses
            elif consecutive_wins >= 3:
                size_pct = min(self.max_position_pct * 1.5, 0.75)  # Max 75% of equity
            else:
                size_pct = self.max_position_pct

            self.last_kelly_pct = size_pct
            position_value = capital * size_pct * vol_adj
            btc_size = position_value / btc_price
        else:
            # Use defaults if not provided
            win_rate = win_rate if win_rate is not None else self.default_win_rate
            avg_win = avg_win if avg_win is not None else self.default_avg_win
            avg_loss = avg_loss if avg_loss is not None else self.default_avg_loss

            # Kelly fraction
            if avg_loss <= 0:
                avg_loss = 0.001  # Floor to prevent division by zero

            b_ratio = avg_win / avg_loss
            kelly_f = max(0.0, (win_rate * b_ratio - (1.0 - win_rate)) / b_ratio)
            kelly_f = min(kelly_f, self.max_kelly_pct)

            half_kelly = kelly_f * self.kelly_fraction
            self.last_kelly_pct = half_kelly

            if half_kelly <= 0:
                self.last_size = 0.0
                self._last_reject = f"kelly≤0: wr={win_rate:.2f} b={b_ratio:.2f} kelly_f={kelly_f:.4f}"
                return 0.0

            # Risk capital this trade can risk
            risk_capital = capital * half_kelly * vol_adj
            risk_capital = min(risk_capital, capital * self.max_risk_per_trade_pct)

            # Convert to BTC
            btc_size = risk_capital / btc_price

        # Apply caps
        btc_size = min(btc_size, self.max_position_btc)

        # Respect max exposure
        max_exposure_btc = (capital * self.max_exposure_pct) / btc_price
        btc_size = min(btc_size, max_exposure_btc)

        # Minimum position
        if btc_size < self.min_position_btc:
            self._last_reject = f"below min: {btc_size:.8f} < {self.min_position_btc}"
            btc_size = 0.0

        self.last_size = btc_size

        logger.debug(
            f"Position sizing ({self.method}): pct={self.last_kelly_pct:.4f} | "
            f"vol_adj={vol_adj:.2f} | "
            f"size={btc_size:.6f} BTC"
        )

        return round(btc_size, 8)

    def get_last_metrics(self) -> dict:
        """Return metrics from the last sizing calculation for audit."""
        return {
            "kelly_pct": round(self.last_kelly_pct, 4),
            "position_btc": self.last_size,
            "reject_reason": self._last_reject,
        }
