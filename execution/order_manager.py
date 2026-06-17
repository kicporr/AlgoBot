"""Order lifecycle management: create, track, cancel, retry."""

import time
from dataclasses import dataclass
from typing import Optional
from loguru import logger
from strategies.base import Signal


@dataclass
class Order:
    id: str
    symbol: str
    side: str
    type: str
    price: float
    amount: float
    status: str
    timestamp: int


class OrderManager:
    """Converts signals to exchange orders with retry logic."""

    def __init__(self, config: dict, exchange_adapter):
        self.config = config
        self.exchange = exchange_adapter
        exec_cfg = config.get("execution", {})
        self.order_type = exec_cfg.get("order_type", "limit")
        self.timeout_s = exec_cfg.get("order_timeout_seconds", 30)
        self.max_retries = exec_cfg.get("max_retries", 3)
        self.current_order: Optional[Order] = None

    def execute_signal(self, signal: Signal, position_size: float) -> Optional[Order]:
        """Execute a trading signal (deprecated/paper helper)."""
        # Kept for backward compatibility, new implementation uses place_order_maker_only directly.
        return None

    def place_order_maker_only(
        self,
        symbol: str,
        side: str,  # "buy" | "sell"
        amount: float,
        fallback_to_market: bool = False,
    ) -> dict:
        """Place a Maker-only limit order. If the price moves away, cancel and replace it.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order quantity
            fallback_to_market: If True, execute remainder at market price after retries

        Returns:
            Dict containing status, filled amount, and average execution price.
        """
        logger.info(f"Initiating Maker-only {side.upper()} order for {amount:.6f} {symbol}")
        
        remaining_amount = amount
        total_filled = 0.0
        pnl_sum = 0.0  # To compute weighted average price
        retries = 0
        last_order_id = None

        while remaining_amount > 0 and retries <= self.max_retries:
            # 1. Fetch Order Book to get best bid/ask
            try:
                order_book = self.exchange.fetch_order_book(symbol, limit=5)
            except Exception as e:
                logger.error(f"Failed to fetch order book: {e}")
                break

            if side == "buy":
                if not order_book.get("bids") or len(order_book["bids"]) == 0:
                    logger.error("Empty bids in order book")
                    break
                price = order_book["bids"][0][0]
            else:
                if not order_book.get("asks") or len(order_book["asks"]) == 0:
                    logger.error("Empty asks in order book")
                    break
                price = order_book["asks"][0][0]

            logger.info(
                f"[Attempt {retries}/{self.max_retries}] Placing Limit {side.upper()} "
                f"for {remaining_amount:.6f} @ ${price:.2f}"
            )

            # 2. Place limit order
            try:
                if side == "buy":
                    order_res = self.exchange.create_limit_buy_order(symbol, remaining_amount, price)
                else:
                    order_res = self.exchange.create_limit_sell_order(symbol, remaining_amount, price)
                order_id = order_res.get("id")
                last_order_id = order_id
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                break

            # 3. Monitor order status
            start_time = time.time()
            filled_this_attempt = 0.0

            while time.time() - start_time < self.timeout_s:
                time.sleep(1.0)
                try:
                    order_status = self.exchange.fetch_order(order_id, symbol)
                except Exception as e:
                    logger.warning(f"Error fetching order {order_id}: {e}")
                    continue

                status = order_status.get("status")
                filled = float(order_status.get("filled", 0.0))
                filled_this_attempt = filled

                if status == "closed":
                    logger.info(f"Order {order_id} fully filled @ ${order_status.get('average', price):.2f}")
                    break
                elif status == "canceled":
                    logger.warning(f"Order {order_id} was externally canceled")
                    break
            else:
                # Timeout reached: cancel the order
                logger.warning(f"Order {order_id} timeout reached. Canceling...")
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    # Small sleep to let cancellation propagate on exchange
                    time.sleep(1.0)
                    # Fetch one last time to get finalized filled amount
                    order_status = self.exchange.fetch_order(order_id, symbol)
                    filled_this_attempt = float(order_status.get("filled", 0.0))
                except Exception as e:
                    logger.error(f"Error canceling order {order_id}: {e}")
            
            # 4. Update stats for this attempt
            # Get average price for filled amount
            try:
                final_order = self.exchange.fetch_order(order_id, symbol)
                avg_price = float(final_order.get("average") or final_order.get("price") or price)
            except Exception:
                avg_price = price

            if filled_this_attempt > 0:
                total_filled += filled_this_attempt
                pnl_sum += filled_this_attempt * avg_price
                remaining_amount -= filled_this_attempt
                logger.info(f"Filled {filled_this_attempt:.6f} {symbol} in this attempt.")

            if remaining_amount <= 1e-8:
                remaining_amount = 0.0
                break

            retries += 1

        # 5. Fallback to Market if remaining unfilled and requested
        if remaining_amount > 0 and fallback_to_market:
            logger.warning(
                f"Order not fully filled. Remaining: {remaining_amount:.6f} {symbol}. "
                "Executing fallback Market order!"
            )
            try:
                if side == "buy":
                    m_res = self.exchange.create_market_buy_order(symbol, remaining_amount)
                else:
                    m_res = self.exchange.create_market_sell_order(symbol, remaining_amount)
                
                # Fetch ticker to estimate execution price or parse from response
                time.sleep(1.0)
                market_order_id = m_res.get("id")
                m_order = self.exchange.fetch_order(market_order_id, symbol)
                m_filled = float(m_order.get("filled", remaining_amount))
                m_avg_price = float(m_order.get("average") or m_order.get("price") or 0.0)
                if m_avg_price == 0.0:
                    # Fallback to ticker close
                    ticker = self.exchange.fetch_ticker(symbol)
                    m_avg_price = ticker.get("last", 0.0)

                total_filled += m_filled
                pnl_sum += m_filled * m_avg_price
                remaining_amount -= m_filled
                last_order_id = market_order_id
                logger.info(f"Market fallback executed: filled {m_filled:.6f} @ ${m_avg_price:.2f}")
            except Exception as e:
                logger.critical(f"Critical: Market fallback failed: {e}")

        # Compute final average price
        avg_exec_price = pnl_sum / total_filled if total_filled > 0 else 0.0
        
        status = "filled" if remaining_amount == 0 else ("partially_filled" if total_filled > 0 else "failed")
        logger.info(
            f"Maker-only order execution complete. Status: {status} | "
            f"Filled: {total_filled:.6f}/{amount:.6f} | Avg Price: ${avg_exec_price:.2f}"
        )
        
        return {
            "status": status,
            "filled": total_filled,
            "average": avg_exec_price,
            "order_id": last_order_id,
        }
