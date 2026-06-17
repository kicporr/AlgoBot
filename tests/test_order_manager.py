"""Tests for Phase 5: Maker-only Order Manager."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from unittest.mock import MagicMock
from execution.order_manager import OrderManager


class MockExchangeAdapter:
    def __init__(self):
        self.symbol = "BTC/USDT"
        self.fetch_order_book = MagicMock()
        self.create_limit_buy_order = MagicMock()
        self.create_limit_sell_order = MagicMock()
        self.create_market_buy_order = MagicMock()
        self.create_market_sell_order = MagicMock()
        self.cancel_order = MagicMock()
        self.fetch_order = MagicMock()
        self.fetch_ticker = MagicMock()


class TestOrderManager:
    def _make_manager(self, config=None):
        cfg = config or {
            "execution": {
                "order_type": "limit",
                "order_timeout_seconds": 1,  # Short timeout for testing
                "max_retries": 2,
            }
        }
        self.adapter = MockExchangeAdapter()
        return OrderManager(cfg, self.adapter)

    def test_place_order_immediate_fill(self):
        mgr = self._make_manager()
        
        # Mock order book best bid
        self.adapter.fetch_order_book.return_value = {
            "bids": [[50000.0, 1.0]],
            "asks": [[50005.0, 1.0]],
        }
        
        # Mock order placement
        self.adapter.create_limit_buy_order.return_value = {"id": "ord123"}
        
        # Mock order status polling: return immediately closed/filled
        self.adapter.fetch_order.return_value = {
            "id": "ord123",
            "status": "closed",
            "filled": 0.1,
            "average": 50000.0,
        }

        res = mgr.place_order_maker_only("BTC/USDT", "buy", 0.1)
        
        assert res["status"] == "filled"
        assert res["filled"] == 0.1
        assert res["average"] == 50000.0
        assert res["order_id"] == "ord123"
        
        self.adapter.create_limit_buy_order.assert_called_once_with("BTC/USDT", 0.1, 50000.0)

    def test_place_order_cancel_replace_retry(self):
        mgr = self._make_manager()
        
        # First attempt order book and order
        self.adapter.fetch_order_book.side_effect = [
            {"bids": [[50000.0, 1.0]], "asks": [[50005.0, 1.0]]},  # Att 0
            {"bids": [[50100.0, 1.0]], "asks": [[50105.0, 1.0]]},  # Att 1
        ]
        self.adapter.create_limit_buy_order.side_effect = [
            {"id": "ord1"},
            {"id": "ord2"},
        ]
        
        # Status checks:
        # Att 0: unfilled -> canceled on timeout
        self.adapter.fetch_order.side_effect = [
            {"id": "ord1", "status": "open", "filled": 0.0},        # poll
            {"id": "ord1", "status": "canceled", "filled": 0.0},    # after cancel
            {"id": "ord1", "status": "canceled", "filled": 0.0},    # final stats check
            # Att 1: filled
            {"id": "ord2", "status": "closed", "filled": 0.1, "average": 50100.0},
            {"id": "ord2", "status": "closed", "filled": 0.1, "average": 50100.0},
        ]
        
        res = mgr.place_order_maker_only("BTC/USDT", "buy", 0.1)
        
        assert res["status"] == "filled"
        assert res["filled"] == 0.1
        assert res["average"] == 50100.0
        assert res["order_id"] == "ord2"
        
        self.adapter.cancel_order.assert_called_once_with("ord1", "BTC/USDT")

    def test_place_order_fallback_to_market(self):
        mgr = self._make_manager()
        
        self.adapter.fetch_order_book.return_value = {
            "bids": [[50000.0, 1.0]],
            "asks": [[50005.0, 1.0]],
        }
        self.adapter.create_limit_buy_order.return_value = {"id": "ord_limit"}
        
        # Limit order remains open and gets cancelled with 0 filled on all 3 attempts
        self.adapter.fetch_order.side_effect = [
            {"id": "ord_limit", "status": "open", "filled": 0.0},  # poll (Att 0)
            {"id": "ord_limit", "status": "canceled", "filled": 0.0},  # after cancel
            {"id": "ord_limit", "status": "canceled", "filled": 0.0},  # stats
            
            {"id": "ord_limit", "status": "open", "filled": 0.0},  # poll (Att 1)
            {"id": "ord_limit", "status": "canceled", "filled": 0.0},  # after cancel
            {"id": "ord_limit", "status": "canceled", "filled": 0.0},  # stats
            
            {"id": "ord_limit", "status": "open", "filled": 0.0},  # poll (Att 2)
            {"id": "ord_limit", "status": "canceled", "filled": 0.0},  # after cancel
            {"id": "ord_limit", "status": "canceled", "filled": 0.0},  # stats
            
            # Market order fetch order status
            {"id": "ord_market", "status": "closed", "filled": 0.1, "average": 50500.0},
        ]
        
        # Mock market order placement
        self.adapter.create_market_buy_order.return_value = {"id": "ord_market"}

        res = mgr.place_order_maker_only("BTC/USDT", "buy", 0.1, fallback_to_market=True)
        
        assert res["status"] == "filled"
        assert res["filled"] == 0.1
        assert res["average"] == 50500.0
        assert res["order_id"] == "ord_market"
        self.adapter.create_market_buy_order.assert_called_once_with("BTC/USDT", 0.1)
