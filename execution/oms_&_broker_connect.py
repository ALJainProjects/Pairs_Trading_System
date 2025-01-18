"""
Broker Module

Provides interfaces for executing trades with a broker (simulated or live).
It includes order management, position tracking, account balance,
and transaction cost handling if desired.
"""

from typing import Dict, Optional
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from config.logging_config import logger


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    status: str = "pending"
    timestamp: datetime = datetime.now()
    order_id: Optional[str] = None


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime = datetime.now()
    pnl: float = 0.0

    def update(self, current_price: float):
        """Update position PnL with the new current price."""
        self.current_price = current_price
        self.pnl = (self.current_price - self.entry_price) * self.quantity


class AbstractBroker(ABC):
    """
    Abstract base class for broker implementations.
    Typically used by a backtester or a live environment
    to place/cancel orders and track positions.
    """

    @abstractmethod
    def place_order(self, order: Order) -> bool:
        """Place a new order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        pass


class SimulatedBroker(AbstractBroker):
    """
    Simulated broker for backtesting or paper trading.

    - Stores a dictionary of positions and orders.
    - Optionally handles transaction cost internally,
      though your Backtester can also do it.
    - Immediately executes MARKET orders; other order types
      remain in 'pending' until something triggers them.
    """

    def __init__(self, initial_balance: float = 100000, transaction_cost: float = 0.001):
        """
        Args:
            initial_balance (float): Starting account balance.
            transaction_cost (float): e.g. 0.001 => 0.1% per trade.
        """
        self.balance = initial_balance
        self.transaction_cost = transaction_cost

        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0

    def place_order(self, order: Order) -> bool:
        """
        Place a new order in the simulated environment.
        MARKET orders are filled immediately; others are stored until triggered.
        """
        try:
            self.order_counter += 1
            order.order_id = f"ORDER_{self.order_counter}"

            if not self._validate_order(order):
                return False

            if order.order_type == OrderType.MARKET:
                return self._execute_order(order)

            self.orders[order.order_id] = order
            logger.info(f"Pending order placed: {order}")
            return True

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing pending order."""
        if order_id in self.orders:
            del self.orders[order_id]
            logger.info(f"Order cancelled: {order_id}")
            return True
        logger.warning(f"No such order_id {order_id} to cancel.")
        return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the position object for a given symbol, if any."""
        return self.positions.get(symbol)

    def get_account_balance(self) -> float:
        """Get current account balance."""
        return self.balance

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update all positions with current market prices.
        Called typically once per bar/day/tick in your backtest or event loop.
        """
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update(current_prices[symbol])

    def _validate_order(self, order: Order) -> bool:
        """Check if the order parameters are valid (enough balance, correct price, etc.)."""
        try:
            if order.quantity <= 0:
                logger.error(f"Invalid order quantity {order.quantity}")
                return False

            if order.order_type != OrderType.MARKET:
                if (order.price is None) or (order.price <= 0):
                    logger.error(f"Invalid price {order.price} for {order.order_type}")
                    return False

            if order.side == OrderSide.BUY:
                cost_estimate = order.quantity * (order.price or 0)
                cost_estimate += cost_estimate * self.transaction_cost
                if cost_estimate > self.balance:
                    logger.error("Insufficient balance to place buy order.")
                    return False

            if order.side == OrderSide.SELL:
                pos = self.get_position(order.symbol)
                if not pos or pos.quantity < order.quantity:
                    logger.error("Insufficient position for sell order.")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False

    def _execute_order(self, order: Order) -> bool:
        """
        Execute the order in the simulated environment.
        Adjust balance and positions immediately for MARKET orders.
        """
        try:
            fill_price = order.price if order.price else 100.0

            order_cost = fill_price * order.quantity
            cost_fee = order_cost * self.transaction_cost

            if order.side == OrderSide.BUY:
                total_spend = order_cost + cost_fee
                self.balance -= total_spend
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    new_qty = pos.quantity + order.quantity
                    avg_price = (pos.entry_price * pos.quantity + order_cost) / new_qty
                    pos.quantity = new_qty
                    pos.entry_price = avg_price
                else:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        entry_price=fill_price,
                        current_price=fill_price
                    )

            elif order.side == OrderSide.SELL:
                proceeds = order_cost - cost_fee
                self.balance += proceeds
                pos = self.positions.get(order.symbol)
                if pos:
                    new_qty = pos.quantity - order.quantity
                    if new_qty > 0:
                        pos.quantity = new_qty
                    else:
                        del self.positions[order.symbol]

            order.status = "filled"
            logger.info(f"Order executed: {order}")
            return True

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return False


class LiveBroker(AbstractBroker):
    """
    Live broker interface for real trading with an external API.
    Typically you'd use an API client like Interactive Brokers,
    Alpaca, TDA, or other. This class is an example stub.
    """

    def __init__(self, api_key: str, api_secret: str, paper_trading: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        # self.client = (Some Client Library)

    def place_order(self, order: Order) -> bool:
        raise NotImplementedError("Implement your live broker order placement here.")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("Implement your live broker order cancellation here.")

    def get_position(self, symbol: str) -> Optional[Position]:
        raise NotImplementedError("Implement your live broker position retrieval here.")

    def get_account_balance(self) -> float:
        raise NotImplementedError("Implement your live broker balance retrieval here.")
