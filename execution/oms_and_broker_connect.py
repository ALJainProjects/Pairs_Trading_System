from abc import ABC, abstractmethod

import pandas as pd
from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime, timezone  # Import timezone for aware datetimes
from dataclasses import dataclass, field
import uuid
import random
import time

# Assuming logging_config.py exists and configures a logger
try:
    from config.logging_config import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Assuming config/settings.py exists with these variables
try:
    from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER_TRADING
except ImportError:
    logger.warning("config/settings.py not found. Using dummy Alpaca credentials and default paper trading.")
    ALPACA_API_KEY = "DUMMY_KEY"
    ALPACA_SECRET_KEY = "DUMMY_SECRET"
    IS_PAPER_TRADING = True

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError, URL  # Import URL directly from alpaca_trade_api.rest
except ImportError:
    logger.error("Alpaca API not installed. Run 'pip install alpaca-trade-api'")
    tradeapi = None
    APIError = None
    URL = None  # Set to None for type hinting if not installed


# --- Order Management System (OMS) Core Classes ---

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Represents a single trading order with a unique ID."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None  # For limit orders
    time_in_force: str = "day"  # For live brokers: 'day', 'gtc', 'opg', 'ioc', 'fok'

    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = field(default_factory=lambda: f"algo_order_{str(uuid.uuid4())}")

    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # Timezone-aware timestamp

    filled_quantity: float = 0.0
    filled_price: Optional[float] = None

    alpaca_id: Optional[str] = None


@dataclass
class Position:
    """Represents an open position in a single asset."""
    symbol: str
    quantity: float  # Positive for long, negative for short
    average_entry_price: float  # Weighted average entry price

    def market_value(self, current_price: float) -> float:
        """Calculates current market value of the position."""
        return self.quantity * current_price  # Works for both long and short (negative quantity)

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculates unrealized Profit and Loss."""
        # PnL for long: qty * (current - entry)
        # PnL for short: qty * (entry - current) where qty is negative -> |qty| * (entry - current)
        return self.quantity * (current_price - self.average_entry_price)  # This works for negative quantity too


# --- Broker Interface and Implementations ---

class AbstractBroker(ABC):
    """Abstract base class defining the interface for all broker implementations."""

    @abstractmethod
    def place_order(self, order: Order) -> bool:
        """Places a new order with the broker."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Requests cancellation of an order by its internal ID."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Gets the current status of an order."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Gets a list of all currently open/pending orders."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Gets the current position for a given symbol."""
        pass

    @abstractmethod
    def get_all_positions(self) -> List[Position]:
        """Gets a list of all open positions."""
        pass

    @abstractmethod
    def get_cash_balance(self) -> float:
        """Gets the current cash balance in the account."""
        pass

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Gets the total portfolio value (cash + positions market value)."""
        pass

    # New method to be called by backtester to trigger order execution
    @abstractmethod
    def execute_orders(self, slippage_pct: float = 0.0005):
        """
        Processes pending orders. For simulated broker, this means attempting fills.
        For live broker, this means polling for status updates.
        """
        pass


class SimulatedBroker(AbstractBroker):
    """A simulated broker for backtesting with realistic execution modeling."""

    def __init__(self, initial_capital: float, transaction_cost_pct: float = 0.001):
        self.cash: float = initial_capital
        self.transaction_cost_pct: float = transaction_cost_pct
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_log: List[Dict] = []
        self.current_prices: Dict[str, float] = {}

    def update_market_data(self, prices: Dict[str, float]):
        """Called by the backtester to provide current market prices."""
        self.current_prices = prices

    def place_order(self, order: Order) -> bool:
        if order.order_id in self.orders:
            logger.warning(f"Order with ID {order.order_id} already exists.")
            return False

        # Basic validation for quantity
        if order.quantity <= 0:
            logger.warning(f"Order quantity must be positive for {order.symbol}. Rejected.")
            order.status = OrderStatus.REJECTED
            self.orders[order.order_id] = order
            return False

        # Validate price for limit orders
        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.warning(f"Limit order for {order.symbol} requires a price. Rejected.")
            order.status = OrderStatus.REJECTED
            self.orders[order.order_id] = order
            return False

        self.orders[order.order_id] = order
        logger.info(
            f"Placed simulated order: {order.symbol} {order.side.value} {order.quantity} {order.order_type.value}")
        return True

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELED
            logger.info(f"Simulated order {order_id} for {order.symbol} cancelled.")
            return True
        logger.warning(f"Cannot cancel simulated order {order_id}: not found or not pending.")
        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [order for order in self.orders.values() if order.status == OrderStatus.PENDING]

    def execute_orders(self, slippage_pct: float = 0.0005):
        """Processes pending orders based on current market prices and applies fills."""
        orders_to_process = list(self.orders.values())

        for order in orders_to_process:
            if order.status == OrderStatus.PENDING:
                current_price_for_symbol = self.current_prices.get(order.symbol)

                if current_price_for_symbol is None:
                    # No price data for this symbol today, keep pending
                    logger.debug(
                        f"No price data for {order.symbol} on {self.current_prices.get('date', 'N/A')}. Order {order.order_id} remains pending.")
                    continue

                # Apply slippage (randomized for simulation)
                slippage_amount = current_price_for_symbol * slippage_pct * random.uniform(-1, 1)
                fill_price = current_price_for_symbol + slippage_amount

                # Check for limit order conditions
                if order.order_type == OrderType.LIMIT:
                    if order.price is None:  # Should have been caught by place_order
                        order.status = OrderStatus.REJECTED
                        logger.error(f"Limit order {order.order_id} for {order.symbol} missing price. Rejected.")
                        continue
                    if order.side == OrderSide.BUY and fill_price > order.price:
                        continue  # Price too high for a buy limit order
                    if order.side == OrderSide.SELL and fill_price < order.price:
                        continue  # Price too low for a sell limit order

                # If conditions met or it's a market order, attempt fill
                self._process_fill(order, fill_price)

    def _process_fill(self, order: Order, fill_price: float):
        trade_value = order.quantity * fill_price
        cost = trade_value * self.transaction_cost_pct

        # Check for sufficient cash (for BUY) or position (for SELL/SHORT)
        can_execute = False
        if order.side == OrderSide.BUY:
            if self.cash >= trade_value + cost:
                can_execute = True
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(
                    f"Simulated order rejected for {order.symbol} BUY: Insufficient cash ({self.cash:.2f} < {trade_value + cost:.2f}).")
        elif order.side == OrderSide.SELL:
            # For selling, we distinguish between closing a long position, opening a short, or covering a short.
            # Here, we support short selling if the quantity to sell is more than the long position, or no long position.
            current_pos = self.positions.get(order.symbol)
            current_qty = current_pos.quantity if current_pos else 0.0

            if order.quantity <= current_qty + 1e-9:  # Selling from existing long or closing exactly
                can_execute = True
            elif order.side == OrderSide.SELL and current_qty < 1e-9:  # Opening a new short position (no current long shares)
                # Model margin for short sales if desired. For simplicity, just allow it if cash is sufficient.
                # A short sale typically means you receive cash, but also have collateral requirements.
                # Here, we assume the broker handles the margin/collateral, just check cash for costs.
                if self.cash >= cost:  # Only transaction cost is a cash outlay for opening short.
                    can_execute = True
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(
                        f"Simulated order rejected for {order.symbol} SHORT: Insufficient cash for costs ({self.cash:.2f} < {cost:.2f}).")
            else:  # Trying to sell more than current long position, turning into a short
                # This path needs more sophisticated margin handling. For basic simulation, allow if cash covers costs.
                if self.cash >= cost:
                    can_execute = True
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(
                        f"Simulated order rejected for {order.symbol} SHORT/PARTIAL COVER: Insufficient cash for costs ({self.cash:.2f} < {cost:.2f}).")

        if can_execute:
            self.cash -= cost  # Deduct transaction cost regardless of side

            # Adjust cash and position
            if order.side == OrderSide.BUY:
                self.cash -= trade_value  # Cash decreases
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    # Handle buying to reduce short or increase long
                    if pos.quantity < 0:  # Currently short, buying to cover
                        new_qty = pos.quantity + order.quantity
                        if new_qty >= -1e-9:  # Covered fully or moved to long
                            # If moved to long, avg entry is fill price. If covered, delete.
                            if new_qty > 1e-9:  # Moved to long
                                self.positions[order.symbol] = Position(order.symbol, new_qty, fill_price)
                            else:  # Covered exactly
                                del self.positions[order.symbol]
                        else:  # Still short after partial cover
                            # Avg entry for short position: remain the same (unless you average by dollars, complex)
                            # Simpler: just update quantity for short. Avg price doesn't meaningfully change in simple model.
                            pos.quantity = new_qty
                    else:  # Currently long or flat, buying to increase long
                        new_total_value = (pos.average_entry_price * pos.quantity) + trade_value
                        pos.quantity += order.quantity
                        pos.average_entry_price = new_total_value / pos.quantity if pos.quantity != 0 else 0
                else:  # No position, opening new long
                    self.positions[order.symbol] = Position(order.symbol, order.quantity, fill_price)

            elif order.side == OrderSide.SELL:
                self.cash += trade_value  # Cash increases from sale
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    # Handle selling to reduce long or open/increase short
                    new_qty = pos.quantity - order.quantity
                    if abs(new_qty) < 1e-9:  # Position closed
                        del self.positions[order.symbol]
                    else:
                        if (pos.quantity > 0 and new_qty < 0):  # Transition from long to short
                            self.positions[order.symbol] = Position(order.symbol, new_qty,
                                                                    fill_price)  # New entry price is fill price
                        elif (pos.quantity < 0 and new_qty < 0):  # Increasing short position
                            # For simplicity, don't re-average entry for short sells. Keep old entry price.
                            # Or re-average as a weighted average. Let's average if short.
                            new_total_value = (pos.average_entry_price * pos.quantity) + (
                                -trade_value)  # Value added to short position
                            pos.quantity = new_qty
                            pos.average_entry_price = new_total_value / pos.quantity if pos.quantity != 0 else 0  # Avg price for short is average exit price.
                        else:  # Reducing long position
                            pos.quantity = new_qty
                else:  # No position, opening new short
                    self.positions[order.symbol] = Position(order.symbol, -order.quantity,
                                                            fill_price)  # Quantity is negative for short

            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = fill_price

            # Log trade with correct date
            log_date = self.current_prices.get('date', datetime.now(timezone.utc).isoformat())
            self.trade_log.append({
                'date': log_date,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': fill_price,
                'transaction_cost': cost,
                'order_id': order.order_id
            })
            logger.info(
                f"Simulated order {order.symbol} {order.side.value} {order.quantity} filled at {fill_price:.4f}. Cash: {self.cash:.2f}")

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_all_positions_in_dollars(self) -> Dict[str, float]:
        current_prices = self.current_prices
        if not current_prices:
            logger.warning("No market data available to calculate position values. Returning empty dict.")
            return {}
        # Ensure we don't try to get market value if symbol is not in current_prices (can happen for closed positions)
        return {p.symbol: p.market_value(current_prices.get(p.symbol, 0))
                for p in self.positions.values() if
                p.symbol in current_prices and current_prices.get(p.symbol) is not None}

    def get_cash_balance(self) -> float:
        return self.cash

    def get_portfolio_value(self) -> float:
        market_value = sum(p.market_value(self.current_prices.get(p.symbol, p.average_entry_price))
                           # Fallback to entry price if current price is missing
                           for p in self.positions.values() if p.symbol in self.current_prices)
        return self.cash + market_value

    def get_trade_log(self) -> List[Dict]:
        return self.trade_log


class LiveBroker(AbstractBroker):
    """
    A broker implementation that connects to the Alpaca API for paper or live trading.
    """

    def __init__(self):
        if not all([tradeapi, APIError, URL, ALPACA_API_KEY, ALPACA_SECRET_KEY]):  # Added URL to check
            raise ImportError("Alpaca API not configured. Check settings and ensure installation.")

        base_url = "https://paper-api.alpaca.markets" if IS_PAPER_TRADING else "https://api.alpaca.markets"
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, URL(base_url), api_version='v2')
        self.internal_orders: Dict[str, Order] = {}
        logger.info(f"LiveBroker connected to Alpaca. Paper Trading: {IS_PAPER_TRADING}")

    def place_order(self, order: Order) -> bool:
        if order.order_id in self.internal_orders:
            logger.warning(f"Order with internal ID {order.order_id} already exists in tracking.")
            return False
        if order.quantity <= 0:
            logger.warning(f"Order quantity must be positive for {order.symbol}. Rejected by client.")
            order.status = OrderStatus.REJECTED
            self.internal_orders[order.order_id] = order
            return False

        try:
            alpaca_order_params = {
                'symbol': order.symbol,
                'qty': order.quantity,
                'side': order.side.value,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force,
                'client_order_id': order.client_order_id
            }
            if order.order_type == OrderType.LIMIT:
                if order.price is None: raise ValueError("Limit order requires a 'price'.")
                alpaca_order_params['limit_price'] = order.price

            api_order = self.api.submit_order(**alpaca_order_params)

            logger.info(f"Placed order with Alpaca: {api_order.id} for {order.symbol} (Internal: {order.order_id})")

            order.alpaca_id = api_order.id
            order.status = OrderStatus.PENDING
            self.internal_orders[order.order_id] = order
            return True
        except APIError as e:
            logger.error(f"Alpaca API error placing order for {order.symbol}: {e}")
            order.status = OrderStatus.REJECTED
            self.internal_orders[order.order_id] = order
            return False
        except ValueError as e:
            logger.error(f"Order validation error: {e}")
            order.status = OrderStatus.REJECTED
            self.internal_orders[order.order_id] = order
            return False

    def cancel_order(self, order_id: str) -> bool:
        order = self.internal_orders.get(order_id)
        if not order or not order.alpaca_id:
            logger.warning(f"Cannot cancel order {order_id}: Not found or no Alpaca ID.")
            return False

        try:
            self.api.cancel_order(order.alpaca_id)
            logger.info(f"Cancellation request sent for order {order.symbol} (Alpaca ID: {order.alpaca_id})")
            return True
        except APIError as e:
            logger.error(f"Alpaca API error canceling order {order.alpaca_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self.internal_orders.get(order_id)

    def execute_orders(self, slippage_pct: float = 0.0005):  # slippage_pct is not used for live
        """
        Polls Alpaca for the latest statuses of all tracked open orders.
        `slippage_pct` parameter is ignored for live execution as fills are actual.
        """
        self.update_all_order_statuses()

    def update_all_order_statuses(self):
        """Polls Alpaca for the latest statuses of all tracked open orders."""
        orders_to_poll = [o for o in self.internal_orders.values()
                          if o.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]]

        if not orders_to_poll: return
        logger.info(f"Polling Alpaca for status of {len(orders_to_poll)} open orders...")

        try:
            alpaca_open_orders = {o.id: o for o in self.api.list_orders(status='open')}
        except APIError as e:
            logger.error(f"Error listing open orders from Alpaca: {e}")
            return

        for order in orders_to_poll:
            if not order.alpaca_id: continue

            try:
                alpaca_order_data = alpaca_open_orders.get(order.alpaca_id)
                if alpaca_order_data is None:
                    alpaca_order_data = self.api.get_order(order.alpaca_id)

                # Map Alpaca status to internal OrderStatus Enum
                if alpaca_order_data.status == 'filled':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = float(alpaca_order_data.filled_qty)
                    order.filled_price = float(alpaca_order_data.filled_avg_price)
                    logger.info(f"Order {order.symbol} (ID: {order.order_id}) FILLED at {order.filled_price}")
                    del self.internal_orders[order.order_id]
                elif alpaca_order_data.status in ['canceled', 'expired', 'rejected']:
                    order.status = OrderStatus[alpaca_order_data.status.upper()]  # Direct mapping for these
                    logger.info(f"Order {order.symbol} (ID: {order.order_id}) {order.status.value.upper()}.")
                    del self.internal_orders[order.order_id]
                elif alpaca_order_data.status == 'partially_filled':
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = float(alpaca_order_data.filled_qty)
                    order.filled_price = float(alpaca_order_data.filled_avg_price)
                    logger.debug(
                        f"Order {order.symbol} (ID: {order.order_id}) PARTIALLY FILLED. Qty: {order.filled_quantity}.")
                elif alpaca_order_data.status in ['new', 'pending_new', 'accepted', 'queue', 'held']:
                    order.status = OrderStatus.PENDING
                    logger.debug(
                        f"Order {order.symbol} (ID: {order.order_id}) is PENDING. Alpaca Status: {alpaca_order_data.status}")

            except APIError as e:
                logger.error(f"Error fetching status for Alpaca order {order.alpaca_id}: {e}")

    def get_open_orders(self) -> List[Order]:
        return [order for order in self.internal_orders.values()
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]]

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            pos = self.api.get_position(symbol)
            return Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                average_entry_price=float(pos.avg_entry_price)
            )
        except APIError as e:
            if "position not found" in str(e).lower(): return None
            logger.error(f"Alpaca API error getting position for {symbol}: {e}")
            return None

    def get_all_positions(self) -> List[Position]:
        try:
            positions = self.api.list_positions()
            return [Position(p.symbol, float(p.qty), float(p.avg_entry_price)) for p in positions]
        except APIError as e:
            logger.error(f"Alpaca API error getting all positions: {e}")
            return []

    def get_cash_balance(self) -> float:
        try:
            account = self.api.get_account()
            return float(account.cash)
        except APIError as e:
            logger.error(f"Alpaca API error getting cash balance: {e}")
            return 0.0

    def get_portfolio_value(self) -> float:
        try:
            account = self.api.get_account()
            return float(account.equity)
        except APIError as e:
            logger.error(f"Alpaca API error getting portfolio value: {e}")
            return self.get_cash_balance()  # Fallback to cash if equity fails


# --- Example Usage (main function and dummy data for testing) ---
def main():
    """Example usage."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)

    # --- Test SimulatedBroker ---
    print("\n--- Testing SimulatedBroker ---")
    sim_broker = SimulatedBroker(initial_capital=100000.0, transaction_cost_pct=0.001)

    # Simulate market data
    sim_broker.update_market_data(
        {'GOOG': 150.0, 'AAPL': 170.0, 'MSFT': 200.0, 'Date': datetime.now(timezone.utc).date()})

    # Place a market buy order (Long GOOG)
    order1 = Order(symbol='GOOG', side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
    sim_broker.place_order(order1)
    sim_broker.execute_orders()  # Explicitly execute after placing

    print(f"Cash: {sim_broker.get_cash_balance():.2f}, Portfolio Value: {sim_broker.get_portfolio_value():.2f}")
    print(f"GOOG pos: {sim_broker.get_position('GOOG')}")
    print(f"Order 1 status: {sim_broker.get_order_status(order1.order_id).status}")

    # Place a market sell order (Short AAPL) - this will open a short position
    order_short_aapl = Order(symbol='AAPL', side=OrderSide.SELL, quantity=5, order_type=OrderType.MARKET)
    sim_broker.place_order(order_short_aapl)
    sim_broker.execute_orders()

    print(f"\nCash: {sim_broker.get_cash_balance():.2f}, Portfolio Value: {sim_broker.get_portfolio_value():.2f}")
    print(f"AAPL pos (short): {sim_broker.get_position('AAPL')}")
    print(f"Order AAPL status: {sim_broker.get_order_status(order_short_aapl.order_id).status}")

    # Place a limit buy order (should not fill immediately if price is higher)
    order2 = Order(symbol='MSFT', side=OrderSide.BUY, quantity=10, order_type=OrderType.LIMIT, price=199.0)
    sim_broker.place_order(order2)
    sim_broker.execute_orders()  # No fill

    print(f"\nMSFT position (before limit fill): {sim_broker.get_position('MSFT')}")
    print(f"Order 2 status: {sim_broker.get_order_status(order2.order_id).status}")
    print(f"Open orders: {len(sim_broker.get_open_orders())}")

    # Simulate price drop for MSFT to fill limit order
    sim_broker.update_market_data(
        {'GOOG': 151.0, 'AAPL': 169.5, 'MSFT': 198.5, 'Date': datetime.now(timezone.utc).date()})
    sim_broker.execute_orders()  # Fill triggered

    print(f"\nCash: {sim_broker.get_cash_balance():.2f}, Portfolio Value: {sim_broker.get_portfolio_value():.2f}")
    print(f"MSFT position (after limit fill): {sim_broker.get_position('MSFT')}")
    print(f"Order 2 status: {sim_broker.get_order_status(order2.order_id).status}")
    print(f"Open orders: {len(sim_broker.get_open_orders())}")

    # Test selling part of GOOG position
    order3 = Order(symbol='GOOG', side=OrderSide.SELL, quantity=5, order_type=OrderType.MARKET)
    sim_broker.place_order(order3)
    sim_broker.execute_orders()

    print(f"\nCash: {sim_broker.get_cash_balance():.2f}, Portfolio Value: {sim_broker.get_portfolio_value():.2f}")
    print(f"GOOG position (after partial sell): {sim_broker.get_position('GOOG')}")

    # Test covering part of AAPL short position
    order_cover_aapl = Order(symbol='AAPL', side=OrderSide.BUY, quantity=2, order_type=OrderType.MARKET)
    sim_broker.place_order(order_cover_aapl)
    sim_broker.execute_orders()

    print(f"\nCash: {sim_broker.get_cash_balance():.2f}, Portfolio Value: {sim_broker.get_portfolio_value():.2f}")
    print(f"AAPL position (after partial cover): {sim_broker.get_position('AAPL')}")
    print(f"Trade Log: {sim_broker.get_trade_log()}")

    # Test cancellation
    order4 = Order(symbol='AMZN', side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
    sim_broker.place_order(order4)
    print(f"\nOpen orders after order 4: {len(sim_broker.get_open_orders())}")
    sim_broker.cancel_order(order4.order_id)
    sim_broker.execute_orders()  # Attempt to execute, but it's cancelled
    print(f"Order 4 status after cancellation: {sim_broker.get_order_status(order4.order_id).status}")
    print(f"Open orders after cancelling order 4: {len(sim_broker.get_open_orders())}")

    # --- Test LiveBroker (requires Alpaca API keys and internet) ---
    print("\n--- Testing LiveBroker (requires valid Alpaca API setup) ---")
    live_broker = None
    try:
        live_broker = LiveBroker()

        print(f"Live Broker Cash Balance: {live_broker.get_cash_balance():.2f}")
        print(f"Live Broker Portfolio Value: {live_broker.get_portfolio_value():.2f}")
        print(f"Live Broker All Positions: {live_broker.get_all_positions()}")
        print(f"Live Broker Open Orders (before placing): {len(live_broker.get_open_orders())}")

        dummy_symbol = "SPY"
        dummy_quantity = 1

        test_market_order = Order(symbol=dummy_symbol, side=OrderSide.BUY, quantity=dummy_quantity,
                                  order_type=OrderType.MARKET)
        market_order_placed = live_broker.place_order(test_market_order)
        if market_order_placed:
            print(f"Market order placed for {dummy_symbol}. Waiting a few seconds for execution update...")
            time.sleep(5)
            live_broker.execute_orders()  # Call execute_orders for live broker
            print(f"Market order status: {live_broker.get_order_status(test_market_order.order_id).status}")
            print(
                f"Market order filled qty: {live_broker.get_order_status(test_market_order.order_id).filled_quantity}")
            print(f"Market order filled price: {live_broker.get_order_status(test_market_order.order_id).filled_price}")
        else:
            print("Failed to place market order.")

        open_orders = live_broker.get_open_orders()
        if open_orders:
            order_to_cancel = open_orders[0]
            print(f"Attempting to cancel order: {order_to_cancel.symbol} (ID: {order_to_cancel.order_id})")
            cancel_success = live_broker.cancel_order(order_to_cancel.order_id)
            if cancel_success:
                print("Cancellation request sent. Waiting for status update...")
                time.sleep(5)
                live_broker.execute_orders()  # Call execute_orders for live broker
                print(f"Status after cancellation: {live_broker.get_order_status(order_to_cancel.order_id).status}")
            else:
                print("Failed to send cancellation request.")
        else:
            print("No open orders to test cancellation.")


    except ImportError as e:
        print(f"Skipping LiveBroker test: {e}")
    except APIError as e:
        print(f"Alpaca API error during LiveBroker test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during LiveBroker test: {e}")


if __name__ == "__main__":
    main()