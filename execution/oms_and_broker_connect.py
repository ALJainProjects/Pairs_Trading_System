from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
import uuid
import random

from config.logging_config import logger
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER_TRADING

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
except ImportError:
    logger.error("Alpaca API not installed. Run 'pip install alpaca-trade-api'")
    tradeapi = None
    APIError = None


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
    CANCELED = "canceled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Represents a single trading order with a unique ID."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None # For limit orders
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    """Represents an open position in a single asset."""
    symbol: str
    quantity: float  # Positive for long, negative for short
    average_entry_price: float
    
    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price
        
    def unrealized_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.average_entry_price)

# --- Broker Interface and Implementations ---

class AbstractBroker(ABC):
    """Abstract base class defining the interface for all broker implementations."""
    @abstractmethod
    def place_order(self, order: Order) -> bool: pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]: pass

    @abstractmethod
    def get_all_positions(self) -> List[Position]: pass
    
    @abstractmethod
    def get_cash_balance(self) -> float: pass

class SimulatedBroker(AbstractBroker):
    """A simulated broker for backtesting with realistic execution modeling."""
    def __init__(self, initial_capital: float, transaction_cost_pct: float):
        self.cash = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.trade_log: List[Dict] = []
        self.current_prices: Dict[str, float] = {}

    def update_market_data(self, prices: Dict[str, float]):
        """Called by the backtester to provide current market prices."""
        self.current_prices = prices

    def place_order(self, order: Order) -> bool:
        self.open_orders.append(order)
        return True

    def execute_orders(self, slippage_pct: float = 0.0005):
        """Processes open orders based on current market prices."""
        for order in self.open_orders:
            if order.status == OrderStatus.PENDING:
                price = self.current_prices.get(order.symbol)
                if price:
                    # Apply slippage
                    slippage = price * slippage_pct * random.choice([-1, 1])
                    fill_price = price + slippage
                    
                    self._execute_fill(order, fill_price)
        self.open_orders = [o for o in self.open_orders if o.status == OrderStatus.PENDING]
    
    def _execute_fill(self, order: Order, fill_price: float):
        trade_value = order.quantity * fill_price
        cost = trade_value * self.transaction_cost_pct
        
        if order.side == OrderSide.BUY:
            if self.cash < trade_value + cost:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected for {order.symbol}: Insufficient cash.")
                return
            
            self.cash -= (trade_value + cost)
            
            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                new_total_value = (pos.average_entry_price * pos.quantity) + trade_value
                pos.quantity += order.quantity
                pos.average_entry_price = new_total_value / pos.quantity if pos.quantity != 0 else 0
            else:
                self.positions[order.symbol] = Position(order.symbol, order.quantity, fill_price)

        elif order.side == OrderSide.SELL:
            current_qty = self.positions.get(order.symbol, Position(order.symbol, 0, 0)).quantity
            if order.quantity > current_qty:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected for {order.symbol}: Not enough shares to sell.")
                return
            
            self.cash += (trade_value - cost)
            
            # Update position
            pos = self.positions[order.symbol]
            pos.quantity -= order.quantity
            if abs(pos.quantity) < 1e-6: # Position closed
                del self.positions[order.symbol]

        order.status = OrderStatus.FILLED
        self.trade_log.append({
            'date': self.current_prices.get('date'),
            'symbol': order.symbol, 'side': order.side.value, 'quantity': order.quantity,
            'price': fill_price, 'cost': cost
        })

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        return list(self.positions.values())
        
    def get_all_positions_in_dollars(self) -> Dict[str, float]:
        return {p.symbol: p.market_value(self.current_prices.get(p.symbol, 0)) for p in self.positions.values()}

    def get_cash_balance(self) -> float:
        return self.cash
        
    def get_portfolio_value(self) -> float:
        market_value = sum(self.get_all_positions_in_dollars().values())
        return self.cash + market_value

    def get_trade_log(self) -> List[Dict]:
        return self.trade_log

class LiveBroker(AbstractBroker):
    """
    A broker implementation that connects to the Alpaca API for paper or live trading.
    """
    def __init__(self):
        if not all([tradeapi, APIError, ALPACA_API_KEY, ALPACA_SECRET_KEY]):
            raise ImportError("Alpaca API not configured. Check settings and ensure installation.")
        
        base_url = "https://paper-api.alpaca.markets" if IS_PAPER_TRADING else "https://api.alpaca.markets"
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')
        logger.info(f"LiveBroker connected to Alpaca. Paper Trading: {IS_PAPER_TRADING}")

    def place_order(self, order: Order) -> bool:
        try:
            api_order = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side.value,
                type=order.order_type.value,
                time_in_force='day'
            )
            logger.info(f"Placed order with Alpaca: {api_order.id} for {order.symbol}")
            return True
        except APIError as e:
            logger.error(f"Alpaca API error placing order for {order.symbol}: {e}")
            return False

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            pos = self.api.get_position(symbol)
            return Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                average_entry_price=float(pos.avg_entry_price)
            )
        except APIError as e:
            if "position not found" in str(e):
                return None
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