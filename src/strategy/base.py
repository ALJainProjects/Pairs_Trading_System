"""
Enhanced Base Strategy Module

Defines abstract base class for trading strategies with improved risk management,
performance optimization, error handling, and feature extensions.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import statsmodels.api as sm
from config.logging_config import logger

@dataclass
class Position:
    """Track position details with enhanced risk management."""
    quantity: float
    entry_price1: float
    entry_price2: float
    entry_date: pd.Timestamp
    hedge_ratio: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_drawdown: Optional[float] = None
    transaction_costs: float = 0.0
    entry_sizes: List[Tuple[float, float]] = None

    def __post_init__(self):
        self.entry_sizes = [] if self.entry_sizes is None else self.entry_sizes
        self.peak_pnl: float = 0.0
        self.current_drawdown: float = 0.0

class BaseStrategy(ABC):
    """Enhanced abstract base class for all trading strategies."""

    def __init__(self,
                 name: str,
                 max_position_size: float = float('inf'),
                 max_portfolio_exposure: float = 1.0,
                 transaction_cost_pct: float = 0.0):
        """
        Initialize strategy with risk parameters.

        Args:
            name: Strategy identifier
            max_position_size: Maximum size for any single position
            max_portfolio_exposure: Maximum total portfolio exposure (1.0 = 100%)
            transaction_cost_pct: Transaction cost as percentage of trade value
        """
        self.name = name
        self.pairs: List[Tuple[str, str]] = []
        self.positions: Dict[Tuple[str, str], Position] = {}
        self.trades: List[Dict] = []

        self.max_position_size = max_position_size
        self.max_portfolio_exposure = max_portfolio_exposure
        self.transaction_cost_pct = transaction_cost_pct

        self.feature_cache: Dict = {}
        self.max_cache_size: int = 1000

        self.hedge_ratio_cache: Dict[Tuple[str, str], Tuple[float, pd.Timestamp]] = {}
        self.hedge_ratio_expiry: pd.Timedelta = pd.Timedelta(days=1)

        self.current_prices: pd.Series = pd.Series(dtype=float)
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.max_history_bars: int = 1000

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> Union[pd.DataFrame, Dict[Tuple[str, str], pd.Series]]:
        """
        Generate trading signals from price data.

        Args:
            prices: DataFrame of asset prices with DateTimeIndex

        Returns:
            Either DataFrame with columns [Date, Pair, predicted_signal]
            or Dict mapping pairs to signal Series
        """
        pass

    def update_data(self, prices: pd.DataFrame) -> None:
        """
        Update price history with length limit and validation.

        Args:
            prices: New price data
        """
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Prices must be a pandas DataFrame")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Price data must have DateTimeIndex")

        if prices.isnull().any().any():
            logger.warning("Missing values detected in price data")
            prices = prices.ffill()

        self.price_history = pd.concat([self.price_history, prices])
        if len(self.price_history) > self.max_history_bars:
            self.price_history = self.price_history.iloc[-self.max_history_bars:]

        self.current_prices = prices.iloc[-1]

    def calculate_hedge_ratio(self,
                            pair: Tuple[str, str],
                            prices: pd.DataFrame,
                            window: int = 63) -> float:
        """
        Calculate hedge ratio using cached results when possible.

        Args:
            pair: Asset pair
            prices: Price data
            window: Lookback window
        """
        current_time = prices.index[-1]

        if pair in self.hedge_ratio_cache:
            ratio, timestamp = self.hedge_ratio_cache[pair]
            if current_time - timestamp <= self.hedge_ratio_expiry:
                return ratio

        try:
            asset1, asset2 = pair
            recent_prices = prices.iloc[-window:]

            if recent_prices[asset1].std() == 0 or recent_prices[asset2].std() == 0:
                logger.warning(f"Zero variance detected in prices for {pair}")
                return 1.0

            X = sm.add_constant(recent_prices[asset2])
            model = sm.OLS(recent_prices[asset1], X).fit()

            ratio = model.params[1]

            if not np.isfinite(ratio) or abs(ratio) > 10:
                logger.warning(f"Suspicious hedge ratio {ratio} for {pair}")
                return 1.0

            self.hedge_ratio_cache[pair] = (ratio, current_time)

            return ratio

        except Exception as e:
            logger.error(f"Error calculating hedge ratio for {pair}: {str(e)}")
            return 1.0

    def calculate_position_sizes(self,
                               capital: float,
                               prices: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calculate position sizes with risk limits.

        Args:
            capital: Available capital
            prices: Current price data
        """
        if capital <= 0:
            raise ValueError("Capital must be positive")

        n_pairs = len(self.pairs)
        if n_pairs == 0:
            return {}

        capital_per_pair = capital / n_pairs
        position_sizes = {}

        total_exposure = 0

        for pair in self.pairs:
            asset1, asset2 = pair
            try:
                price1 = prices[asset1].iloc[-1]
                price2 = prices[asset2].iloc[-1]

                if not (np.isfinite(price1) and np.isfinite(price2) and price1 > 0 and price2 > 0):
                    logger.warning(f"Invalid prices for {pair}: {price1}, {price2}")
                    continue

                max_price = max(price1, price2)
                position_size = min(
                    capital_per_pair / max_price,
                    self.max_position_size
                )

                exposure = position_size * max_price / capital
                if total_exposure + exposure > self.max_portfolio_exposure:
                    position_size *= (self.max_portfolio_exposure - total_exposure) / exposure
                    total_exposure = self.max_portfolio_exposure
                else:
                    total_exposure += exposure

                position_sizes[pair] = position_size

            except Exception as e:
                logger.error(f"Error calculating position size for {pair}: {str(e)}")
                continue

        return position_sizes

    def calculate_pair_spread(self,
                            pair: Tuple[str, str],
                            prices: Optional[pd.DataFrame] = None,
                            hedge_ratio: Optional[float] = None) -> pd.Series:
        """
        Calculate price spread for a pair.

        Args:
            pair: Asset pair
            prices: Optional price data (uses price_history if not provided)
            hedge_ratio: Optional fixed hedge ratio (calculates if not provided)
        """
        if prices is None:
            prices = self.price_history

        asset1, asset2 = pair
        price1 = prices[asset1]
        price2 = prices[asset2]

        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(pair, prices)

        return price1 - hedge_ratio * price2

    def validate_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Validate and filter pairs based on available price data.

        Args:
            prices: Price data
        """
        valid_pairs = []
        for pair in self.pairs:
            asset1, asset2 = pair
            if asset1 in prices.columns and asset2 in prices.columns:
                valid_pairs.append(pair)
            else:
                logger.warning(f"Pair {pair} not found in price data")
        return valid_pairs

    def _check_risk_limits(self, pair: Tuple[str, str]) -> None:
        """Check stop loss, take profit, and drawdown levels."""
        if pair not in self.positions:
            return

        position = self.positions[pair]
        if position.stop_loss is None and position.take_profit is None and position.max_drawdown is None:
            return

        asset1, asset2 = pair
        current_spread = self.calculate_pair_spread(pair)
        entry_spread = position.entry_price1 - position.hedge_ratio * position.entry_price2

        pnl = position.quantity * (current_spread - entry_spread)

        position.peak_pnl = max(position.peak_pnl, pnl)
        position.current_drawdown = (position.peak_pnl - pnl) / abs(position.peak_pnl) if position.peak_pnl != 0 else 0

        if position.max_drawdown is not None and position.current_drawdown > position.max_drawdown:
            self._close_position(pd.Timestamp.now(), pair)
            return

        if position.stop_loss is not None:
            if position.quantity > 0 and current_spread <= entry_spread - position.stop_loss:
                self._close_position(pd.Timestamp.now(), pair)
                return
            elif position.quantity < 0 and current_spread >= entry_spread + position.stop_loss:
                self._close_position(pd.Timestamp.now(), pair)
                return

        if position.take_profit is not None:
            if position.quantity > 0 and current_spread >= entry_spread + position.take_profit:
                self._close_position(pd.Timestamp.now(), pair)
            elif position.quantity < 0 and current_spread <= entry_spread - position.take_profit:
                self._close_position(pd.Timestamp.now(), pair)

    def _open_position(self,
                      date: pd.Timestamp,
                      pair: Tuple[str, str],
                      signal: float,
                      quantity: float) -> None:
        """Open new position with transaction costs."""
        asset1, asset2 = pair
        price1 = self.current_prices[asset1]
        price2 = self.current_prices[asset2]

        hedge_ratio = self.calculate_hedge_ratio(pair, self.price_history)

        trade_value = abs(quantity) * (price1 + hedge_ratio * price2)
        transaction_cost = trade_value * self.transaction_cost_pct

        position = Position(
            quantity=quantity * signal,
            entry_price1=price1,
            entry_price2=price2,
            entry_date=date,
            hedge_ratio=hedge_ratio,
            transaction_costs=transaction_cost
        )

        position.entry_sizes.append((quantity * signal, price1))

        self.positions[pair] = position

        self.trades.append({
            'Date': date,
            'Pair': f"{asset1}/{asset2}",
            'Action': 'ENTRY',
            'Signal': signal,
            'Price1': price1,
            'Price2': price2,
            'Quantity': quantity,
            'HedgeRatio': hedge_ratio,
            'TransactionCost': transaction_cost
        })

    def _close_position(self,
                       date: pd.Timestamp,
                       pair: Tuple[str, str]) -> None:
        """Close existing position with transaction costs."""
        if pair not in self.positions:
            return

        position = self.positions[pair]
        asset1, asset2 = pair
        price1 = self.current_prices[asset1]
        price2 = self.current_prices[asset2]

        spread_entry = position.entry_price1 - position.hedge_ratio * position.entry_price2
        spread_exit = price1 - position.hedge_ratio * price2
        gross_pnl = position.quantity * (spread_exit - spread_entry)

        trade_value = abs(position.quantity) * (price1 + position.hedge_ratio * price2)
        exit_transaction_cost = trade_value * self.transaction_cost_pct

        net_pnl = gross_pnl - position.transaction_costs - exit_transaction_cost

        self.trades.append({
            'Date': date,
            'Pair': f"{asset1}/{asset2}",
            'Action': 'EXIT',
            'Price1': price1,
            'Price2': price2,
            'Quantity': -position.quantity,
            'GrossPnL': gross_pnl,
            'TransactionCosts': position.transaction_costs + exit_transaction_cost,
            'NetPnL': net_pnl,
            'HoldingPeriod': (date - position.entry_date).days,
            'MaxDrawdown': position.current_drawdown
        })

        del self.positions[pair]

    def calculate_portfolio_stats(self) -> Dict:
        """Calculate enhanced portfolio statistics."""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)
        exit_trades = trades_df[trades_df['Action'] == 'EXIT']

        stats = {
            'total_trades': len(trades_df[trades_df['Action'] == 'ENTRY']),
            'winning_trades': len(exit_trades[exit_trades['NetPnL'] > 0]),
            'gross_pnl': exit_trades['GrossPnL'].sum(),
            'total_transaction_costs': exit_trades['TransactionCosts'].sum(),
            'net_pnl': exit_trades['NetPnL'].sum(),
            'average_holding_period': exit_trades['HoldingPeriod'].mean(),
            'win_rate': len(exit_trades[exit_trades['NetPnL'] > 0]) / len(exit_trades),
            'max_drawdown': exit_trades['MaxDrawdown'].max(),
            'sharpe_ratio': exit_trades['NetPnL'].mean() / exit_trades['NetPnL'].std()
                if len(exit_trades) > 1 and exit_trades['NetPnL'].std() > 0 else 0,
            'profit_factor': abs(exit_trades[exit_trades['NetPnL'] > 0]['NetPnL'].sum() /
                               exit_trades[exit_trades['NetPnL'] < 0]['NetPnL'].sum())
                if len(exit_trades[exit_trades['NetPnL'] < 0]) > 0 else float('inf')
        }

        return stats