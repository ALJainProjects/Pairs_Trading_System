import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import traceback
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression

from config.logging_config import logger
from src.strategy.backtest import MultiPairBackTester
from src.strategy.dynamic_pairs_strategy import DynamicPairTradingSystem
from src.strategy.pairs_strategy_integrated import IntegratedPairsStrategy, create_strategy_dashboard
from src.strategy.risk import PairRiskManager
from src.strategy.pairs_strategy_SL import EnhancedStatPairsStrategy
from src.strategy.pairs_strategy_ML import MLPairsStrategy
from src.strategy.pairs_strategy_DL import PairsTradingDL
from streamlit_system.components.session_state_management import SessionStateManager


class PairModel:
    """Single pair trading model managing one pair's strategy"""

    def __init__(
            self,
            pair: Tuple[str, str],
            prices: pd.DataFrame,
            initial_capital: float = 100000,
            window_size: int = 90,  # Single window for both regression and spread
            threshold: float = 2.0,
            transaction_cost_bps: float = 1,
            stop_loss_pct: float = 0.05,
            capital_utilization: float = 0.8,
            max_holding_period: int = 30,  # Maximum days to hold a position
            profit_target_pct: float = 0.05,  # Target profit to exit
            loss_limit_pct: float = 0.03  # Tighter loss limit than stop loss
    ):
        """
        Initialize a trading model for a single pair.
        """
        # Store pair information
        self.pair = pair
        self.symbol_x, self.symbol_y = pair

        # Ensure both symbols exist in the dataframe
        if self.symbol_x not in prices.columns or self.symbol_y not in prices.columns:
            raise ValueError(f"One or both symbols in pair {pair} not found in price data")

        # Filter prices for just this pair and handle any missing values
        self.data = prices[[self.symbol_x, self.symbol_y]].copy()
        self.data = self.data.ffill().bfill()

        # Make sure both columns are numeric and positive
        for symbol in [self.symbol_x, self.symbol_y]:
            self.data[symbol] = pd.to_numeric(self.data[symbol], errors='coerce')
            # Replace any remaining NaN or non-positive values with small positive values
            self.data[symbol] = self.data[symbol].replace(0, np.nan).fillna(0.01)
            self.data[symbol] = self.data[symbol].apply(lambda x: max(x, 0.01))

        # Parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.window_size = window_size  # Single window size
        self.base_threshold = threshold
        self.entry_threshold = threshold  # Dynamic threshold that will be adjusted
        self.exit_threshold_factor = 0.5  # How much of the threshold to use for exit
        self.transaction_cost_bps = transaction_cost_bps
        self.stop_loss_pct = stop_loss_pct
        self.capital_utilization = capital_utilization
        self.max_holding_period = max_holding_period
        self.profit_target_pct = profit_target_pct
        self.loss_limit_pct = loss_limit_pct

        # Regression parameters storage
        self.beta = None
        self.alpha = None
        self.last_regression_update = None

        # Volatility tracking
        self.historical_spread_std = None
        self.volatility_history = []

        # Position tracking
        self.positions = {self.symbol_x: 0, self.symbol_y: 0}
        self.entry_prices = {self.symbol_x: None, self.symbol_y: None}
        self.entry_date = None

        # Activity tracking
        self.active = False  # Is this pair currently being traded
        self.portfolio_history = []
        self.trade_history = []
        self.spread_history = []
        self.total_transaction_costs = 0

        # Performance tracking for allocation decisions
        self.recent_trades = []  # Store most recent trade results

        # Debug counters
        self.regression_updates = 0
        self.nan_errors = 0

    def track_drawdown(self):
        """Calculate and track drawdown for this specific pair"""
        if len(self.portfolio_history) > 1:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df.set_index('date', inplace=True)

            # Calculate current drawdown
            peak = portfolio_df['portfolio_value'].cummax()
            drawdown = (peak - portfolio_df['portfolio_value']) / peak
            current_drawdown = drawdown.iloc[-1]
            max_drawdown = drawdown.max()

            # Store for allocation decisions
            self.current_drawdown = current_drawdown
            self.max_drawdown = max_drawdown

            return current_drawdown, max_drawdown
        return 0, 0

    def calculate_regression_parameters(self, prices_x: pd.Series, prices_y: pd.Series) -> Tuple[float, float]:
        """
        Calculate the regression parameters (β, α) from price series

        This uses log prices to estimate the relationship between the two assets.
        """
        try:
            # Safety check for data length
            if len(prices_x) < 100 or len(prices_y) < 100:
                if self.beta is None:
                    # If we don't have existing regression parameters, use defaults
                    return 1.0, 0.0
                else:
                    # If we already have parameters, keep using them
                    return self.beta, self.alpha

            # Ensure positive prices for log operations
            prices_x_adj = prices_x.copy().apply(lambda x: max(x, 0.01))
            prices_y_adj = prices_y.copy().apply(lambda x: max(x, 0.01))

            # Calculate log prices
            log_prices_x = np.log(prices_x_adj)
            log_prices_y = np.log(prices_y_adj)

            # Regression using log prices
            regr = LinearRegression()
            X = log_prices_x.values.reshape(-1, 1)
            y = log_prices_y.values
            regr.fit(X, y)

            # Increment counter
            self.regression_updates += 1

            return regr.coef_[0], regr.intercept_

        except Exception as e:
            # Log error but return last known parameters or defaults
            print(f"Error in regression calculation for {self.pair}: {e}")

            if self.beta is not None:
                return self.beta, self.alpha
            else:
                return 1.0, 0.0  # Default fallback

    def calculate_spread(self, prices_x: pd.Series, prices_y: pd.Series, beta: float, alpha: float) -> np.ndarray:
        """
        Calculate the spread using existing regression parameters

        The spread is the difference between the predicted value of y and the actual value,
        using the log-linear relationship: log(y) = β*log(x) + α
        """
        try:
            prices_x = fill_missing_values(prices_x)
            prices_y = fill_missing_values(prices_y)
            # Ensure positive prices for log operations
            prices_x_adj = prices_x.copy().apply(lambda x: max(x, 0.01))
            prices_y_adj = prices_y.copy().apply(lambda x: max(x, 0.01))

            # Calculate log prices
            log_prices_x = np.log(prices_x_adj)
            log_prices_y = np.log(prices_y_adj)

            # Calculate spread as deviation from estimated relationship
            spread = log_prices_y - (beta * log_prices_x + alpha)

            return spread.values

        except Exception as e:
            # Log error and return empty array
            print(f"Error in spread calculation for {self.pair}: {e}")
            self.nan_errors += 1
            return np.zeros(len(prices_x))

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss has been triggered"""
        if self.entry_prices[symbol] is None:
            return False

        if self.positions[symbol] > 0:
            return current_price < self.entry_prices[symbol] * (1 - self.stop_loss_pct)
        elif self.positions[symbol] < 0:
            return current_price > self.entry_prices[symbol] * (1 + self.stop_loss_pct)

        return False

    def calculate_position_size(self, price_x: float, price_y: float, ratio: float, spread_std: float = None):
        """
        Calculate appropriate position size based on available capital and volatility

        Args:
            price_x: Current price of symbol_x
            price_y: Current price of symbol_y
            ratio: Price ratio for hedging (typically price_y/price_x)
            spread_std: Current spread standard deviation (optional)

        Returns:
            Position size for symbol_x (symbol_y position will be this * ratio)
        """
        # Calculate total cost for a pair of positions
        unit_cost = price_x + (ratio * price_y)

        # Calculate maximum positions based on available capital with utilization limit
        max_position = int((self.current_capital * self.capital_utilization) / unit_cost)

        # Apply volatility adjustment if spread_std is provided
        if spread_std is not None and spread_std > 0:
            # Track volatility history
            if self.historical_spread_std is None:
                self.historical_spread_std = spread_std

            # Calculate volatility ratio (higher ratio = lower volatility compared to historical)
            vol_ratio = self.historical_spread_std / spread_std

            # Cap adjustment to reasonable bounds
            vol_adjustment = max(0.5, min(1.5, vol_ratio))

            # Update historical volatility with exponential smoothing
            self.historical_spread_std = 0.95 * self.historical_spread_std + 0.05 * spread_std

            # Adjust position size (smaller positions in high volatility regimes)
            max_position = int(max_position * vol_adjustment)

        # Ensure minimum position size
        position_size = max(1, max_position)

        # Double-check we can afford the position
        total_cost = position_size * price_x + (position_size * ratio) * price_y
        if total_cost > self.current_capital:
            # Reduce position size if necessary
            position_size = max(1, int(self.current_capital / unit_cost))

        return position_size

    def execute_trade(self, symbol: str, quantity: int, price: float, trade_type: str, date):
        """Execute a trade and update positions with transaction costs"""
        # Sanity check on quantity
        if quantity == 0:
            return

        # Calculate costs
        cost = quantity * price
        transaction_cost = abs(quantity) * price * (self.transaction_cost_bps / 10000)

        if trade_type == 'buy':
            total_cost = cost + transaction_cost

            # Ensure we have enough capital
            if total_cost > self.current_capital:
                # Reduce quantity if necessary
                adjusted_quantity = max(1, int((self.current_capital - transaction_cost) / price))

                # Recalculate with adjusted quantity
                cost = adjusted_quantity * price
                transaction_cost = abs(adjusted_quantity) * price * (self.transaction_cost_bps / 10000)
                total_cost = cost + transaction_cost
                quantity = adjusted_quantity

            self.current_capital -= total_cost
            self.positions[symbol] += quantity
            self.entry_prices[symbol] = price
            trade_value = -total_cost  # Negative for buys (money outflow)

        else:  # sell
            total_proceeds = cost - transaction_cost
            self.current_capital += total_proceeds
            self.positions[symbol] -= quantity
            self.entry_prices[symbol] = price
            trade_value = total_proceeds  # Positive for sells (money inflow)

        self.total_transaction_costs += transaction_cost

        # Add trade to history
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'type': trade_type,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'transaction_cost': transaction_cost,
            'trade_value': trade_value,
            'trade_direction': 1 if trade_type == 'buy' else -1,
            'capital_after': self.current_capital
        })

    def close_position(self, current_prices: Dict[str, float], date, reason="manual"):
        """Close all positions for this pair and record the reason"""
        if not self.active:
            return

        # Record pre-close data for performance tracking
        entry_value = 0
        exit_value = 0

        for symbol, position in self.positions.items():
            if position != 0:
                quantity = abs(position)
                trade_type = 'buy' if position < 0 else 'sell'

                # Track entry and exit values for P&L calculation
                if position > 0:  # Long position
                    entry_value += quantity * self.entry_prices[symbol]
                    exit_value += quantity * current_prices[symbol]
                else:  # Short position
                    entry_value -= quantity * self.entry_prices[symbol]
                    exit_value -= quantity * current_prices[symbol]

                # Execute the closing trade
                self.execute_trade(symbol, quantity, current_prices[symbol], trade_type, date)

        # Calculate trade result
        trade_pnl = exit_value - entry_value - self.total_transaction_costs
        trade_result = {
            'entry_date': self.entry_date,
            'exit_date': date,
            'pnl': trade_pnl,
            'holding_period': (date - self.entry_date).days if self.entry_date else 0,
            'reason': reason
        }

        # Record for performance-based allocation
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > 10:
            self.recent_trades.pop(0)

        self.active = False
        self.entry_date = None

    def update_thresholds(self, spread_std: float):
        """Update entry and exit thresholds based on recent spread volatility"""
        # Track volatility regime
        self.volatility_history.append(spread_std)
        if len(self.volatility_history) > 20:
            self.volatility_history.pop(0)

        # Compare current volatility to recent average
        if len(self.volatility_history) > 5:  # Need some history
            avg_vol = sum(self.volatility_history) / len(self.volatility_history)
            vol_ratio = spread_std / avg_vol if avg_vol > 0 else 1.0

            # Adjust thresholds: higher in high volatility, lower in low volatility
            if vol_ratio > 1.2:  # High volatility regime
                self.entry_threshold = self.base_threshold * 1.2
                self.exit_threshold_factor = 0.6  # Wait longer to exit
            elif vol_ratio < 0.8:  # Low volatility regime
                self.entry_threshold = self.base_threshold * 0.8
                self.exit_threshold_factor = 0.4  # Exit faster
            else:  # Normal regime
                self.entry_threshold = self.base_threshold
                self.exit_threshold_factor = 0.5
        else:
            # Not enough history, use base values
            self.entry_threshold = self.base_threshold
            self.exit_threshold_factor = 0.5

    def update(self, date, current_prices: Dict[str, float]):
        """
        Update the pair model for the current date

        Args:
            date: Current date
            current_prices: Dictionary of current prices for all symbols
        """
        # Skip if we don't have prices for both symbols
        if self.symbol_x not in current_prices or self.symbol_y not in current_prices:
            return

        # Check time-based and profit-based exit conditions if position is active
        if self.active and self.entry_date is not None:
            # Calculate days in trade
            days_in_trade = (date - self.entry_date).days

            # Calculate current P&L
            entry_value = sum(abs(self.positions[symbol]) * self.entry_prices[symbol]
                              for symbol in self.positions.keys() if self.positions[symbol] != 0)
            current_value = sum(abs(self.positions[symbol]) * current_prices[symbol]
                                for symbol in self.positions.keys() if self.positions[symbol] != 0)

            profit_pct = (current_value - entry_value) / entry_value if entry_value > 0 else 0

            # Exit rules
            if days_in_trade > self.max_holding_period:
                self.close_position(current_prices, date, "max holding period")
                return
            elif profit_pct > self.profit_target_pct:
                self.close_position(current_prices, date, "profit target")
                return
            elif profit_pct < -self.loss_limit_pct:
                self.close_position(current_prices, date, "loss limit")
                return

        # Check stop losses for active positions
        for symbol, position in self.positions.items():
            if position != 0 and self.check_stop_loss(symbol, current_prices[symbol]):
                self.close_position(current_prices, date, "stop loss")
                return

        # Get data index for the current date
        try:
            current_idx = self.data.index.get_loc(date)
        except KeyError:
            # Date not in index, skip update
            return

        # Use a single window for both regression parameters and spread statistics
        window_start_idx = max(0, current_idx - self.window_size)
        if current_idx <= window_start_idx:
            # Not enough data yet
            return

        window = self.data.iloc[window_start_idx:current_idx]

        # Calculate regression parameters using this window
        self.beta, self.alpha = self.calculate_regression_parameters(
            window[self.symbol_x],
            window[self.symbol_y]
        )

        if self.beta is None or self.alpha is None:
            # Regression failed
            return

        # Calculate historical spreads using the same window and parameters
        historical_spreads = self.calculate_spread(
            window[self.symbol_x],
            window[self.symbol_y],
            self.beta,
            self.alpha
        )

        # Calculate spread statistics with safeguards for NaN and Inf values
        if len(historical_spreads) > 0:
            # Filter out any potential NaN or Inf values
            valid_spreads = historical_spreads[np.isfinite(historical_spreads)]

            if len(valid_spreads) > 0:
                mean_spread = np.mean(valid_spreads)
                std_spread = np.std(valid_spreads)
            else:
                # Fallback if no valid spreads
                mean_spread = 0
                std_spread = 0.01  # Avoid division by zero
        else:
            # Fallback if no historical spreads
            mean_spread = 0
            std_spread = 0.01

        # Update thresholds based on recent volatility
        self.update_thresholds(std_spread)

        # Calculate current spread
        try:
            current_price_x = current_prices[self.symbol_x]
            current_price_y = current_prices[self.symbol_y]

            # Ensure positive values for log operation
            current_price_x = max(current_price_x, 0.01)
            current_price_y = max(current_price_y, 0.01)

            # Calculate current spread using same method as historical spreads
            current_spread = np.log(current_price_y) - (self.beta * np.log(current_price_x) + self.alpha)

            # Store spread information
            self.spread_history.append({
                'date': date,
                'spread': current_spread,
                'upper_band': mean_spread + self.entry_threshold * std_spread,
                'lower_band': mean_spread - self.entry_threshold * std_spread,
                'mean': mean_spread,
                'std': std_spread,
                'beta': self.beta,
                'alpha': self.alpha,
                'price_x': current_price_x,
                'price_y': current_price_y,
                'entry_threshold': self.entry_threshold,
                'exit_threshold_factor': self.exit_threshold_factor
            })

            # Trading logic with dynamic thresholds
            price_x = current_price_x
            price_y = current_price_y

            # Calculate hedge ratio based on current prices and beta
            # This ensures dollar-neutral positions
            hedge_ratio = (price_y / price_x) * self.beta

            # Calculate upper and lower bands with dynamic thresholds
            upper_band = mean_spread + self.entry_threshold * std_spread
            lower_band = mean_spread - self.entry_threshold * std_spread
            exit_band_width = self.exit_threshold_factor * self.entry_threshold * std_spread

            if current_spread > upper_band:
                # Short Y, Long X
                if not self.active and self.positions[self.symbol_x] == 0 and self.positions[self.symbol_y] == 0:
                    # Calculate position size based on volatility-adjusted capital
                    position_size = self.calculate_position_size(price_x, price_y, hedge_ratio, std_spread)

                    if position_size > 0:  # Ensure we have enough capital
                        # Short Y first (to ensure we have capital for both legs)
                        y_quantity = int(position_size * hedge_ratio)
                        if y_quantity > 0:  # Ensure valid quantity
                            self.execute_trade(self.symbol_y, -y_quantity, price_y, 'sell', date)
                            self.execute_trade(self.symbol_x, position_size, price_x, 'buy', date)
                            self.active = True
                            self.entry_date = date

            elif current_spread < lower_band:
                # Long Y, Short X
                if not self.active and self.positions[self.symbol_x] == 0 and self.positions[self.symbol_y] == 0:
                    # Calculate position size based on volatility-adjusted capital
                    position_size = self.calculate_position_size(price_x, price_y, hedge_ratio, std_spread)

                    if position_size > 0:  # Ensure we have enough capital
                        # Short X first (to ensure we have capital for both legs)
                        self.execute_trade(self.symbol_x, -position_size, price_x, 'sell', date)

                        # Calculate Y quantity based on hedge ratio
                        y_quantity = int(position_size * hedge_ratio)
                        if y_quantity > 0:  # Ensure valid quantity
                            self.execute_trade(self.symbol_y, y_quantity, price_y, 'buy', date)
                            self.active = True
                            self.entry_date = date
                        else:
                            # If Y quantity is invalid, reverse the X position
                            self.execute_trade(self.symbol_x, position_size, price_x, 'buy', date)

            elif self.active and abs(current_spread - mean_spread) < exit_band_width:
                # Mean reversion achieved, close position with adaptive exit band
                self.close_position(current_prices, date, "mean reversion")

            # Record portfolio value
            if self.active:
                # If we have open positions, calculate portfolio value including positions
                portfolio_value = self.current_capital + sum(
                    self.positions[symbol] * current_prices[symbol]
                    for symbol in self.positions.keys()
                )
            else:
                # If no open positions, portfolio value is just cash
                portfolio_value = self.current_capital

            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'active': self.active,
                'cash': self.current_capital,
                'position_value': portfolio_value - self.current_capital
            })

        except Exception as e:
            print(f"Error in update for {self.pair} on {date}: {e}")

    def get_recent_metrics(self):
        """Calculate metrics based on recent trades for allocation decisions"""
        if not self.recent_trades:
            return {
                'win_rate': 50,
                'profit_factor': 1.0,
                'avg_holding_period': 10,
                'avg_pnl': 0
            }

        # Calculate win rate
        winning_trades = [t for t in self.recent_trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.recent_trades) * 100

        # Calculate profit factor with safeguards
        winning_sum = sum(t['pnl'] for t in winning_trades)
        losing_sum = sum(t['pnl'] for t in self.recent_trades if t['pnl'] < 0)

        if losing_sum < 0:
            profit_factor = abs(winning_sum / losing_sum) if winning_sum > 0 else 0
        else:
            profit_factor = 2.0 if winning_sum > 0 else 0.5  # Default values

        # Calculate average holding period
        avg_holding = sum(t['holding_period'] for t in self.recent_trades) / len(self.recent_trades)

        # Calculate average P&L
        avg_pnl = sum(t['pnl'] for t in self.recent_trades) / len(self.recent_trades)

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding,
            'avg_pnl': avg_pnl
        }

    def get_metrics(self) -> Dict:
        """Calculate performance metrics for this pair"""
        if not self.portfolio_history:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        # Calculate returns and drawdown
        returns = portfolio_df['portfolio_value'].pct_change()
        returns = returns.dropna()
        drawdown = ((portfolio_df['portfolio_value'].cummax() - portfolio_df['portfolio_value']) /
                    portfolio_df['portfolio_value'].cummax())

        # Process trade history to calculate trade metrics
        if not self.trade_history:
            return {
                'total_return': (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': drawdown.max() * 100 if not drawdown.empty else 0,
                'sharpe_ratio': (returns.mean() * 4 / returns.std()) * np.sqrt(252) if len(
                    returns) > 0 and returns.std() > 0 else 0,
                'regression_updates': self.regression_updates,
                'nan_errors': self.nan_errors
            }

        # Convert trade history to DataFrame
        trade_df = pd.DataFrame(self.trade_history)

        # Calculate trade-level metrics with proper PnL
        trade_df = trade_df.sort_values('date')

        # Group trades by cycle (open/close position)
        trade_df['position_change'] = trade_df['trade_direction'].rolling(window=2).sum().fillna(0)
        trade_df['cycle_id'] = (trade_df['position_change'] == 0).cumsum()

        # Calculate PnL for each complete trade cycle
        cycle_pnl = trade_df.groupby('cycle_id')['trade_value'].sum()

        # Make sure we have at least one complete cycle
        if len(cycle_pnl) == 0:
            return {
                'total_return': (portfolio_df['portfolio_value'].iloc[
                                     -1] + cycle_pnl.sum() / self.initial_capital - 1) * 100,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': drawdown.max() * 100 if not drawdown.empty else 0,
                'sharpe_ratio': (returns.mean() * 4 / returns.std()) * np.sqrt(252) if len(
                    returns) > 0 and returns.std() > 0 else 0,
                'regression_updates': self.regression_updates,
                'nan_errors': self.nan_errors
            }

        # Identify winning and losing trades
        winning_trades = cycle_pnl[cycle_pnl > 0]
        losing_trades = cycle_pnl[cycle_pnl < 0]

        # Calculate metrics
        metrics = {
            'total_return': (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100,
            'num_trades': len(cycle_pnl),
            'win_rate': len(winning_trades) / len(cycle_pnl) * 100 if len(cycle_pnl) > 0 else 0,
            'profit_factor': abs(winning_trades.sum() / losing_trades.sum()) if len(
                losing_trades) > 0 and losing_trades.sum() != 0 else 0,
            'max_drawdown': drawdown.max() * 100 if not drawdown.empty else 0,
            'sharpe_ratio': (returns.mean() * 4 / returns.std()) * np.sqrt(252) if len(
                returns) > 0 and returns.std() > 0 else 0,
            'total_pnl': cycle_pnl.sum(),
            'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'best_trade': cycle_pnl.max() if len(cycle_pnl) > 0 else 0,
            'worst_trade': cycle_pnl.min() if len(cycle_pnl) > 0 else 0,
            'total_transaction_costs': self.total_transaction_costs,
            'regression_updates': self.regression_updates,
            'nan_errors': self.nan_errors
        }

        return metrics

    def plot_pair_analysis(self):
        """Create detailed visualization for this pair"""
        if not self.portfolio_history:
            print(f"No trading history for {self.symbol_x}-{self.symbol_y}")
            return

        # Convert history data to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        spread_df = pd.DataFrame(self.spread_history)
        spread_df.set_index('date', inplace=True)

        # Create subplots
        fig = make_subplots(
            rows=5, cols=1,  # Increased to 5 rows
            subplot_titles=(
                f'Price Series: {self.symbol_x} vs {self.symbol_y}',
                'Normalized Prices',
                'Pair Model Equity Curve',
                'Spread with Bands',
                'Trade Points and P&L'
            ),
            vertical_spacing=0.08,
            row_heights=[0.2, 0.15, 0.15, 0.2, 0.2]
        )

        # 1. Price Series
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data[self.symbol_x],
                name=self.symbol_x,
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data[self.symbol_y],
                name=self.symbol_y,
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # 2. Normalized Prices
        # Normalize from the first date
        normalized_x = self.data[self.symbol_x] / self.data[self.symbol_x].iloc[0]
        normalized_y = self.data[self.symbol_y] / self.data[self.symbol_y].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=normalized_x,
                name=f"{self.symbol_x} (Normalized)",
                line=dict(color='blue', dash='dash')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=normalized_y,
                name=f"{self.symbol_y} (Normalized)",
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )

        # 3. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='green')
            ),
            row=3, col=1
        )

        # Add cash component as separate line
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['cash'],
                name='Cash',
                line=dict(color='lightgreen', dash='dot')
            ),
            row=3, col=1
        )

        # 4. Spread with Bands
        if not spread_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['spread'],
                    name='Spread',
                    line=dict(color='purple')
                ),
                row=4, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['upper_band'],
                    name='Upper Band',
                    line=dict(color='gray', dash='dash')
                ),
                row=4, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['lower_band'],
                    name='Lower Band',
                    line=dict(color='gray', dash='dash')
                ),
                row=4, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['mean'],
                    name='Mean',
                    line=dict(color='black', dash='dot')
                ),
                row=4, col=1
            )

        # 5. Trade Points and P&L
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)

            # Process trade data for visualization
            trade_df = trade_df.sort_values('date')

            # Group by trade cycle
            trade_df['position_change'] = trade_df['trade_direction'].rolling(window=2).sum().fillna(0)
            trade_df['cycle_id'] = (trade_df['position_change'] == 0).cumsum()

            # Calculate PnL for each cycle and cumulative PnL
            cycle_pnl = trade_df.groupby('cycle_id')['trade_value'].sum()
            cycle_dates = trade_df.groupby('cycle_id')['date'].max()

            # Create cumulative PnL
            if len(cycle_pnl) > 0:
                cumulative_pnl = cycle_pnl.cumsum()

                # Create a DataFrame for plotting
                cycle_df = pd.DataFrame({
                    'date': cycle_dates,
                    'pnl': cumulative_pnl
                })

                if not cycle_df.empty:
                    # Plot cumulative P&L
                    fig.add_trace(
                        go.Scatter(
                            x=cycle_df['date'],
                            y=cycle_df['pnl'],
                            name='Cumulative P&L',
                            line=dict(color='orange')
                        ),
                        row=5, col=1
                    )

                    # Add trade markers on price chart
                    for _, trade in trade_df.iterrows():
                        marker_color = 'green' if trade['type'] == 'buy' else 'red'
                        marker_symbol = 'triangle-up' if trade['type'] == 'buy' else 'triangle-down'

                        fig.add_trace(
                            go.Scatter(
                                x=[trade['date']],
                                y=[trade['price']],
                                mode='markers',
                                marker=dict(
                                    symbol=marker_symbol,
                                    color=marker_color,
                                    size=10
                                ),
                                name=f"{trade['type'].capitalize()} {trade['symbol']}",
                                showlegend=False
                            ),
                            row=1, col=1
                        )

        # Update layout
        fig.update_layout(
            height=1500,  # Increased height for 5 rows
            width=1000,
            showlegend=True,
            title_text=f"Pair Analysis: {self.symbol_x}-{self.symbol_y}"
        )

        # Update axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Normalized Price", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
        fig.update_yaxes(title_text="Spread", row=4, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=5, col=1)

        return fig


class MultiPairTradingSystem:
    """Manager for multiple pair trading models"""

    def __init__(
            self,
            pairs: List[Tuple[str, str]],
            prices: pd.DataFrame,
            initial_capital: float = 1500000,  # $1.5M split equally
            window_size: int = 90,  # Single window for consistency
            threshold: float = 2.0,
            transaction_cost_bps: float = 1,
            stop_loss_pct: float = 0.05,
            capital_utilization: float = 0.8,
            max_holding_period: int = 30,  # Maximum days to hold a position
            profit_target_pct: float = 0.05,  # Target profit to exit
            loss_limit_pct: float = 0.03,  # Tighter loss limit than stop loss
            capital_reallocation_freq: int = 60  # Reallocate capital every 60 days
    ):
        """
        Initialize the multi-pair trading system with the updated parameter structure
        """
        self.pairs = pairs
        self.prices = prices.copy()
        for col in self.prices.columns:
            self.prices[col] = fill_missing_values(self.prices[col])
        self.capital_reallocation_freq = capital_reallocation_freq
        self.last_reallocation_date = None

        # Clean the price data
        # Replace zeros or negative values with small positive values
        for col in self.prices.columns:
            self.prices[col] = self.prices[col].apply(lambda x: max(x, 0.01) if pd.notnull(x) else 0.01)

        # Validate the pairs
        valid_pairs = []
        for pair in pairs:
            if pair[0] in prices.columns and pair[1] in prices.columns:
                valid_pairs.append(pair)
            else:
                print(f"Warning: Pair {pair} contains symbols not found in the data. Skipping.")

        self.pairs = valid_pairs
        if not self.pairs:
            raise ValueError("No valid pairs found in the provided data")

        # Calculate capital per pair
        self.initial_capital = initial_capital
        self.capital_per_pair = initial_capital / len(self.pairs)

        # Create separate models for each pair with the CORRECT parameter structure
        self.pair_models = {}
        for pair in self.pairs:
            self.pair_models[pair] = PairModel(
                pair=pair,
                prices=self.prices,
                initial_capital=self.capital_per_pair,
                window_size=window_size,  # CORRECT parameter
                threshold=threshold,
                transaction_cost_bps=transaction_cost_bps,
                stop_loss_pct=stop_loss_pct,
                capital_utilization=capital_utilization,
                max_holding_period=max_holding_period,  # New parameter
                profit_target_pct=profit_target_pct,  # New parameter
                loss_limit_pct=loss_limit_pct  # New parameter
            )

        # Track overall portfolio performance
        self.portfolio_history = []

        # Debug statistics
        self.nan_count = 0
        self.processing_errors = 0

    @property
    def trade_history(self):
        """Aggregate trade history from all pair models into a single list"""
        all_trades = []
        for pair, model in self.pair_models.items():
            for trade in model.trade_history:
                trade_copy = dict(trade)
                trade_copy['pair_id'] = pair
                all_trades.append(trade_copy)
        return all_trades

    @property
    def data(self):
        """Alias for prices to maintain compatibility with optimizer"""
        return self.prices

    def get_pair_model_params(self):
        """Get current parameters from a representative pair model"""
        if not self.pairs or not self.pair_models:
            return {}
        # Use first pair model as representative
        model = self.pair_models[self.pairs[0]]
        return {
            'threshold': model.entry_threshold,
            'stop_loss_pct': model.stop_loss_pct,
            'profit_target_pct': model.profit_target_pct,
            'loss_limit_pct': model.loss_limit_pct,
            'max_holding_period': model.max_holding_period
        }

    def update_all_model_params(self, new_params: Dict):
        """Update parameters for all pair models"""
        for model in self.pair_models.values():
            for param, value in new_params.items():
                if hasattr(model, param):
                    setattr(model, param, value)
                    # Special handling for threshold
                    if param == 'threshold':
                        model.base_threshold = value
                        model.entry_threshold = value

    def apply_portfolio_weights(self, weights: Dict[Tuple[str, str], float]):
        """
        Apply optimized portfolio weights by adjusting capital allocation

        Args:
            weights: Dictionary mapping pair tuples to their target weights
        """
        if not weights:
            return

        # Calculate new capital allocations
        total_capital = sum(model.current_capital for model in self.pair_models.values())

        for pair, weight in weights.items():
            if pair in self.pair_models:
                model = self.pair_models[pair]
                # Only adjust if no active position
                if not model.active:
                    target_capital = total_capital * weight
                    # Gradual adjustment
                    model.current_capital = (0.9 * model.current_capital +
                                             0.1 * target_capital)

    def update_capital_allocation(self, current_date):
        """Reallocate capital based on recent performance metrics"""
        # Only reallocate periodically
        if (self.last_reallocation_date is not None and
                (current_date - self.last_reallocation_date).days < self.capital_reallocation_freq):
            return

        # Calculate performance scores for each pair
        performance_scores = {}
        total_score = 0

        for pair, model in self.pair_models.items():
            # Get recent metrics (from the model's recent trades)
            metrics = model.get_recent_metrics()

            # Calculate performance score based on win rate and profit factor
            win_rate = metrics.get('win_rate', 50)  # Default if no data
            profit_factor = metrics.get('profit_factor', 1.0)  # Default if no data

            # Simple scoring function (can be made more sophisticated)
            score = win_rate * profit_factor / 100

            # Ensure minimum allocation
            performance_scores[pair] = max(0.5, score)  # Minimum 50% allocation
            total_score += performance_scores[pair]

        # Reallocate capital based on performance scores
        if total_score > 0:
            for pair, score in performance_scores.items():
                # Calculate new capital allocation (normalized by total score)
                allocation_pct = score / total_score
                new_capital_target = self.initial_capital * allocation_pct

                # Update model capital (gradually to avoid abrupt changes)
                current_capital = self.pair_models[pair].current_capital

                # Only adjust if the model doesn't have an active position
                if not self.pair_models[pair].active:
                    self.pair_models[pair].current_capital = 0.9 * current_capital + 0.1 * new_capital_target

        self.last_reallocation_date = current_date

    def run_backtest(self):
        """Run the backtest for all pair models with enhanced capital allocation"""
        # Get all unique dates from the price data
        dates = self.prices.index.sort_values()

        # Progress tracking
        total_dates = len(dates)
        print(f"Running backtest over {total_dates} trading days with {len(self.pairs)} pairs")

        # Iterate through all dates
        for i, date in enumerate(dates):
            if i % 100 == 0:  # Print progress every 100 days
                print(f"Processing date {i + 1}/{total_dates}: {date.strftime('%Y-%m-%d')}")

            # Get current prices for all symbols
            current_prices = {}
            for symbol in self.prices.columns:
                try:
                    price = self.prices.loc[date, symbol]
                    if pd.notnull(price) and price > 0:
                        current_prices[symbol] = price
                    else:
                        # Use last valid price or fallback to 0.01
                        last_valid = self.prices[symbol].loc[:date].dropna()
                        if len(last_valid) > 0:
                            current_prices[symbol] = last_valid.iloc[-1]
                        else:
                            current_prices[symbol] = 0.01
                            self.nan_count += 1
                except Exception as e:
                    self.processing_errors += 1
                    continue

            # First, update capital allocation based on performance
            if i > 180:  # Allow some initial trading history to accumulate
                self.update_capital_allocation(date)

            # Update each pair model with the current data
            total_portfolio_value = 0
            active_pairs = 0

            for pair, model in self.pair_models.items():
                try:
                    model.update(date, current_prices)

                    # Get latest portfolio value from this model
                    if model.portfolio_history:
                        total_portfolio_value += model.portfolio_history[-1]['portfolio_value']
                        active_pairs += 1 if model.portfolio_history[-1]['active'] else 0
                except Exception as e:
                    print(f"Error updating model for {pair} on {date}: {e}")
                    self.processing_errors += 1

            # Record overall portfolio statistics
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': total_portfolio_value,
                'active_pairs': active_pairs
            })

        print("Backtest completed successfully")
        print(f"NaN values encountered: {self.nan_count}")
        print(f"Processing errors: {self.processing_errors}")

    def get_portfolio_metrics(self) -> Dict:
        """Calculate and return aggregated portfolio metrics with proper NaN handling"""
        if not self.portfolio_history:
            return {'error': 'No portfolio history available'}

        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        # Calculate returns with proper error handling
        if len(portfolio_df) > 1:
            # Calculate daily returns with NaN and infinity handling
            returns = portfolio_df['portfolio_value'].pct_change()
            returns = returns.replace([np.inf, -np.inf], np.nan)
            returns = returns.ffill().bfill()

            # Calculate volatility - safely
            returns_std = returns.std()
            annual_vol = returns_std * np.sqrt(252) * 100 if returns_std > 0 else 0.01

            # Calculate Sharpe - safely
            mean_return = returns.mean()

            # Calculate drawdown - safely
            peak = portfolio_df['portfolio_value'].cummax()
            drawdown = ((peak - portfolio_df['portfolio_value']) / peak) * 100
            max_dd = drawdown.max() if not drawdown.empty else 0
        else:
            # Not enough data for return calculations
            annual_return = 0
            annual_vol = 0
            sharpe = 0
            max_dd = 0

        # Get individual pair metrics
        pair_metrics = {}
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        total_transaction_costs = 0

        for pair, model in self.pair_models.items():
            metrics = model.get_metrics()
            pair_metrics[pair] = metrics

            # Aggregate statistics
            total_trades += metrics.get('num_trades', 0)
            winning_trades += metrics.get('num_trades', 0) * metrics.get('win_rate', 0) / 100
            total_pnl += metrics.get('total_pnl', 0)
            total_transaction_costs += metrics.get('total_transaction_costs', 0)

        annual_return = (((portfolio_df['portfolio_value'].iloc[-1] + total_pnl) / self.initial_capital) **
                         (252 / len(portfolio_df)) - 1) * 100
        sharpe = ((mean_return * ((total_pnl / self.initial_capital) / (annual_return / 100))) / returns_std) * np.sqrt(
            252) if returns_std > 0 else 0

        # Calculate overall metrics with safe operations
        portfolio_metrics = {
            'Portfolio Metrics': {
                'Initial Capital': self.initial_capital,
                'Final Portfolio Value': portfolio_df['portfolio_value'].iloc[
                                             -1] + total_pnl if not portfolio_df.empty else self.initial_capital,
                'Total Return (%)': ((portfolio_df['portfolio_value'].iloc[
                                          -1] + total_pnl) / self.initial_capital - 1) * 100 if not portfolio_df.empty else 0,
                'Annual Return (%)': annual_return,
                'Annual Volatility (%)': annual_vol,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_dd,
                'Total Transaction Costs': total_transaction_costs,
                'Total PnL': total_pnl
            },
            'Trading Activity': {
                'Total Number of Trades': total_trades,
                'Average Win Rate (%)': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'Average Trades Per Day': total_trades / len(portfolio_df) if len(portfolio_df) > 0 else 0,
                'Average Active Pairs': portfolio_df['active_pairs'].mean() if not portfolio_df.empty else 0,
                'Max Active Pairs': portfolio_df['active_pairs'].max() if not portfolio_df.empty else 0,
            },
            'Data Quality': {  # Add this section
                'Total NaN Values': self.nan_count,
                'Total Processing Errors': self.processing_errors,
                'Total Regression Updates': sum(model.regression_updates for model in self.pair_models.values()),
                'Total NaN Errors in Spread Calc': sum(model.nan_errors for model in self.pair_models.values())
            },
            'Pair-Specific Metrics': pair_metrics
        }

        return portfolio_metrics

    def plot_portfolio_overview(self):
        """Create comprehensive portfolio performance visualization"""
        if not self.portfolio_history:
            print("No portfolio history available to plot")
            return

        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        # Calculate daily returns
        returns = portfolio_df['portfolio_value'].pct_change().dropna()

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value', 'Daily Returns',
                'Drawdown', 'Active Pairs',
                'Pair Performance', 'Pair Win Rates'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. Portfolio Value
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Add initial capital reference line
        fig.add_trace(
            go.Scatter(
                x=[portfolio_df.index[0], portfolio_df.index[-1]],
                y=[self.initial_capital, self.initial_capital],
                name='Initial Capital',
                line=dict(color='black', dash='dash')
            ),
            row=1, col=1
        )

        # 2. Daily Returns
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns * 100,  # Convert to percentage
                name='Daily Returns',
                line=dict(color='green')
            ),
            row=1, col=2
        )

        # 3. Drawdown
        drawdown = ((portfolio_df['portfolio_value'].cummax() - portfolio_df['portfolio_value']) /
                    portfolio_df['portfolio_value'].cummax()) * 100  # Convert to percentage
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=drawdown,
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        # 4. Active Pairs
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['active_pairs'],
                name='Active Pairs',
                line=dict(color='purple')
            ),
            row=2, col=2
        )

        # 5. Pair Performance - Final equity for each pair
        pair_performance = []
        for pair, model in self.pair_models.items():
            if model.portfolio_history:
                model_df = pd.DataFrame(model.portfolio_history)
                final_value = model_df['portfolio_value'].iloc[-1]
                pair_performance.append({
                    'pair': f"{pair[0]}-{pair[1]}",
                    'final_value': final_value,
                    'return_pct': (final_value / self.capital_per_pair - 1) * 100
                })

        pair_perf_df = pd.DataFrame(pair_performance)
        if not pair_perf_df.empty:
            fig.add_trace(
                go.Bar(
                    x=pair_perf_df['pair'],
                    y=pair_perf_df['return_pct'],
                    name='Return %',
                    marker_color='blue'
                ),
                row=3, col=1
            )

        # 6. Pair Win Rates
        pair_win_rates = []
        for pair, model in self.pair_models.items():
            metrics = model.get_metrics()
            if metrics:
                pair_win_rates.append({
                    'pair': f"{pair[0]}-{pair[1]}",
                    'win_rate': metrics.get('win_rate', 0),
                    'num_trades': metrics.get('num_trades', 0)
                })

        pair_win_df = pd.DataFrame(pair_win_rates)
        if not pair_win_df.empty:
            fig.add_trace(
                go.Bar(
                    x=pair_win_df['pair'],
                    y=pair_win_df['win_rate'],
                    name='Win Rate %',
                    marker_color='green',
                    text=pair_win_df['num_trades'].astype(str) + ' trades',
                    textposition='auto'
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=1200,
            width=1000,
            showlegend=True,
            title_text="Portfolio Performance Dashboard"
        )

        return fig

def find_correlated_pairs(prices_df: pd.DataFrame, correlation_threshold: float = 0.7, max_pairs: int = 10) -> List[
    Tuple[str, str]]:
    """
    Find highly correlated pairs based on returns correlation

    Args:
        prices_df: DataFrame with price data for all symbols
        correlation_threshold: Minimum correlation to consider (absolute value)
        max_pairs: Maximum number of pairs to return

    Returns:
        List of tuples containing highly correlated pairs
    """
    # Calculate returns with proper handling of NaN values
    returns_df = prices_df.pct_change().fillna(0)

    # Calculate correlation matrix
    corr_matrix = returns_df.corr()

    # Find pairs with high correlation
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            symbol1 = corr_matrix.columns[i]
            symbol2 = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]

            if abs(correlation) >= correlation_threshold:
                pairs.append((symbol1, symbol2, correlation))

    # Sort by absolute correlation (descending)
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Return top pairs
    return [(p[0], p[1]) for p in pairs[:max_pairs]]


def find_cointegrated_pairs(prices_df: pd.DataFrame, p_value_threshold: float = 0.05, max_pairs: int = 10) -> List[
    Tuple[str, str]]:
    """
    Find potentially cointegrated pairs using Augmented Dickey-Fuller test

    Args:
        prices_df: DataFrame with price data for all symbols
        p_value_threshold: Maximum p-value to consider cointegration
        max_pairs: Maximum number of pairs to return

    Returns:
        List of tuples containing potentially cointegrated pairs
    """
    from statsmodels.tsa.stattools import adfuller

    # Function to test if the spread of two securities is stationary (cointegrated)
    def is_cointegrated(s1, s2):
        try:
            # Handle zeros or negative values
            s1_adj = s1.apply(lambda x: max(x, 0.01))
            s2_adj = s2.apply(lambda x: max(x, 0.01))

            # Log prices
            s1_log = np.log(s1_adj)
            s2_log = np.log(s2_adj)

            # Align series and drop any NaNs
            s1_aligned, s2_aligned = s1_log.align(s2_log, join='inner')

            if len(s1_aligned) < 100:  # Need sufficient data
                return False, 1.0

            # Run linear regression
            X = s1_aligned.values.reshape(-1, 1)
            y = s2_aligned.values
            model = LinearRegression()
            model.fit(X, y)

            # Calculate the spread/residuals
            spread = y - model.predict(X)

            # Test for stationarity using Augmented Dickey-Fuller test
            adf_result = adfuller(spread)
            p_value = adf_result[1]

            return p_value < p_value_threshold, p_value
        except Exception as e:
            print(f"Error testing cointegration: {e}")
            return False, 1.0

    # Find cointegrated pairs
    pairs = []
    tested = set()
    symbols = prices_df.columns

    total_tests = len(symbols) * (len(symbols) - 1) // 2
    print(f"Testing {total_tests} potential pairs for cointegration...")

    count = 0
    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols):
            if i >= j or (s1, s2) in tested or (s2, s1) in tested:
                continue

            tested.add((s1, s2))
            count += 1
            if count % 100 == 0:
                print(f"Tested {count}/{total_tests} pairs...")

            try:
                # Get price series
                price1 = prices_df[s1]
                price2 = prices_df[s2]

                # Test for cointegration
                is_coint, p_value = is_cointegrated(price1, price2)

                if is_coint:
                    # Calculate correlation for additional info
                    returns1 = price1.pct_change().fillna(0)
                    returns2 = price2.pct_change().fillna(0)
                    corr = returns1.corr(returns2)

                    pairs.append((s1, s2, p_value, corr))
            except Exception as e:
                print(f"Error testing pair ({s1}, {s2}): {e}")

    # Sort by p-value (ascending)
    pairs.sort(key=lambda x: x[2])

    print(f"Found {len(pairs)} cointegrated pairs")

    # Return top pairs
    return [(p[0], p[1]) for p in pairs[:max_pairs]]


def fill_missing_values(price_series: pd.Series) -> pd.Series:
    """
    Fill missing values in a price series using interpolation and extrapolation

    Args:
        price_series: Series of price data with possible NaN values

    Returns:
        Series with NaN values filled using interpolation/extrapolation
    """
    # First check if we have any NaN values to fill
    if not price_series.isna().any():
        return price_series

    # Make a copy to avoid modifying the original
    clean_series = price_series.copy()

    # Step 1: Interpolate missing values (for NaNs between known values)
    # This uses linear interpolation by default
    clean_series = clean_series.interpolate(method='linear')

    # Step 2: Handle missing values at the beginning (extrapolate backwards)
    # For the beginning of the series, use forward fill as a simple extrapolation
    if clean_series.isna().any():
        clean_series = clean_series.bfill()

    # Step 3: Handle any remaining NaNs at the end (extrapolate forward)
    if clean_series.isna().any():
        clean_series = clean_series.ffill()

    # Step 4: Replace any remaining NaNs or non-positive values with a small positive number
    clean_series = clean_series.fillna(0.01)
    clean_series = clean_series.apply(lambda x: max(x, 0.01))

    return clean_series


class EnhancedStrategyBuilder:
    """Enhanced strategy building component with multiple strategy types."""

    def __init__(self):
        """Initialize the strategy builder with a risk manager."""
        self.risk_manager = PairRiskManager()
        self.session_manager = SessionStateManager()

    def render(self):
        """Render the strategy builder interface."""
        st.header("Strategy Builder")

        if not self.session_manager.has_required_data():
            st.warning("Please select pairs first in the Pair Analysis section.")
            return

        strategy_type = self._render_strategy_selection()

        risk_params = self._render_risk_management()

        backtest_params = self._render_backtest_config()

        if st.button("Run Backtest"):
            self._run_backtest(
                strategy_type=strategy_type,
                risk_params=risk_params,
                backtest_params=backtest_params
            )

            if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None:
                if 'metrics' in st.session_state['backtest_results']:
                    self._display_backtest_results()
                else:
                    st.error("Backtest completed but no metrics were generated. Please check the logs for details.")

        elif 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None:
            if st.checkbox("Show Previous Backtest Results", value=False):
                if 'metrics' in st.session_state['backtest_results']:
                    self._display_backtest_results()
                else:
                    st.error("Previous backtest results are invalid. Please run a new backtest.")

    def _render_strategy_selection(self) -> Dict:
        """Render strategy selection and configuration interface."""
        st.subheader("Strategy Configuration")

        strategy_type = st.selectbox(
            "Strategy Type",
            ["Integrated", "Statistical", "Machine Learning", "Deep Learning", "Multi-Pair Statistical",
             "Dynamic Pairs Trading"]
        )

        params = {}

        if strategy_type == "Dynamic Pairs Trading":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Core Parameters")
                params.update({
                    'initial_capital': st.number_input(
                        "Initial Capital",
                        min_value=100000,
                        max_value=10000000,
                        value=1500000,
                        step=100000
                    ),
                    'window_size': st.number_input(
                        "Window Size (days)",
                        min_value=30,
                        max_value=252,
                        value=90
                    ),
                    'threshold': st.number_input(
                        "Z-Score Threshold",
                        min_value=0.5,
                        max_value=4.0,
                        value=2.0,
                        step=0.1
                    ),
                    'transaction_cost_bps': st.number_input(
                        "Transaction Cost (bps)",
                        min_value=0.0,
                        max_value=20.0,
                        value=1.0,
                        step=0.1
                    ),
                    'max_active_pairs': st.number_input(
                        "Maximum Active Pairs",
                        min_value=1,
                        max_value=20,
                        value=15
                    )
                })

            with col2:
                st.markdown("### Risk Parameters")
                params.update({
                    'stop_loss_pct': st.number_input(
                        "Stop Loss (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5
                    ) / 100,
                    'capital_utilization': st.number_input(
                        "Capital Utilization",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.8,
                        step=0.1
                    ),
                    'max_holding_period': st.number_input(
                        "Max Holding Period (days)",
                        min_value=5,
                        max_value=90,
                        value=30
                    ),
                    'profit_target_pct': st.number_input(
                        "Profit Target (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5
                    ) / 100,
                    'loss_limit_pct': st.number_input(
                        "Loss Limit (%)",
                        min_value=1.0,
                        max_value=15.0,
                        value=3.0,
                        step=0.5
                    ) / 100
                })

            st.markdown("### Pair Selection & Evaluation Parameters")
            col3, col4 = st.columns(2)

            with col3:
                params.update({
                    'pair_evaluation_freq': st.number_input(
                        "Pair Evaluation Frequency (days)",
                        min_value=5,
                        max_value=60,
                        value=20
                    ),
                    'universe_reevaluation_freq': st.number_input(
                        "Universe Reevaluation Frequency (days)",
                        min_value=20,
                        max_value=120,
                        value=60
                    ),
                    'capital_reallocation_freq': st.number_input(
                        "Capital Reallocation Frequency (days)",
                        min_value=10,
                        max_value=120,
                        value=60
                    )
                })

            with col4:
                params.update({
                    'min_cointegration_pvalue': st.number_input(
                        "Min. Cointegration P-value",
                        min_value=0.01,
                        max_value=0.2,
                        value=0.05,
                        step=0.01
                    ),
                    'min_correlation': st.number_input(
                        "Min. Correlation",
                        min_value=0.4,
                        max_value=0.9,
                        value=0.6,
                        step=0.05
                    ),
                    'volatility_adjustment_factor': st.number_input(
                        "Volatility Adjustment Factor",
                        min_value=0.5,
                        max_value=3.0,
                        value=1.5,
                        step=0.1
                    )
                })

            st.markdown("### Advanced Parameters")
            col5, col6 = st.columns(2)

            with col5:
                params.update({
                    'lookback_window': st.number_input(
                        "Lookback Window for Stability Tests (days)",
                        min_value=63,
                        max_value=756,
                        value=252
                    ),
                    'min_data_points': st.number_input(
                        "Minimum Data Points Required",
                        min_value=100,
                        max_value=500,
                        value=252
                    )
                })

            # Pair selection method
            with col6:
                pair_selection = st.radio(
                    "Pair Selection Method",
                    ["Use Selected Pairs", "Auto-Generate Pairs"]
                )

                if pair_selection == "Auto-Generate Pairs":
                    params.update({
                        'auto_generate_pairs': True,
                        'max_pairs_to_generate': st.number_input(
                            "Number of Pairs to Generate",
                            min_value=5,
                            max_value=30,
                            value=15
                        )
                    })
                else:
                    params.update({
                        'auto_generate_pairs': False
                    })
        elif strategy_type == "Integrated":
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Core Parameters")
                params.update({
                    'lookback_window': st.number_input(
                        "Lookback Window (days)",
                        min_value=10,
                        max_value=252,
                        value=126
                    ),
                    'zscore_entry': st.number_input(
                        "Z-Score Entry",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.25,
                        step=0.05
                    ),
                    'zscore_exit': st.number_input(
                        "Z-Score Exit",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.5,
                        step=0.05
                    ),
                    'instant_entry_threshold': st.number_input(
                        "Instant Entry Z-Score",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.75,
                        step=0.05
                    )
                })

            with col2:
                st.markdown("### Risk Parameters")
                params.update({
                    'stop_loss': st.number_input(
                        "Stop Loss (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=4.0,
                        step=0.5
                    ) / 100,
                    'trailing_stop': st.number_input(
                        "Trailing Stop (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=2.0,
                        step=0.5
                    ) / 100,
                    'profit_take': st.number_input(
                        "Profit Take (%)",
                        min_value=1.0,
                        max_value=30.0,
                        value=10.0,
                        step=0.5
                    ) / 100,
                    'time_stop': st.number_input(
                        "Time Stop (days)",
                        min_value=1,
                        max_value=100,
                        value=25
                    )
                })

            with col3:
                st.markdown("### Position Parameters")
                params.update({
                    'position_size': st.number_input(
                        "Position Size (%)",
                        min_value=1.0,
                        max_value=50.0,
                        value=10.0,
                        step=1.0
                    ) / 100,
                    'min_correlation': st.number_input(
                        "Min Correlation",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05
                    ),
                    'momentum_filter': st.number_input(
                        "Momentum Filter Period",
                        min_value=5,
                        max_value=100,
                        value=20
                    )
                })

            st.markdown("### Advanced Parameters")
            col4, col5 = st.columns(2)

            with col4:
                params.update({
                    'regime_lookback': st.number_input(
                        "Regime Detection Window",
                        min_value=21,
                        max_value=252,
                        value=126
                    ),
                    'confirmation_periods': st.number_input(
                        "Signal Confirmation Periods",
                        min_value=1,
                        max_value=10,
                        value=1
                    ),
                    'partial_take_profits': st.multiselect(
                        "Partial Take Profit Levels",
                        options=[
                            (0.05, 0.3),
                            (0.08, 0.5),
                            (0.12, 0.7)
                        ],
                        default=[(0.05, 0.3), (0.08, 0.5)]
                    )
                })

            with col5:
                params.update({
                    'signal_exit_threshold': st.number_input(
                        "Signal Exit Threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1
                    ),
                    'max_portfolio_vol': st.number_input(
                        "Max Portfolio Volatility",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.15,
                        step=0.01
                    )
                })

        elif strategy_type == "Statistical":
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Core Parameters")
                params.update({
                    'lookback_window': st.number_input(
                        "Lookback Window (days)",
                        min_value=10,
                        max_value=252,
                        value=252
                    ),
                    'zscore_entry': st.number_input(
                        "Z-Score Entry",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.0,
                        step=0.1
                    ),
                    'zscore_exit': st.number_input(
                        "Z-Score Exit",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.5,
                        step=0.1
                    ),
                    'coint_threshold': st.number_input(
                        "Cointegration Threshold",
                        min_value=0.01,
                        max_value=0.10,
                        value=0.05,
                        step=0.01,
                        help="Maximum p-value for cointegration test"
                    )
                })

            with col2:
                st.markdown("### Trading Parameters")
                params.update({
                    'min_half_life': st.number_input(
                        "Min Half-Life (days)",
                        min_value=1,
                        max_value=63,
                        value=5
                    ),
                    'max_half_life': st.number_input(
                        "Max Half-Life (days)",
                        min_value=64,
                        max_value=252,
                        value=126
                    ),
                    'max_spread_vol': st.number_input(
                        "Max Spread Volatility",
                        min_value=0.01,
                        max_value=1.0,
                        value=0.1,
                        step=0.01
                    ),
                    'min_correlation': st.number_input(
                        "Min Correlation",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1
                    )
                })

            with col3:
                st.markdown("### Risk Parameters")
                params.update({
                    'max_pairs': st.number_input(
                        "Maximum Pairs",
                        min_value=1,
                        max_value=50,
                        value=10
                    ),
                    'position_size': st.number_input(
                        "Position Size",
                        min_value=0.01,
                        max_value=1.0,
                        value=0.1,
                        step=0.01
                    ),
                    'stop_loss': st.number_input(
                        "Stop Loss",
                        min_value=0.01,
                        max_value=0.1,
                        value=0.02,
                        step=0.01
                    ),
                    'max_drawdown': st.number_input(
                        "Max Drawdown",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.2,
                        step=0.01
                    )
                })

            st.markdown("### Advanced Parameters")
            col4, col5 = st.columns(2)

            with col4:
                params.update({
                    'cointegration_windows': st.multiselect(
                        "Cointegration Windows",
                        options=[21, 63, 126, 252],
                        default=[63, 126, 252]
                    ),
                    'min_votes': st.number_input(
                        "Minimum Cointegration Votes",
                        min_value=1,
                        max_value=3,
                        value=2
                    ),
                    'regime_adaptation': st.checkbox(
                        "Enable Regime Adaptation",
                        value=True
                    )
                })

            with col5:
                params.update({
                    'close_on_signal_flip': st.checkbox(
                        "Close on Signal Flip",
                        value=True
                    ),
                    'signal_exit_threshold': st.number_input(
                        "Signal Exit Threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.3,
                        step=0.1
                    ),
                    'confirmation_periods': st.number_input(
                        "Confirmation Periods",
                        min_value=1,
                        max_value=5,
                        value=2
                    ),
                    'close_on_regime_change': st.checkbox(
                        "Close on Regime Change",
                        value=True
                    )
                })

        elif strategy_type == "Machine Learning":
            col1, col2 = st.columns(2)
            with col1:
                params.update({
                    'lookback_windows': st.multiselect(
                        "Lookback Windows",
                        [5, 10, 21, 63, 126],
                        default=[21, 63]
                    ),
                    'zscore_threshold': st.slider(
                        "Z-Score Threshold",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.0,
                        step=0.1
                    )
                })
            with col2:
                params.update({
                    'train_size': st.number_input(
                        "Training Window (days)",
                        min_value=126,
                        max_value=756,
                        value=252
                    ),
                    'validation_size': st.number_input(
                        "Validation Window (days)",
                        min_value=21,
                        max_value=252,
                        value=63
                    )
                })

        elif strategy_type == "Deep Learning":
            col1, col2 = st.columns(2)
            with col1:
                params.update({
                    'sequence_length': st.number_input(
                        "Sequence Length",
                        min_value=5,
                        max_value=100,
                        value=20
                    ),
                    'prediction_horizon': st.number_input(
                        "Prediction Horizon",
                        min_value=1,
                        max_value=10,
                        value=1
                    ),
                    'train_size': st.number_input(
                        "Training Window (days)",
                        min_value=126,
                        max_value=756,
                        value=252
                    )
                })
            with col2:
                params.update({
                    'zscore_threshold': st.slider(
                        "Z-Score Threshold",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.0,
                        step=0.1
                    ),
                    'validation_size': st.number_input(
                        "Validation Window (days)",
                        min_value=21,
                        max_value=252,
                        value=63
                    )
                })

        elif strategy_type == "Multi-Pair Statistical":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Core Parameters")
                params.update({
                    'window_size': st.number_input(
                        "Window Size (days)",
                        min_value=30,
                        max_value=252,
                        value=90
                    ),
                    'threshold': st.number_input(
                        "Z-Score Threshold",
                        min_value=0.5,
                        max_value=4.0,
                        value=2.0,
                        step=0.1
                    ),
                    'initial_capital': st.number_input(
                        "Initial Capital",
                        min_value=100000,
                        max_value=10000000,
                        value=1500000,
                        step=100000
                    ),
                    'transaction_cost_bps': st.number_input(
                        "Transaction Cost (bps)",
                        min_value=0.0,
                        max_value=20.0,
                        value=1.0,
                        step=0.1
                    )
                })

            with col2:
                st.markdown("### Risk Parameters")
                params.update({
                    'stop_loss_pct': st.number_input(
                        "Stop Loss (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5
                    ) / 100,
                    'capital_utilization': st.number_input(
                        "Capital Utilization",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.8,
                        step=0.1
                    ),
                    'max_holding_period': st.number_input(
                        "Max Holding Period (days)",
                        min_value=5,
                        max_value=90,
                        value=30
                    ),
                    'profit_target_pct': st.number_input(
                        "Profit Target (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5
                    ) / 100,
                    'loss_limit_pct': st.number_input(
                        "Loss Limit (%)",
                        min_value=1.0,
                        max_value=15.0,
                        value=3.0,
                        step=0.5
                    ) / 100
                })

            st.markdown("### Capital Allocation")
            params.update({
                'capital_reallocation_freq': st.number_input(
                    "Capital Reallocation Frequency (days)",
                    min_value=10,
                    max_value=120,
                    value=60
                )
            })

            # Pair selection method
            st.markdown("### Pair Selection")
            pair_selection = st.radio(
                "Pair Selection Method",
                ["Use Selected Pairs", "Find Correlated Pairs", "Find Cointegrated Pairs"]
            )

            if pair_selection == "Find Correlated Pairs":
                params.update({
                    'correlation_threshold': st.number_input(
                        "Correlation Threshold",
                        min_value=0.5,
                        max_value=0.95,
                        value=0.7,
                        step=0.05
                    ),
                    'max_pairs': st.number_input(
                        "Maximum Number of Pairs",
                        min_value=1,
                        max_value=20,
                        value=5
                    ),
                    'pair_selection': 'correlation'
                })
            elif pair_selection == "Find Cointegrated Pairs":
                params.update({
                    'p_value_threshold': st.number_input(
                        "Cointegration P-Value Threshold",
                        min_value=0.01,
                        max_value=0.15,
                        value=0.05,
                        step=0.01
                    ),
                    'max_pairs': st.number_input(
                        "Maximum Number of Pairs",
                        min_value=1,
                        max_value=20,
                        value=5
                    ),
                    'pair_selection': 'cointegration'
                })
            else:
                params.update({
                    'pair_selection': 'selected'
                })

        self.session_manager.update_strategy_params(strategy_type, params)

        return {
            'type': strategy_type,
            'params': params
        }

    def _render_risk_management(self) -> Dict:
        """
        Render risk management configuration interface.

        Returns:
            Dict: Risk management parameters
        """
        st.subheader("Risk Management")

        col1, col2, col3 = st.columns(3)
        with col1:
            max_position_size = st.number_input(
                "Max Position Size (%)",
                min_value=1.0,
                max_value=100.0,
                value=5.0,
                step=1.0
            ) / 100

            stop_loss = st.number_input(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0
            ) / 100

        with col2:
            max_drawdown = st.number_input(
                "Max Drawdown (%)",
                min_value=1.0,
                max_value=50.0,
                value=20.0,
                step=1.0
            ) / 100

            max_correlation = st.slider(
                "Max Pair Correlation",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )

        with col3:
            leverage_limit = st.number_input(
                "Leverage Limit",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )

        risk_params = {
            'max_position_size': max_position_size,
            'stop_loss': stop_loss,
            'max_drawdown': max_drawdown,
            'max_correlation': max_correlation,
            'leverage_limit': leverage_limit
        }

        self.session_manager.update_risk_params(risk_params)

        return risk_params

    def _render_backtest_config(self) -> Dict:
        """
        Render backtesting configuration interface.

        Returns:
            Dict: Backtest configuration parameters
        """
        st.subheader("Backtest Configuration")

        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000,
                value=100000,
                step=1000
            )

            transaction_cost = st.number_input(
                "Transaction Cost (bps)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0
            ) / 10000

        with col2:
            max_pairs = st.number_input(
                "Maximum Concurrent Pairs",
                min_value=1,
                max_value=20,
                value=5
            )

        return {
            'initial_capital': initial_capital,
            'transaction_cost': transaction_cost,
            'max_pairs': max_pairs
        }

    def _streamlit_plot_pair_rotation(self, dynamic_system):
        """Create a Streamlit-compatible visualization showing pair rotation over time"""
        st.subheader("Pair Rotation Timeline")

        if not dynamic_system.portfolio_history:
            st.warning("No portfolio history available to plot pair rotation")
            return

        # Convert all trade history to DataFrame
        all_trades = dynamic_system.trade_history

        if not all_trades:
            st.warning("No trade history available to plot pair rotation")
            return

        trades_df = pd.DataFrame(all_trades)

        # Get unique pair IDs from trades
        all_pair_ids = trades_df['pair_id'].unique() if 'pair_id' in trades_df.columns else []

        if len(all_pair_ids) == 0:
            st.warning("No pair IDs found in trade history")
            return

        # Create a timeline of active pairs
        dates = [d['date'] for d in dynamic_system.portfolio_history]
        date_range = pd.date_range(start=min(dates), end=max(dates), freq='D')

        # Create a dataframe for pair activity
        pair_activity = {}
        for pair_id in all_pair_ids:
            # Filter trades for this pair
            pair_trades = trades_df[trades_df['pair_id'] == pair_id]

            # Create a series of 1s for active days, 0s for inactive
            activity = pd.Series(0, index=date_range)

            # Group by date to handle multiple trades on same day
            daily_trades = pair_trades.groupby('date').size()

            # Set active days as 1
            for date in daily_trades.index:
                activity[date] = 1

            pair_activity[f"{pair_id[0]}-{pair_id[1]}"] = activity

        # Create a dataframe with all pair activity
        if pair_activity:
            activity_df = pd.DataFrame(pair_activity)

            # Create a Plotly figure for Streamlit
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig = go.Figure()

            # Plot each pair's activity
            for i, pair_name in enumerate(activity_df.columns):
                active_dates = activity_df.index[activity_df[pair_name] == 1]
                if len(active_dates) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=active_dates,
                            y=[i] * len(active_dates),
                            mode='markers',
                            marker=dict(symbol='line-ns', size=15, line=dict(width=1)),
                            name=pair_name
                        )
                    )

            # Update the layout
            fig.update_layout(
                title='Pair Trading Activity Timeline',
                xaxis_title='Date',
                yaxis_title='Pair',
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(activity_df.columns))),
                    ticktext=activity_df.columns
                ),
                height=500 + 20 * len(activity_df.columns),
                width=900,
                showlegend=True
            )

            # Add grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Display in Streamlit
            st.plotly_chart(fig)

    def _streamlit_plot_market_volatility(self, dynamic_system):
        """Create a Streamlit-compatible plot of market volatility over time"""
        st.subheader("Market Volatility Analysis")

        if not dynamic_system.market_state_history:
            st.warning("No market state history available to plot")
            return

        # Convert to DataFrame
        vol_df = pd.DataFrame(dynamic_system.market_state_history)
        vol_df.set_index('date', inplace=True)

        fig = go.Figure()

        # Plot volatility ratio
        fig.add_trace(
            go.Scatter(
                x=vol_df.index,
                y=vol_df['volatility_ratio'],
                mode='lines',
                name='Volatility Ratio',
                line=dict(color='blue', width=2)
            )
        )

        # Add a horizontal line at the threshold
        fig.add_trace(
            go.Scatter(
                x=[vol_df.index[0], vol_df.index[-1]],
                y=[1.5, 1.5],
                mode='lines',
                name='High Volatility Threshold',
                line=dict(color='red', width=1, dash='dash')
            )
        )

        # Shade high volatility regions
        high_vol = vol_df['is_high_volatility']

        # Find contiguous regions
        if any(high_vol):
            switches = high_vol.diff().fillna(0).astype(int)
            starts = vol_df.index[switches == 1].tolist()
            ends = vol_df.index[switches == -1].tolist()

            # If it starts in high volatility
            if high_vol.iloc[0]:
                starts.insert(0, vol_df.index[0])

            # If it ends in high volatility
            if high_vol.iloc[-1]:
                ends.append(vol_df.index[-1])

            # Add shaded regions
            for start, end in zip(starts, ends):
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="red", opacity=0.15,
                    layer="below", line_width=0,
                )

        # Update layout
        fig.update_layout(
            title='Market Volatility Over Time',
            xaxis_title='Date',
            yaxis_title='Volatility Ratio',
            height=500,
            width=900,
        )

        # Display in Streamlit
        st.plotly_chart(fig)

    def _streamlit_plot_pair_metrics(self, dynamic_system):
        """Create a Streamlit-compatible plot of pair quality metrics"""
        st.subheader("Pair Quality Metrics")

        if not dynamic_system.pair_quality_metrics:
            st.warning("No pair quality metrics available to plot")
            return

        # Prepare the data
        pairs = list(dynamic_system.pair_quality_metrics.keys())

        if not pairs:
            st.warning("No pairs with quality metrics found")
            return

        # Create a DataFrame for easier plotting with Plotly
        metrics_data = []
        for pair in pairs:
            pair_name = f"{pair[0]}-{pair[1]}"
            metrics = dynamic_system.pair_quality_metrics[pair]
            metrics_data.append({
                'pair': pair_name,
                'correlation': metrics.get('correlation', 0),
                'pvalue': metrics.get('pvalue', 1.0),
                'score': metrics.get('score', 0)
            })

        metrics_df = pd.DataFrame(metrics_data)

        # Create separate plots for each metric
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Define metrics to plot
        metrics_to_plot = ['correlation', 'pvalue', 'score']

        # Create subplots
        fig = make_subplots(
            rows=len(metrics_to_plot),
            cols=1,
            subplot_titles=[f'Pair {metric.capitalize()}' for metric in metrics_to_plot],
            vertical_spacing=0.1
        )

        # Add bars for each metric
        for i, metric in enumerate(metrics_to_plot):
            fig.add_trace(
                go.Bar(
                    x=metrics_df['pair'],
                    y=metrics_df[metric],
                    name=metric.capitalize()
                ),
                row=i + 1, col=1
            )

        # Update layout
        fig.update_layout(
            height=300 * len(metrics_to_plot),
            width=900,
            showlegend=False
        )

        # Customize y-axis titles
        for i, metric in enumerate(metrics_to_plot):
            fig.update_yaxes(title_text=metric.capitalize(), row=i + 1, col=1)

        # Rotate x-axis labels if many pairs
        if len(pairs) > 5:
            fig.update_xaxes(tickangle=45)

        # Display in Streamlit
        st.plotly_chart(fig)

    def _streamlit_plot_portfolio_overview(self, dynamic_system):
        """Create a Streamlit-compatible enhanced portfolio overview"""
        st.subheader("Portfolio Overview with Volatility Regime")

        if not dynamic_system.portfolio_history:
            st.warning("No portfolio history available to plot")
            return

        portfolio_df = pd.DataFrame(dynamic_system.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        # Create a merged dataframe with volatility info if available
        if dynamic_system.market_state_history:
            vol_df = pd.DataFrame(dynamic_system.market_state_history)
            vol_df.set_index('date', inplace=True)

            # Align the indices (may have different frequencies)
            merged = portfolio_df.join(vol_df[['is_high_volatility', 'volatility_ratio']], how='left')

            # Fill any missing volatility data
            merged['is_high_volatility'].fillna(False, inplace=True)
            merged['volatility_ratio'].fillna(1.0, inplace=True)
        else:
            merged = portfolio_df.copy()
            merged['is_high_volatility'] = False
            merged['volatility_ratio'] = 1.0

        # Create Plotly figure
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Portfolio Value with Volatility Regime', 'Active Pairs and Volatility Ratio')
        )

        # Plot portfolio value
        fig.add_trace(
            go.Scatter(
                x=merged.index,
                y=merged['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Add initial capital reference line
        fig.add_trace(
            go.Scatter(
                x=[merged.index[0], merged.index[-1]],
                y=[dynamic_system.initial_capital, dynamic_system.initial_capital],
                mode='lines',
                name='Initial Capital',
                line=dict(color='black', width=1, dash='dash')
            ),
            row=1, col=1
        )

        # Shade high volatility regions
        high_vol = merged['is_high_volatility']

        # Find contiguous regions
        if any(high_vol):
            switches = high_vol.diff().fillna(0).astype(int)
            starts = merged.index[switches == 1].tolist()
            ends = merged.index[switches == -1].tolist()

            # If it starts in high volatility
            if high_vol.iloc[0]:
                starts.insert(0, merged.index[0])

            # If it ends in high volatility
            if high_vol.iloc[-1]:
                ends.append(merged.index[-1])

            # Add shaded regions to both subplots
            for start, end in zip(starts, ends):
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="red", opacity=0.15,
                    layer="below", line_width=0,
                    row=1, col=1
                )
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="red", opacity=0.15,
                    layer="below", line_width=0,
                    row=2, col=1
                )

        # Plot active pairs
        fig.add_trace(
            go.Scatter(
                x=merged.index,
                y=merged['active_pairs'],
                mode='lines',
                name='Active Pairs',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )

        # Add volatility ratio on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=merged.index,
                y=merged['volatility_ratio'],
                mode='lines',
                name='Volatility Ratio',
                line=dict(color='red', width=1.5),
                yaxis="y3"
            ),
            row=2, col=1
        )

        # Add threshold line on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=[merged.index[0], merged.index[-1]],
                y=[1.5, 1.5],
                mode='lines',
                name='Volatility Threshold',
                line=dict(color='red', width=1, dash='dash'),
                yaxis="y3"
            ),
            row=2, col=1
        )

        # Update layout with secondary y-axis
        fig.update_layout(
            yaxis3=dict(
                title="Volatility Ratio",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                anchor="x",
                overlaying="y2",
                side="right"
            ),
            height=800,
            width=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Set y-axis titles
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Active Pairs", row=2, col=1)

        # Display in Streamlit
        st.plotly_chart(fig)

    def _run_dynamic_pairs_backtest(self, prices: pd.DataFrame, pairs: List[Tuple[str, str]],
                                    params: Dict, backtest_params: Dict):
        """
        Run backtest with the Dynamic Pairs Trading strategy with Streamlit integration.

        Args:
            prices: Price data
            pairs: Selected pairs for trading
            params: Strategy parameters
            backtest_params: General backtest configuration
        """
        # Get price data in pivot format (required for DynamicPairTradingSystem)
        if 'pivot_prices' in st.session_state:
            prices_df = st.session_state['pivot_prices']
        else:
            # Create pivot table from historical data
            data = st.session_state['historical_data']
            data['Date'] = pd.to_datetime(data['Date'])

            prices_df = data.pivot(
                index='Date',
                columns='Symbol',
                values='Adj_Close'
            )
            prices_df = prices_df.sort_index()
            prices_df = prices_df.ffill().bfill()
            st.session_state['pivot_prices'] = prices_df

        # Handle pair generation if auto-generate is selected
        initial_pairs = None
        if not params.get('auto_generate_pairs', False):
            initial_pairs = pairs

        # Create the dynamic pairs trading system
        with st.spinner("Initializing Dynamic Pairs Trading System..."):
            dynamic_system = DynamicPairTradingSystem(
                prices=prices_df,
                initial_pairs=initial_pairs,
                initial_capital=params.get('initial_capital', 1500000),
                window_size=params.get('window_size', 90),
                threshold=params.get('threshold', 2.0),
                transaction_cost_bps=params.get('transaction_cost_bps', 1),
                stop_loss_pct=params.get('stop_loss_pct', 0.05),
                capital_utilization=params.get('capital_utilization', 0.8),
                max_holding_period=params.get('max_holding_period', 30),
                profit_target_pct=params.get('profit_target_pct', 0.05),
                loss_limit_pct=params.get('loss_limit_pct', 0.03),
                capital_reallocation_freq=params.get('capital_reallocation_freq', 60),
                max_active_pairs=params.get('max_active_pairs', 15),
                pair_evaluation_freq=params.get('pair_evaluation_freq', 20),
                universe_reevaluation_freq=params.get('universe_reevaluation_freq', 60),
                min_cointegration_pvalue=params.get('min_cointegration_pvalue', 0.05),
                min_correlation=params.get('min_correlation', 0.6),
                lookback_window=params.get('lookback_window', 252),
                volatility_adjustment_factor=params.get('volatility_adjustment_factor', 1.5),
                min_data_points=params.get('min_data_points', 252)
            )

        # Run the backtest (not run_strategy to avoid matplotlib plots)
        with st.spinner("Running Dynamic Pairs Trading backtest..."):
            # Set initial dates to avoid immediate evaluation
            first_date = dynamic_system.prices.index[0]
            dynamic_system.last_pair_evaluation = first_date
            dynamic_system.last_universe_evaluation = first_date

            # Run backtest
            dynamic_system.run_backtest()

            # Get portfolio metrics
            metrics = dynamic_system.get_portfolio_metrics()

        # Display metrics and visualizations
        st.subheader("Dynamic Pairs Trading Results")

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        total_return = metrics['Portfolio Metrics'].get('Total Return (%)', 0)
        with col1:
            st.metric(
                "Total Return",
                f"{total_return:.2f}%" if isinstance(total_return, (int, float)) else total_return
            )

        sharpe = metrics['Portfolio Metrics'].get('Sharpe Ratio', 0)
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else sharpe
            )

        drawdown = metrics['Portfolio Metrics'].get('Max Drawdown (%)', 0)
        with col3:
            st.metric(
                "Max Drawdown",
                f"{drawdown:.2f}%" if isinstance(drawdown, (int, float)) else drawdown
            )

        win_rate = metrics['Trading Activity'].get('Average Win Rate (%)', 0)
        with col4:
            st.metric(
                "Win Rate",
                f"{win_rate:.2f}%" if isinstance(win_rate, (int, float)) else win_rate
            )

        # Display Streamlit-compatible visualizations
        self._streamlit_plot_portfolio_overview(dynamic_system)
        self._streamlit_plot_pair_rotation(dynamic_system)
        self._streamlit_plot_market_volatility(dynamic_system)
        self._streamlit_plot_pair_metrics(dynamic_system)

        # Display individual pair analyses
        st.subheader("Individual Pair Analysis")

        # Select which pair to view
        all_pairs = list(dynamic_system.pair_models.keys())
        if all_pairs:
            selected_pair = st.selectbox(
                "Select a pair to analyze:",
                options=all_pairs,
                format_func=lambda p: f"{p[0]}-{p[1]}"
            )

            with st.spinner(f"Generating analysis for {selected_pair[0]}-{selected_pair[1]}..."):
                model = dynamic_system.pair_models[selected_pair]

                # Get metrics for this pair
                pair_metrics = model.get_metrics()

                # Display key pair metrics
                pair_cols = st.columns(4)
                pair_cols[0].metric(
                    "Pair Return",
                    f"{pair_metrics.get('total_return', 0):.2f}%"
                )
                pair_cols[1].metric(
                    "Win Rate",
                    f"{pair_metrics.get('win_rate', 0):.2f}%"
                )
                pair_cols[2].metric(
                    "Profit Factor",
                    f"{pair_metrics.get('profit_factor', 0):.2f}"
                )
                pair_cols[3].metric(
                    "# Trades",
                    pair_metrics.get('num_trades', 0)
                )

                # Create and display pair visualization using Plotly
                if model.portfolio_history and model.spread_history:
                    self._display_pair_analysis_plotly(model)
                else:
                    st.warning("Insufficient data to generate pair analysis visualization")
        else:
            st.warning("No pairs available for analysis")

        # Detailed metrics in expandable sections
        with st.expander("Portfolio Metrics", expanded=False):
            for category, category_metrics in metrics.items():
                st.write(f"### {category}")
                metrics_df = pd.DataFrame(
                    {k: [v] for k, v in category_metrics.items()}
                ).T.reset_index()
                metrics_df.columns = ['Metric', 'Value']
                st.dataframe(metrics_df, use_container_width=True)

        # Store the results in session state
        portfolio_df = pd.DataFrame(dynamic_system.portfolio_history)

        # Get trades from all pair models
        all_trades = dynamic_system.trade_history
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

        # Create aggregate performance metrics
        portfolio_performance = metrics.get('Portfolio Metrics', {})

        st.session_state['backtest_results'] = {
            'equity_curve': portfolio_df.set_index('date')[
                'portfolio_value'] if 'date' in portfolio_df.columns else None,
            'metrics': {
                'Total Return': portfolio_performance.get('Total Return (%)', 0),
                'Annual Return': portfolio_performance.get('Annual Return (%)', 0),
                'Annual Volatility': portfolio_performance.get('Annual Volatility (%)', 0),
                'Sharpe Ratio': portfolio_performance.get('Sharpe Ratio', 0),
                'Max Drawdown': portfolio_performance.get('Max Drawdown (%)', 0),
                'Win Rate': metrics.get('Trading Activity', {}).get('Average Win Rate (%)', 0)
            },
            'trades': trades_df,
            'parameters': {
                'strategy': {'type': 'Dynamic Pairs Trading', 'params': params},
                'backtest': backtest_params,
                'pairs': dynamic_system.pairs if hasattr(dynamic_system, 'pairs') else []
            },
            'system': dynamic_system  # Store the system for further analysis
        }

    def _display_pair_analysis_plotly(self, pair_model: PairModel):
        """Display detailed pair analysis using Plotly."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Convert history data to DataFrames
        portfolio_df = pd.DataFrame(pair_model.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        spread_df = pd.DataFrame(pair_model.spread_history)
        spread_df.set_index('date', inplace=True)

        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'Price Series: {pair_model.symbol_x} vs {pair_model.symbol_y}',
                'Pair Model Equity Curve',
                'Spread with Bands'
            ),
            vertical_spacing=0.1,
            shared_xaxes=True,
            row_heights=[0.4, 0.3, 0.3]
        )

        # 1. Price Series
        fig.add_trace(
            go.Scatter(
                x=pair_model.data.index,
                y=pair_model.data[pair_model.symbol_x],
                name=pair_model.symbol_x,
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=pair_model.data.index,
                y=pair_model.data[pair_model.symbol_y],
                name=pair_model.symbol_y,
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # Add normalized prices on the same chart
        normalized_x = pair_model.data[pair_model.symbol_x] / pair_model.data[pair_model.symbol_x].iloc[0]
        normalized_y = pair_model.data[pair_model.symbol_y] / pair_model.data[pair_model.symbol_y].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=pair_model.data.index,
                y=normalized_x,
                name=f"{pair_model.symbol_x} (Norm)",
                line=dict(color='lightblue', dash='dash')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=pair_model.data.index,
                y=normalized_y,
                name=f"{pair_model.symbol_y} (Norm)",
                line=dict(color='lightcoral', dash='dash')
            ),
            row=1, col=1
        )

        # 2. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='green')
            ),
            row=2, col=1
        )

        # Add cash component
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['cash'],
                name='Cash',
                line=dict(color='lightgreen', dash='dot')
            ),
            row=2, col=1
        )

        # 3. Spread with Bands
        if not spread_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['spread'],
                    name='Spread',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['upper_band'],
                    name='Upper Band',
                    line=dict(color='gray', dash='dash')
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['lower_band'],
                    name='Lower Band',
                    line=dict(color='gray', dash='dash')
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df['mean'],
                    name='Mean',
                    line=dict(color='black', dash='dot')
                ),
                row=3, col=1
            )

        # Add trade markers if available
        if pair_model.trade_history:
            trade_df = pd.DataFrame(pair_model.trade_history)

            # Add trade markers on price chart
            for _, trade in trade_df.iterrows():
                marker_color = 'green' if trade['type'] == 'buy' else 'red'
                marker_symbol = 'triangle-up' if trade['type'] == 'buy' else 'triangle-down'

                fig.add_trace(
                    go.Scatter(
                        x=[trade['date']],
                        y=[trade['price']],
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbol,
                            color=marker_color,
                            size=10
                        ),
                        name=f"{trade['type']} {trade['symbol']}",
                        showlegend=False
                    ),
                    row=1, col=1
                )

        # Update layout
        fig.update_layout(
            height=900,
            width=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display the figure
        st.plotly_chart(fig)

    def _run_backtest(self,
                      strategy_type: Dict,
                      risk_params: Dict,
                      backtest_params: Dict):
        """
        Run backtest with configured strategy and parameters.

        Args:
            strategy_type: Strategy type and parameters
            risk_params: Risk management parameters
            backtest_params: Backtest configuration parameters
        """
        try:
            with st.spinner("Running backtest..."):
                prices = self._get_price_data()
                pairs = self._get_selected_pairs()

                if strategy_type['type'] == "Multi-Pair Statistical":
                    # For Multi-Pair Statistical strategy, use the integrated code from paste.txt
                    self._run_multi_pair_backtest(
                        prices=prices,
                        pairs=pairs,
                        params=strategy_type['params'],
                        backtest_params=backtest_params
                    )
                elif strategy_type['type'] == "Dynamic Pairs Trading":
                    # For Dynamic Pairs Trading, use the new implementation
                    self._run_dynamic_pairs_backtest(
                        prices=prices,
                        pairs=pairs,
                        params=strategy_type['params'],
                        backtest_params=backtest_params
                    )
                else:
                    # For other strategies, use the existing implementation
                    strategy = self._create_strategy(
                        strategy_type['type'],
                        strategy_type['params']
                    )

                    if hasattr(strategy, 'set_tradeable_pairs'):
                        strategy.set_tradeable_pairs(pairs)
                    else:
                        strategy.pairs = pairs

                    risk_manager = PairRiskManager(
                        max_position_size=risk_params['max_position_size'],
                        max_drawdown=risk_params['max_drawdown'],
                        stop_loss_threshold=risk_params['stop_loss'],
                        max_correlation=risk_params['max_correlation'],
                        leverage_limit=risk_params['leverage_limit']
                    )

                    backtester = MultiPairBackTester(
                        strategy=strategy,
                        prices=prices,
                        initial_capital=backtest_params['initial_capital'],
                        risk_manager=risk_manager,
                        transaction_cost=backtest_params['transaction_cost'],
                        max_pairs=backtest_params['max_pairs']
                    )

                    equity_curve = backtester.run_backtest()

                    if isinstance(strategy, IntegratedPairsStrategy):
                        st.subheader("Strategy Performance Dashboard")

                        fig = create_strategy_dashboard(
                            backtester.strategy_results,
                            backtester.asset1_data,
                            backtester.asset2_data,
                            strategy.trades
                        )
                        st.plotly_chart(fig)

                    st.session_state['backtest_results'] = {
                        'equity_curve': equity_curve,
                        'metrics': backtester._calculate_performance_metrics(),
                        'trades': backtester.trade_history,
                        'parameters': {
                            'strategy': strategy_type,
                            'risk': risk_params,
                            'backtest': backtest_params,
                            'pairs': pairs
                        }
                    }

            st.success("Backtest completed successfully!")

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            logger.error(f"Backtest error: {str(e)}", exc_info=True)
            error_trace = traceback.format_exc()
            logger.error(f"Error in backtest at: {str(e)}\nFull traceback:\n{error_trace}")

    def _run_multi_pair_backtest(self, prices: pd.DataFrame, pairs: List[Tuple[str, str]],
                                 params: Dict, backtest_params: Dict):
        """
        Run backtest with the Multi-Pair Statistical strategy from paste.txt.

        Args:
            prices: Price data
            pairs: Selected pairs for trading
            params: Strategy parameters
            backtest_params: General backtest configuration
        """
        # Handle different pair selection methods
        if params.get('pair_selection') == 'correlation':
            # Find pairs using correlation
            pairs = find_correlated_pairs(
                prices_df=prices,
                correlation_threshold=params.get('correlation_threshold', 0.7),
                max_pairs=params.get('max_pairs', 5)
            )
            st.info(f"Found {len(pairs)} correlated pairs: {pairs}")

        elif params.get('pair_selection') == 'cointegration':
            # Find pairs using cointegration
            pairs = find_cointegrated_pairs(
                prices_df=prices,
                p_value_threshold=params.get('p_value_threshold', 0.05),
                max_pairs=params.get('max_pairs', 5)
            )
            st.info(f"Found {len(pairs)} cointegrated pairs: {pairs}")

        # Initialize MultiPairTradingSystem with the specified parameters
        system = MultiPairTradingSystem(
            pairs=pairs,
            prices=prices,
            initial_capital=params.get('initial_capital', 1500000),
            window_size=params.get('window_size', 90),
            threshold=params.get('threshold', 2.0),
            transaction_cost_bps=params.get('transaction_cost_bps', 1.0),
            stop_loss_pct=params.get('stop_loss_pct', 0.05),
            capital_utilization=params.get('capital_utilization', 0.8),
            max_holding_period=params.get('max_holding_period', 30),
            profit_target_pct=params.get('profit_target_pct', 0.05),
            loss_limit_pct=params.get('loss_limit_pct', 0.03),
            capital_reallocation_freq=params.get('capital_reallocation_freq', 60)
        )

        # Run the backtest
        with st.spinner("Running Multi-Pair Trading System backtest..."):
            system.run_backtest()

            # Get and display portfolio metrics
            portfolio_metrics = system.get_portfolio_metrics()

            # Create portfolio overview visualization
            st.subheader("Portfolio Overview")
            fig = system.plot_portfolio_overview()
            st.plotly_chart(fig)

            # Display individual pair analyses
            st.subheader("Individual Pair Analysis")

            for pair, model in system.pair_models.items():
                pair_name = f"{pair[0]}-{pair[1]}"
                with st.expander(f"Pair Analysis: {pair_name}"):
                    # Get metrics for this pair
                    metrics = model.get_metrics()

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
                    col2.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
                    col3.metric("Number of Trades", metrics.get('num_trades', 0))

                    # Create and display pair visualization
                    pair_fig = model.plot_pair_analysis()
                    st.plotly_chart(pair_fig)

            # Store the results in session state
            trades_df = pd.DataFrame(system.trade_history)

            # Create aggregate performance metrics
            portfolio_performance = portfolio_metrics.get('Portfolio Metrics', {})

            st.session_state['backtest_results'] = {
                'equity_curve': pd.DataFrame(system.portfolio_history).set_index('date')['portfolio_value'],
                'metrics': {
                    'Total Return': portfolio_performance.get('Total Return (%)', 0),
                    'Annual Return': portfolio_performance.get('Annual Return (%)', 0),
                    'Annual Volatility': portfolio_performance.get('Annual Volatility (%)', 0),
                    'Sharpe Ratio': portfolio_performance.get('Sharpe Ratio', 0),
                    'Max Drawdown': portfolio_performance.get('Max Drawdown (%)', 0),
                    'Win Rate': portfolio_metrics.get('Trading Activity', {}).get('Average Win Rate (%)', 0)
                },
                'trades': trades_df,
                'parameters': {
                    'strategy': {'type': 'Multi-Pair Statistical', 'params': params},
                    'backtest': backtest_params,
                    'pairs': pairs
                },
                'system': system  # Store the system for further analysis
            }

    def _get_price_data(self) -> pd.DataFrame:
        """
        Get price data from session state.

        Returns:
            pd.DataFrame: Price data for analysis
        """
        if 'historical_data' not in st.session_state:
            raise ValueError("No historical data found in session state")

        data = st.session_state['historical_data']

        if not all(col in data.columns for col in ['Date', 'Symbol', 'Adj_Close', 'Volume']):
            raise ValueError("Required columns missing in historical data")

        available_symbols = set(data['Symbol'].unique())

        for pair in self._get_selected_pairs():
            asset1, asset2 = pair
            if asset1 not in available_symbols:
                raise ValueError(f"Symbol {asset1} not found in historical data")
            if asset2 not in available_symbols:
                raise ValueError(f"Symbol {asset2} not found in historical data")

        filled_data = data.groupby('Symbol').apply(
            lambda x: x.sort_values('Date').ffill().bfill()
        ).reset_index(drop=True)

        # For multi-pair statistical strategy, we need data in pivot format
        # Create a second version in pivot format for price series
        prices_df = filled_data.pivot(
            index='Date',
            columns='Symbol',
            values='Adj_Close'
        )

        # Store prices in session state for potential reuse
        st.session_state['pivot_prices'] = prices_df

        return filled_data.sort_values(['Date', 'Symbol'])

    def _get_selected_pairs(self) -> List[Tuple[str, str]]:
        """
        Get selected pairs from session state with strict type checking.

        Returns:
            List[Tuple[str, str]]: List of validated asset pairs

        Raises:
            ValueError: If no pairs selected
        """
        if 'selected_pairs' not in st.session_state:
            raise ValueError("No pairs selected. Please select pairs first.")

        pairs_df = st.session_state['selected_pairs']
        validated_pairs: List[Tuple[str, str]] = []

        try:
            if 'Asset1' in pairs_df.columns and 'Asset2' in pairs_df.columns:
                for _, row in pairs_df.iterrows():
                    asset1, asset2 = str(row['Asset1']), str(row['Asset2'])
                    validated_pairs.append((min(asset1, asset2), max(asset1, asset2)))

            elif 'Pair' in pairs_df.columns:
                for pair in pairs_df['Pair']:
                    if isinstance(pair, str) and '/' in pair:
                        assets = pair.split('/')
                        if len(assets) == 2:
                            asset1, asset2 = str(assets[0]), str(assets[1])
                            validated_pairs.append((min(asset1, asset2), max(asset1, asset2)))
                    elif isinstance(pair, tuple):
                        if len(pair) == 2:
                            asset1, asset2 = str(pair[0]), str(pair[1])
                            validated_pairs.append((asset1, asset2))

            if not validated_pairs:
                raise ValueError("Could not extract valid pairs from selected pairs data")

            if 'historical_data' in st.session_state:
                data = st.session_state['historical_data']
                available_tickers = set(data['Symbol'].unique())

                valid_pairs = [
                    pair for pair in validated_pairs
                    if pair[0] in available_tickers and pair[1] in available_tickers
                ]

                if not valid_pairs:
                    raise ValueError("No valid pairs found in historical data")

                return valid_pairs

            return validated_pairs

        except Exception as e:
            raise ValueError(f"Error validating pairs: {str(e)}")

    def _create_strategy(self, strategy_type: str, params: Dict):
        """Create and initialize strategy with separate models for each pair."""
        try:
            train_data, test_data = self._get_train_test_data()
            pairs = self._get_selected_pairs()

            if strategy_type == "Integrated":
                strategy = IntegratedPairsStrategy(
                    lookback_window=params['lookback_window'],
                    zscore_entry=params['zscore_entry'],
                    zscore_exit=params['zscore_exit'],
                    stop_loss=params['stop_loss'],
                    trailing_stop=params['trailing_stop'],
                    time_stop=params['time_stop'],
                    profit_take=params['profit_take'],
                    position_size=params['position_size'],
                    min_correlation=params['min_correlation'],
                    signal_exit_threshold=params['signal_exit_threshold'],
                    confirmation_periods=params['confirmation_periods'],
                    max_portfolio_vol=params['max_portfolio_vol'],
                    regime_lookback=params['regime_lookback'],
                    instant_confirm_threshold=params['instant_entry_threshold'],
                    momentum_filter_period=params['momentum_filter'],
                    partial_take_profit_levels=params['partial_take_profits']
                )
                strategy.pairs = pairs
                return strategy

            elif strategy_type == "Statistical":
                strategy = EnhancedStatPairsStrategy(
                    lookback_window=params['lookback_window'],
                    zscore_entry=params['zscore_entry'],
                    zscore_exit=params['zscore_exit'],
                    min_half_life=params['min_half_life'],
                    max_half_life=params['max_half_life'],
                    max_spread_vol=params['max_spread_vol'],
                    min_correlation=params['min_correlation'],
                    coint_threshold=params['coint_threshold'],
                    max_pairs=params['max_pairs'],
                    position_size=params['position_size'],
                    stop_loss=params['stop_loss'],
                    max_drawdown=params['max_drawdown'],
                    cointegration_windows=params['cointegration_windows'],
                    min_votes=params['min_votes'],
                    regime_adaptation=params['regime_adaptation'],
                    close_on_signal_flip=params['close_on_signal_flip'],
                    signal_exit_threshold=params['signal_exit_threshold'],
                    confirmation_periods=params['confirmation_periods'],
                    close_on_regime_change=params['close_on_regime_change']
                )
                strategy.pairs = pairs
                return strategy

            elif strategy_type == "Machine Learning":
                strategy = MLPairsStrategy(
                    initial_capital=1_000_000.0,
                    lookback_window=max(params['lookback_windows']),
                    model_confidence_threshold=params.get('model_confidence_threshold', 0.6),
                    zscore_threshold=params['zscore_threshold'],
                    max_position_size=0.1,
                    stop_loss=0.02,
                    take_profit=0.04,
                )
                strategy.pairs = pairs

                with st.spinner("Training ML models for each pair..."):
                    progress_bar = st.progress(0)
                    for i, pair in enumerate(pairs):
                        progress = (i + 1) / len(pairs)
                        progress_bar.progress(progress)
                        st.write(f"Training model for pair {pair[0]}/{pair[1]}...")

                        pair_data = train_data[[pair[0], pair[1]]]

                        try:
                            strategy.initialize_models(pair_data)
                            st.write(f"✓ Successfully trained model for {pair[0]}/{pair[1]}")
                        except Exception as e:
                            st.error(f"Failed to train model for {pair[0]}/{pair[1]}: {str(e)}")
                            continue

                return strategy

            else:  # Deep Learning
                strategy = PairsTradingDL(
                    sequence_length=params['sequence_length'],
                    prediction_horizon=params['prediction_horizon'],
                    zscore_threshold=params['zscore_threshold'],
                    min_confidence=0.6,
                    max_position_size=0.1,
                )

                strategy.pairs = pairs

                with st.spinner("Training DL models for each pair..."):
                    progress_bar = st.progress(0)
                    for i, pair in enumerate(pairs):
                        progress = (i + 1) / len(pairs)
                        progress_bar.progress(progress)
                        st.write(f"Training deep learning models for pair {pair[0]}/{pair[1]}...")

                        pair_data = train_data[[pair[0], pair[1]]]

                        try:
                            strategy.initialize_models(pair_data)
                            st.write(f"✓ Successfully trained models for {pair[0]}/{pair[1]}")
                        except Exception as e:
                            st.error(f"Failed to train models for {pair[0]}/{pair[1]}: {str(e)}")
                            continue

                return strategy

        except Exception as e:
            st.error(f"Error creating strategy: {str(e)}")
            raise

    def _get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and test data from session state and perform proper splitting.
        All tickers are split at the same time point to maintain pair relationships.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test price data
        """
        if 'pivot_prices' in st.session_state:
            # Use the pivoted data that's already been prepared
            prices = st.session_state['pivot_prices']
        elif 'historical_data' in st.session_state:
            # Create pivot table from historical data
            data = st.session_state['historical_data']
            data['Date'] = pd.to_datetime(data['Date'])

            prices = data.pivot(
                index='Date',
                columns='Symbol',
                values='Adj_Close'
            )
        else:
            raise ValueError("No historical data found in session state")

        prices = prices.sort_index()
        prices = prices.ffill().bfill()

        if ('strategy_params' in st.session_state and
                isinstance(st.session_state['strategy_params'], dict) and
                'train_size' in st.session_state['strategy_params']):
            train_days = st.session_state['strategy_params']['train_size']
            train_size = min(train_days, int(len(prices) * 0.8))
        else:
            train_size = int(len(prices) * 0.7)

        min_train_size = 100
        min_test_size = 20

        if len(prices) < (min_train_size + min_test_size):
            raise ValueError(
                f"Insufficient data. Need at least {min_train_size + min_test_size} "
                f"periods, but only have {len(prices)}"
            )

        train_data = prices.iloc[:train_size]
        test_data = prices.iloc[train_size:]

        logger.info(
            f"Data split created - Training: {len(train_data)} periods "
            f"({train_data.index[0]} to {train_data.index[-1]}), "
            f"Testing: {len(test_data)} periods "
            f"({test_data.index[0]} to {test_data.index[-1]})"
        )

        missing_data = {
            ticker: train_data[ticker].isna().sum() + test_data[ticker].isna().sum()
            for ticker in prices.columns
        }

        if any(missing_data.values()):
            missing_tickers = {k: v for k, v in missing_data.items() if v > 0}
            logger.warning(f"Missing data in tickers: {missing_tickers}")
            st.warning(
                "Some tickers have missing data. This might affect model performance. "
                f"Affected tickers: {list(missing_tickers.keys())}"
            )

        return train_data, test_data

    def _display_backtest_results(self):
        """Display comprehensive backtest results."""

        if ('backtest_results' not in st.session_state or
                st.session_state['backtest_results'] is None or
                'metrics' not in st.session_state['backtest_results']):
            return

        results = st.session_state['backtest_results']

        if results is None or 'metrics' not in results:
            st.error("No valid backtest results available. Please run the backtest first.")
            return

        self._display_summary_metrics(results['metrics'])

        self._display_performance_charts(
            results['equity_curve'],
            results['trades']
        )

        self._display_trade_analysis(results['trades'])

        self._display_risk_analysis(
            results['equity_curve'],
            results['trades']
        )

    def _display_summary_metrics(self, metrics: Dict):
        """
        Display summary performance metrics.

        Args:
            metrics (Dict): Performance metrics to display
        """
        st.subheader("Performance Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Total Return",
            f"{metrics['Total Return']:.1%}"
        )
        col2.metric(
            "Sharpe Ratio",
            f"{metrics['Sharpe Ratio']:.2f}"
        )
        col3.metric(
            "Max Drawdown",
            f"{metrics['Max Drawdown']:.1%}"
        )
        col4.metric(
            "Win Rate",
            f"{metrics['Win Rate']:.1%}"
        )

    def _display_performance_charts(self, equity_curve: pd.Series, trades: pd.DataFrame):
        """
        Display performance visualization charts.

        Args:
            equity_curve (pd.Series): Portfolio equity curve
            trades (pd.DataFrame): Trade history
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Portfolio Value", "Drawdown"],
            vertical_spacing=0.12
        )

        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name="Portfolio Value",
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        for trade_type in ['ENTRY', 'EXIT']:
            trade_points = trades[trades['Action'] == trade_type]
            fig.add_trace(
                go.Scatter(
                    x=trade_points['Date'],
                    y=[equity_curve[d] for d in trade_points['Date']],
                    mode='markers',
                    name=f'{trade_type}s',
                    marker=dict(
                        size=8,
                        color='green' if trade_type == 'ENTRY' else 'red'
                    )
                ),
                row=1, col=1
            )

        drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                line=dict(color='red'),
                fill='tozeroy'
            ),
            row=2, col=1
        )

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig)

    def _display_trade_analysis(self, trades: pd.DataFrame):
        """
        Display trade analysis.

        Args:
            trades (pd.DataFrame): Trade history to analyze
        """
        st.subheader("Trade Analysis")

        trade_stats = trades.groupby('Pair').agg({
            'PnL': ['count', 'mean', 'sum'],
            'Duration': 'mean',
            'Cost': 'sum'
        })
        trade_stats.columns = [
            'Number of Trades',
            'Average PnL',
            'Total PnL',
            'Average Duration',
            'Total Costs'
        ]
        st.dataframe(trade_stats)

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=trades['PnL'],
                name='PnL Distribution',
                nbinsx=50,
                opacity=0.7
            )
        )
        fig.update_layout(
            title="Trade PnL Distribution",
            xaxis_title="PnL",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig)

    def _display_risk_analysis(self, equity_curve: pd.Series, trades: pd.DataFrame):
        """
        Display risk analysis metrics and visualizations.

        Args:
            equity_curve (pd.Series): Portfolio equity curve
            trades (pd.DataFrame): Trade history
        """
        st.subheader("Risk Analysis")

        returns = equity_curve.pct_change().dropna()
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        rolling_sharpe = (returns.rolling(window=63).mean() * 252) / \
                         (returns.rolling(window=63).std() * np.sqrt(252))
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Annualized Volatility",
                f"{returns.std() * np.sqrt(252):.1%}"
            )
        with col2:
            st.metric(
                "Value at Risk (95%)",
                f"{abs(var_95):.1%}"
            )
        with col3:
            st.metric(
                "Value at Risk (99%)",
                f"{abs(var_99):.1%}"
            )
        with col4:
            st.metric(
                "Average Position Size",
                f"{trades['Quantity'].abs().mean():.2f}"
            )

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Rolling Volatility",
                "Rolling Sharpe Ratio",
                "Return Distribution",
                "Position Exposure"
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name="21d Volatility"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="63d Sharpe"
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name="Returns",
                nbinsx=50
            ),
            row=2, col=1
        )

        daily_exposure = trades.groupby('Date')['Quantity'].sum().abs()
        fig.add_trace(
            go.Scatter(
                x=daily_exposure.index,
                y=daily_exposure.values,
                name="Exposure"
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig)

    def _analyze_pair_performance(self, trades: pd.DataFrame):
        """
        Analyze performance of individual pairs.

        Args:
            trades (pd.DataFrame): Trade history to analyze
        """
        st.subheader("Pair Performance Analysis")

        pair_metrics = trades.groupby('Pair').agg({
            'PnL': [
                'count',
                'sum',
                'mean',
                lambda x: (x > 0).mean(),
                lambda x: x[x > 0].mean(),
                lambda x: x[x < 0].mean()
            ],
            'Duration': ['mean', 'min', 'max'],
            'Cost': 'sum'
        })

        pair_metrics.columns = [
            'Number of Trades',
            'Total PnL',
            'Average PnL',
            'Win Rate',
            'Average Win',
            'Average Loss',
            'Avg Duration',
            'Min Duration',
            'Max Duration',
            'Total Costs'
        ]

        st.dataframe(pair_metrics)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pair_metrics['Win Rate'],
                y=pair_metrics['Average PnL'],
                mode='markers+text',
                text=pair_metrics.index,
                textposition="top center",
                name='Pairs'
            )
        )

        fig.update_layout(
            title="Pair Performance Comparison",
            xaxis_title="Win Rate",
            yaxis_title="Average PnL",
            height=600
        )

        st.plotly_chart(fig)

    def _analyze_market_conditions(self, trades: pd.DataFrame, market_data: pd.DataFrame):
        """
        Analyze strategy performance under different market conditions.

        Args:
            trades (pd.DataFrame): Trade history
            market_data (pd.DataFrame): Market data for analysis
        """
        st.subheader("Market Condition Analysis")

        market_returns = market_data.pct_change()
        market_vol = market_returns.rolling(window=21).std()

        conditions = pd.qcut(market_vol, q=3, labels=['Low Vol', 'Med Vol', 'High Vol'])
        trades['Market_Regime'] = trades['Date'].map(conditions)

        regime_performance = trades.groupby('Market_Regime').agg({
            'PnL': ['count', 'sum', 'mean', lambda x: (x > 0).mean()],
            'Duration': 'mean'
        })

        regime_performance.columns = [
            'Number of Trades',
            'Total PnL',
            'Average PnL',
            'Win Rate',
            'Avg Duration'
        ]

        st.dataframe(regime_performance)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "PnL by Market Regime",
                "Win Rate by Market Regime"
            ]
        )

        fig.add_trace(
            go.Box(
                y=trades['PnL'],
                x=trades['Market_Regime'],
                name='PnL Distribution'
            ),
            row=1, col=1
        )

        win_rates = trades.groupby('Market_Regime')['PnL'].apply(
            lambda x: (x > 0).mean()
        )

        fig.add_trace(
            go.Bar(
                x=win_rates.index,
                y=win_rates.values,
                name='Win Rate'
            ),
            row=1, col=2
        )

        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig)

    def _analyze_strategy_behavior(self, trades: pd.DataFrame, equity_curve: pd.Series):
        """
        Analyze strategy trading behavior and patterns.

        Args:
            trades (pd.DataFrame): Trade history
            equity_curve (pd.Series): Portfolio equity curve
        """
        st.subheader("Strategy Behavior Analysis")

        monthly_activity = trades.groupby(
            pd.Grouper(key='Date', freq='M')
        ).agg({
            'PnL': ['count', 'sum', 'mean'],
            'Cost': 'sum'
        })

        monthly_activity.columns = [
            'Number of Trades',
            'Total PnL',
            'Average PnL',
            'Total Costs'
        ]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Trading Activity Over Time",
                "PnL vs Trade Duration",
                "Trade Size Distribution",
                "Hour of Day Analysis"
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_activity.index,
                y=monthly_activity['Number of Trades'],
                name='Number of Trades'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=trades['Duration'],
                y=trades['PnL'],
                mode='markers',
                name='Trades'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(
                x=trades['Quantity'].abs(),
                name='Trade Size'
            ),
            row=2, col=1
        )

        if 'Date' in trades.columns:
            trades['Hour'] = trades['Date'].dt.hour
            hourly_pnl = trades.groupby('Hour')['PnL'].mean()

            fig.add_trace(
                go.Bar(
                    x=hourly_pnl.index,
                    y=hourly_pnl.values,
                    name='Hourly PnL'
                ),
                row=2, col=2
            )

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig)

    def export_results(self):
        """Export backtest results to Excel."""
        if 'backtest_results' not in st.session_state:
            st.warning("No backtest results to export.")
            return

        results = st.session_state['backtest_results']

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results['equity_curve'].to_excel(
                writer,
                sheet_name='Equity_Curve'
            )

            results['trades'].to_excel(
                writer,
                sheet_name='Trades'
            )

            pd.DataFrame([results['metrics']]).to_excel(
                writer,
                sheet_name='Metrics'
            )

            pd.DataFrame([results['parameters']]).to_excel(
                writer,
                sheet_name='Parameters'
            )

        buffer.seek(0)
        st.download_button(
            label="Download Results",
            data=buffer,
            file_name="backtest_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    @staticmethod
    def _get_returns_data() -> pd.DataFrame:
        """
        Get returns data from session state.

        Returns:
            pd.DataFrame: Returns data for analysis

        Raises:
            ValueError: If no historical data found
        """
        if 'historical_data' not in st.session_state:
            raise ValueError("No historical data found in session state")

        data = st.session_state['historical_data']
        prices = data.pivot(
            index='Date',
            columns='Symbol',
            values='Adj_Close'
        )
        returns = prices.pct_change().dropna()
        return returns