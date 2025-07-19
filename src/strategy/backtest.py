import pandas as pd
import numpy as np
from typing import Any, Dict, List, Set, Tuple
from tqdm import tqdm

from config.logging_config import logger
from src.strategy.base import BaseStrategy
from src.strategy.risk import PortfolioRiskManager
from execution.oms_and_broker_connect import SimulatedBroker, Order, OrderSide, OrderType
from src.utils.metrics import calculate_all_metrics


class Backtester:
    """
    A high-fidelity, event-driven backtesting engine that simulates trading strategies
    with a realistic broker and portfolio risk management.
    """

    def __init__(self,
                 strategy: BaseStrategy,
                 historical_data: pd.DataFrame,  # Full historical data available
                 initial_capital: float = 100000.0,
                 transaction_cost_pct: float = 0.001,
                 slippage_pct: float = 0.0005):
        """
        Initializes the Backtester.

        Args:
            strategy (BaseStrategy): The trading strategy to be backtested.
            historical_data (pd.DataFrame): DataFrame containing all historical market data
                                            (e.g., 'Date', 'Symbol', 'Adj_Close', 'Volume').
            initial_capital (float): Starting capital for the backtest.
            transaction_cost_pct (float): Percentage cost per trade (e.g., 0.001 for 0.1%).
            slippage_pct (float): Percentage slippage per trade.
        """
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("Strategy must be an instance of BaseStrategy.")
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
            raise ValueError("Historical data must be a non-empty Pandas DataFrame.")
        if not all(col in historical_data.columns for col in ['Date', 'Symbol', 'Adj_Close']):
            raise ValueError("Historical data must contain 'Date', 'Symbol', and 'Adj_Close' columns.")
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if not (0 <= transaction_cost_pct < 1):
            raise ValueError("Transaction cost percentage must be between 0 and 1.")
        if not (0 <= slippage_pct < 1):
            raise ValueError("Slippage percentage must be between 0 and 1.")

        self.strategy = strategy
        self.historical_data_raw = historical_data.copy()  # Store a copy of raw data
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct

        # Prepare pivoted price data for faster lookups during backtest
        # Ffill and bfill handle NaNs that might appear due to missing data for some symbols
        self.prices_pivot = self.historical_data_raw.pivot(index='Date', columns='Symbol',
                                                           values='Adj_Close').ffill().bfill()
        if self.prices_pivot.empty:
            raise ValueError("Pivoted price data is empty. Check historical_data content.")

        # Ensure dates are sorted for sequential processing
        self.dates = sorted(self.prices_pivot.index.unique())
        if not self.dates:
            raise ValueError("No unique dates found in historical data after pivoting.")

        # Core Components
        self.broker = SimulatedBroker(initial_capital, transaction_cost_pct)
        self.risk_manager = PortfolioRiskManager(initial_capital)  # Risk manager gets initial capital

        # Backtest Results & State
        self.equity_curve = pd.Series(index=self.dates, dtype=float)
        self.trade_log: List[Dict] = []
        logger.info(f"Backtester initialized for strategy '{self.strategy.name}' with {len(self.dates)} days of data.")

    def run(self) -> Dict[str, Any]:
        """
        Executes the backtest simulation day-by-day.

        Returns:
            A dictionary containing the backtest results: equity curve, performance metrics, and trade log.
        """
        logger.info(f"Running backtest for strategy '{self.strategy.name}' from {self.dates[0]} to {self.dates[-1]}...")

        # Initially fit the strategy with all available data.
        # For walk-forward optimization, `strategy.fit` will be called repeatedly on smaller train folds.
        # This initial call covers cases where strategy is run standalone (not via WFO/CV).
        self.strategy.fit(self.prices_pivot)
        logger.info(f"Strategy '{self.strategy.name}' initial fitting complete.")

        # Initialize progress bar for daily loop
        for date in tqdm(self.dates, desc="Backtesting Progress"):
            try:
                # 1. Get current market prices for this timestamp
                current_day_prices = self.prices_pivot.loc[date].to_dict()

                # Check if all values are NaN for the current day. If so, skip.
                if all(pd.isna(v) for v in current_day_prices.values()):
                    logger.warning(f"All prices for {date} are NaN. Skipping this day.")
                    self.equity_curve[date] = self.broker.get_portfolio_value()
                    continue

                # Prepare current prices dict for broker (includes date for trade logging)
                current_prices_for_broker = current_day_prices.copy()
                current_prices_for_broker['date'] = date

                # 2. Update broker with current prices (for mark-to-market and order execution)
                self.broker.update_market_data(current_prices_for_broker)

                # 3. Create portfolio context for the strategy
                portfolio_context = {
                    'date': date,
                    'cash': self.broker.get_cash_balance(),
                    'equity': self.broker.get_portfolio_value(),
                    'positions': self.broker.get_all_positions(),  # Detailed position objects
                    'open_pair_symbols': self._get_open_asset_symbols_from_positions()
                    # Returns Set[str] of individual assets
                }

                # 4. Prepare data window for strategy.
                # Crucially, strategy only sees data up to current date for signal generation.
                data_window_for_strategy = self.prices_pivot.loc[:date]
                if data_window_for_strategy.empty:
                    logger.warning(f"Data window is empty for date {date}. Skipping signal generation.")
                    self.equity_curve[date] = self.broker.get_portfolio_value()
                    continue

                # Re-fit strategy dynamically if it's an adaptive strategy (e.g., DynamicPairsStrategy)
                # or if it's configured to retrain on a rolling basis.
                # The strategy's `fit` method must be idempotent or handle repeated calls gracefully.
                self.strategy.fit(data_window_for_strategy)  # Strategy re-fits/adapts based on data up to `date`

                # Get raw signals from the strategy
                raw_signals = self.strategy.generate_signals(data_window_for_strategy, portfolio_context)
                logger.debug(f"Date {date}: Strategy '{self.strategy.name}' generated {len(raw_signals)} raw signals.")

                # 5. Risk manager converts signals to target dollar positions
                # It needs the full historical data up to this point for its rolling calculations
                target_positions = self.risk_manager.calculate_target_positions(
                    raw_signals,
                    current_day_prices,  # Pass only symbol -> price dict
                    data_window_for_strategy,  # Pass historical data relevant for risk calculations
                    portfolio_context['equity']
                )
                logger.debug(f"Date {date}: Risk manager calculated {len(target_positions)} target positions.")

                # 6. Generate orders to align current portfolio with target
                self.generate_rebalancing_orders(target_positions, current_day_prices)

                # 7. Broker processes orders and generates fills based on current prices
                self.broker.execute_orders()  # Slippage is handled internally by broker
                logger.debug(f"Date {date}: Broker executed orders.")

                # 8. Record daily equity
                self.equity_curve[date] = self.broker.get_portfolio_value()
                logger.debug(f"Date {date}: Daily equity updated to ${self.equity_curve[date]:,.2f}.")

            except Exception as e:
                logger.error(f"Error during backtest on date {date}: {e}. Skipping to next day.")
                # If an error occurs, record current equity (or last valid) and continue
                self.equity_curve[date] = self.broker.get_portfolio_value()  # Maintain continuity

        # Finalize results after the loop
        self.trade_log = self.broker.get_trade_log()
        metrics = calculate_all_metrics(self.equity_curve)

        logger.info("Backtest complete.")
        logger.info(f"Final Portfolio Value: ${self.equity_curve.iloc[-1]:,.2f}")
        logger.info(f"Total Return: {metrics.get('Total Return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")

        return {
            'equity_curve': self.equity_curve,
            'metrics': metrics,
            'trades': self.trade_log
        }

    def generate_rebalancing_orders(self, target_positions: Dict[Tuple[str, str], Dict[str, float]],
                                    current_prices: Dict[str, float]):
        """
        Compares current positions with target positions and generates market orders to rebalance the portfolio.
        Orders are placed for individual assets.
        """
        current_holdings = self.broker.get_all_positions_in_dollars()

        flat_target_dollars: Dict[str, float] = {}
        for pair_legs_dict in target_positions.values():
            for asset, value in pair_legs_dict.items():
                flat_target_dollars[asset] = flat_target_dollars.get(asset, 0.0) + value

        all_assets = set(current_holdings.keys()) | set(flat_target_dollars.keys())

        for asset in all_assets:
            current_value = current_holdings.get(asset, 0.0)
            target_value = flat_target_dollars.get(asset, 0.0)
            trade_value_dollars = target_value - current_value

            if abs(trade_value_dollars) < 1.0:  # Minimum trade value to avoid dust
                logger.debug(f"Skipping trade for {asset}: value {trade_value_dollars:.2f} too small.")
                continue

            current_price = current_prices.get(asset)
            if current_price is None or current_price <= 0:
                logger.warning(f"Cannot trade {asset}: price is {current_price}. Skipping order.")
                continue

            quantity = trade_value_dollars / current_price
            order_side = OrderSide.BUY if quantity > 0 else OrderSide.SELL

            # Round quantity to nearest whole share for realism
            shares_to_trade = int(round(abs(quantity)))

            if shares_to_trade > 0:  # Ensure we don't place orders for 0 shares after rounding
                order = Order(
                    symbol=asset,
                    side=order_side,
                    quantity=shares_to_trade,
                    order_type=OrderType.MARKET  # All orders are market orders for simplicity
                )
                self.broker.place_order(order)
                logger.debug(
                    f"Placed {order_side.name} order for {shares_to_trade} shares of {asset} (Value: {trade_value_dollars:.2f}).")
            else:
                logger.debug(f"Trade quantity for {asset} is zero after rounding. Skipping order.")

    def _get_open_asset_symbols_from_positions(self) -> Set[str]:
        """
        Helper to get symbols of assets that currently have open positions.
        This provides context for strategies to check if a specific asset (or leg of a pair) is held.
        """
        return {pos.symbol for pos in self.broker.get_all_positions()}