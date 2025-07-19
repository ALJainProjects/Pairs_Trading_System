import pandas as pd
from typing import Any, Dict, List
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
                 historical_data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 transaction_cost_pct: float = 0.001,
                 slippage_pct: float = 0.0005):

        self.strategy = strategy
        self.historical_data = historical_data
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct

        # Core Components
        self.broker = SimulatedBroker(initial_capital, transaction_cost_pct)
        self.risk_manager = PortfolioRiskManager(initial_capital)

        # Data & State
        self.dates = sorted(self.historical_data['Date'].unique())
        self.prices_pivot = self.historical_data.pivot(index='Date', columns='Symbol', values='Adj_Close').ffill().bfill()
        self.equity_curve = pd.Series(index=self.dates, dtype=float)
        self.trade_log: List[Dict] = []

    def run(self) -> Dict[str, Any]:
        """
        Executes the backtest.

        Returns:
            A dictionary containing the results: equity curve, performance metrics, and trade log.
        """
        logger.info(f"Running backtest for strategy '{self.strategy.name}'...")

        # Fit the strategy on the full historical data (for ML models, etc.)
        self.strategy.fit(self.prices_pivot)

        for date in tqdm(self.dates, desc="Backtesting Progress"):
            # 1. Get current market prices for this timestamp
            current_prices = self.prices_pivot.loc[date].to_dict()
            current_prices['date'] = date # Pass date to broker for trade log

            # 2. Update broker with current prices (for mark-to-market)
            self.broker.update_market_data(current_prices)

            # 3. Create portfolio context for the strategy
            portfolio_context = {
                'date': date,
                'cash': self.broker.get_cash_balance(),
                'equity': self.broker.get_portfolio_value(),
                'positions': self.broker.get_all_positions()
            }

            # 4. Get raw signals from the strategy
            data_window = self.prices_pivot.loc[:date]
            raw_signals = self.strategy.generate_signals(data_window, portfolio_context)

            # 5. Let the risk manager convert signals to target dollar positions
            target_positions = self.risk_manager.calculate_target_positions(
                raw_signals,
                current_prices,
                self.prices_pivot.loc[:date],
                portfolio_context['equity']
            )

            # 6. Generate orders to align current portfolio with target
            self.generate_rebalancing_orders(target_positions, current_prices)

            # 7. Broker processes orders and generates fills based on current prices
            self.broker.execute_orders(self.slippage_pct)

            # 8. Record daily equity
            self.equity_curve[date] = self.broker.get_portfolio_value()

        # Finalize results after the loop
        self.trade_log = self.broker.get_trade_log()
        metrics = calculate_all_metrics(self.equity_curve)

        logger.info("Backtest complete.")
        logger.info(f"Final Portfolio Value: ${self.equity_curve.iloc[-1]:,.2f}")
        logger.info(f"Total Return: {metrics.get('Total Return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")

        return {
            'equity_curve': self.equity_curve,
            'metrics': metrics,
            'trades': self.trade_log
        }

    def generate_rebalancing_orders(self, target_positions: Dict, current_prices: Dict):
        """
        Compares current positions with targets and generates orders to rebalance.
        """
        current_holdings = self.broker.get_all_positions_in_dollars()

        flat_target = {}
        for pair, legs in target_positions.items():
            for asset, value in legs.items():
                flat_target[asset] = flat_target.get(asset, 0) + value

        all_assets = set(current_holdings.keys()) | set(flat_target.keys())

        for asset in all_assets:
            current_value = current_holdings.get(asset, 0)
            target_value = flat_target.get(asset, 0)

            trade_value = target_value - current_value

            if abs(trade_value) > 1.0: # Minimum trade value to avoid dust
                quantity = trade_value / current_prices[asset]
                order_side = OrderSide.BUY if quantity > 0 else OrderSide.SELL

                order = Order(
                    symbol=asset,
                    side=order_side,
                    quantity=abs(quantity),
                    order_type=OrderType.MARKET
                )
                self.broker.place_order(order)