"""
Enhanced Backtesting Module with Multi-Pair Support

Features:
1. Multiple pair trading support
2. Position tracking per pair
3. Enhanced risk management for portfolio level metrics
4. Transaction cost handling per leg
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from pandas import DataFrame

from src.strategy.risk import PairRiskManager
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.logging_config import logger

class MultiPairBacktester:
    """Enhanced backtester with support for multiple pairs."""

    def __init__(
            self,
            strategy: Any,
            returns: pd.DataFrame,
            initial_capital: float = 100000,
            risk_manager: Optional[PairRiskManager] = None,
            transaction_cost: float = 0.001,
            max_pairs: Optional[int] = None
    ):
        """
        Initialize the backtester.

        Args:
            strategy: Trading strategy instance
            returns: DataFrame of asset returns
            initial_capital: Starting capital
            risk_manager: Risk manager instance
            transaction_cost: Transaction cost per trade
            max_pairs: Maximum concurrent pairs to trade
        """
        self.strategy = strategy
        self.returns = returns
        self.initial_capital = initial_capital
        self.risk_manager = risk_manager or PairRiskManager()
        self.transaction_cost = transaction_cost
        self.max_pairs = max_pairs

        self.equity_curve = pd.Series(dtype=float)
        self.current_capital = initial_capital
        self.active_pairs = {}

        self.pair_performance = {}
        self.trade_history = pd.DataFrame(
            columns=['Date', 'Pair', 'Action', 'Quantity', 'Price', 'Cost']
        )

        self.asset_prices = self._initialize_asset_prices()

    def _initialize_asset_prices(self) -> pd.DataFrame:
        """Initialize asset prices from returns."""
        logger.info("Initializing asset prices")

        return self.returns

    def _validate_pair_trade(self, pair: Tuple[str, str],
                            quantity: float) -> bool:
        """
        Validate if a pair trade is allowable.

        Args:
            pair: Tuple of (asset1, asset2)
            quantity: Trade quantity

        Returns:
            bool: Whether trade is valid
        """
        if self.max_pairs and len(self.active_pairs) >= self.max_pairs:
            if pair not in self.active_pairs:
                logger.warning(f"Max pairs ({self.max_pairs}) reached")
                return False

        required_capital = self._calculate_required_capital(pair, quantity)
        if required_capital > self.current_capital:
            logger.warning(f"Insufficient capital for {pair}")
            return False

        return True

    def _calculate_required_capital(self, pair: Tuple[str, str],
                                  quantity: float) -> float:
        """Calculate required capital for a pair trade."""
        asset1, asset2 = pair
        price1 = self.asset_prices[asset1].iloc[-1]
        price2 = self.asset_prices[asset2].iloc[-1]

        trade_value = (price1 + price2) * quantity
        costs = trade_value * self.transaction_cost

        return trade_value + costs

    def run_backtest(self) -> pd.Series:
        """
        Execute the backtest.

        Returns:
            Series of portfolio values
        """
        logger.info("Starting backtest")
        self.equity_curve = pd.Series(index=self.returns.index, dtype=float)
        portfolio_value = self.initial_capital

        self.equity_curve.iloc[0] = portfolio_value

        for i, current_date in enumerate(self.returns.index[1:], 1):
            historical_data = self.returns.iloc[:i + 1]

            if hasattr(self.strategy, 'predict_signals'):
                features = self.strategy.prepare_features(historical_data)
                predictions = self.strategy.predict_signals(features)
                signals = predictions.loc[current_date:current_date]
            else:
                signals = self.strategy.generate_signals(historical_data)

            portfolio_value = self._process_signals_and_update(
                signals,
                portfolio_value,
                current_date
            )

            self.equity_curve.loc[current_date] = portfolio_value

            if self._check_risk_constraints(current_date):
                logger.warning("Risk constraints violated")
                break

        logger.info("Backtest completed")
        return self.equity_curve

    def _process_signals_and_update(self,
                                    signals: pd.DataFrame,
                                    portfolio_value: float,
                                    current_date: pd.Timestamp) -> float:
        """Process signals and update portfolio."""
        current_prices = self.asset_prices.loc[current_date]

        if isinstance(signals, pd.DataFrame) and 'predicted_signal' in signals.columns:
            for pair in self.strategy.pairs:
                if isinstance(pair, tuple) and len(pair) == 2:
                    asset1, asset2 = pair
                    if asset1 not in current_prices or asset2 not in current_prices:
                        continue

                    pair_signals = signals[signals['Pair'] == f"{asset1}/{asset2}"]
                    if not pair_signals.empty:
                        signal_value = float(pair_signals.iloc[-1]['predicted_signal'])
                        portfolio_value = self._process_pair_signal(
                            pair,
                            signal_value,
                            portfolio_value,
                            current_date
                        )
        else:
            for pair, signal_row in signals.items():
                if isinstance(pair, tuple) and len(pair) == 2:
                    asset1, asset2 = pair
                    if asset1 not in current_prices or asset2 not in current_prices:
                        continue

                    signal_value = float(signal_row.loc[current_date])
                    portfolio_value = self._process_pair_signal(
                        pair,
                        signal_value,
                        portfolio_value,
                        current_date
                    )

        portfolio_value = self._update_active_positions(
            portfolio_value,
            current_date
        )

        return portfolio_value

    def _process_pair_signal(self,
                           pair: Tuple[str, str],
                           signal: float,
                           portfolio_value: float,
                           current_date: pd.Timestamp) -> float:
        """Process signal for a single pair."""
        asset1, asset2 = pair

        if pair in self.active_pairs:
            existing_signal = self.active_pairs[pair]['signal']
            if signal == 0 or (signal * existing_signal < 0):
                portfolio_value = self._close_pair_position(
                    pair,
                    portfolio_value,
                    current_date
                )

        if signal != 0 and pair not in self.active_pairs:
            quantity = self._calculate_position_size(
                pair,
                portfolio_value
            )

            if self._validate_pair_trade(pair, quantity):
                portfolio_value = self._open_pair_position(
                    pair,
                    signal,
                    quantity,
                    portfolio_value,
                    current_date
                )

        return portfolio_value

    def _calculate_position_size(self,
                               pair: Tuple[str, str],
                               portfolio_value: float) -> float:
        """Calculate position size for a pair."""
        asset1, asset2 = pair
        price1 = self.asset_prices[asset1].iloc[-1]
        price2 = self.asset_prices[asset2].iloc[-1]

        if hasattr(self.risk_manager, 'calculate_position_size'):
            return self.risk_manager.calculate_position_size(
                portfolio_value,
                price1 + price2,
                self.transaction_cost
            )

        risk_per_trade = 0.01
        return (portfolio_value * risk_per_trade) / (price1 + price2)

    def _open_pair_position(self,
                          pair: Tuple[str, str],
                          signal: float,
                          quantity: float,
                          portfolio_value: float,
                          current_date: pd.Timestamp) -> float:
        """Open a new pair position."""
        asset1, asset2 = pair
        price1 = self.asset_prices[asset1].loc[current_date]
        price2 = self.asset_prices[asset2].loc[current_date]

        trade_value = (price1 + price2) * quantity
        costs = trade_value * self.transaction_cost

        for asset, price in [(asset1, price1), (asset2, price2)]:
            self.trade_history = pd.concat([
                self.trade_history,
                pd.DataFrame([{
                    'Date': current_date,
                    'Pair': f"{asset1}/{asset2}",
                    'Asset': asset,
                    'Action': 'BUY' if signal > 0 else 'SELL',
                    'Quantity': quantity * signal,
                    'Price': price,
                    'Cost': costs / 2
                }])
            ], ignore_index=True)

        self.active_pairs[pair] = {
            'signal': signal,
            'quantity': quantity,
            'entry_date': current_date,
            'entry_prices': (price1, price2)
        }

        return portfolio_value - costs

    def _close_pair_position(self,
                           pair: Tuple[str, str],
                           portfolio_value: float,
                           current_date: pd.Timestamp) -> float:
        """Close an existing pair position."""
        if pair not in self.active_pairs:
            return portfolio_value

        asset1, asset2 = pair
        position = self.active_pairs[pair]

        price1 = self.asset_prices[asset1].loc[current_date]
        price2 = self.asset_prices[asset2].loc[current_date]

        entry_price1, entry_price2 = position['entry_prices']
        signal = position['signal']
        quantity = position['quantity']

        pnl = signal * quantity * (
            (price1 - entry_price1) - (price2 - entry_price2)
        )

        close_value = (price1 + price2) * quantity
        costs = close_value * self.transaction_cost

        for asset, price in [(asset1, price1), (asset2, price2)]:
            self.trade_history = pd.concat([
                self.trade_history,
                pd.DataFrame([{
                    'Date': current_date,
                    'Pair': f"{asset1}/{asset2}",
                    'Asset': asset,
                    'Action': 'SELL' if signal > 0 else 'BUY',
                    'Quantity': -quantity * signal,
                    'Price': price,
                    'Cost': costs / 2
                }])
            ], ignore_index=True)

        self.pair_performance[pair] = self.pair_performance.get(pair, [])
        self.pair_performance[pair].append({
            'Entry_Date': position['entry_date'],
            'Exit_Date': current_date,
            'PnL': pnl,
            'Return': pnl / (close_value - costs)
        })

        del self.active_pairs[pair]

        return portfolio_value + pnl - costs

    def _update_active_positions(self,
                               portfolio_value: float,
                               current_date: pd.Timestamp) -> float:
        """Update all active positions."""
        if not self.active_pairs:
            return portfolio_value

        total_pnl = 0
        for pair in list(self.active_pairs.keys()):
            asset1, asset2 = pair
            position = self.active_pairs[pair]

            if (asset1 not in self.asset_prices.columns or
                asset2 not in self.asset_prices.columns):
                logger.warning(f"Assets for pair {pair} no longer available")
                portfolio_value = self._close_pair_position(
                    pair,
                    portfolio_value,
                    current_date
                )
                continue

            price1 = self.asset_prices[asset1].loc[current_date]
            price2 = self.asset_prices[asset2].loc[current_date]
            entry_price1, entry_price2 = position['entry_prices']

            pair_pnl = position['signal'] * position['quantity'] * (
                (price1 - entry_price1) - (price2 - entry_price2)
            )
            total_pnl += pair_pnl

        return portfolio_value + total_pnl

    def _check_risk_constraints(self, current_date: pd.Timestamp) -> bool:
        """Check if risk constraints are violated."""
        current_equity = self.equity_curve.loc[:current_date]

        if hasattr(self.risk_manager, 'check_drawdown'):
            if self.risk_manager.check_drawdown(current_equity):
                return True

        if hasattr(self.risk_manager, 'check_portfolio_risk'):
            positions = pd.DataFrame([
                {
                    'Pair': f"{p[0]}/{p[1]}",
                    'Signal': v['signal'],
                    'Quantity': v['quantity']
                }
                for p, v in self.active_pairs.items()
            ])
            if self.risk_manager.check_portfolio_risk(current_equity, positions):
                return True

        return False

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating performance metrics")

        returns = self.equity_curve.pct_change().dropna()

        if len(returns) == 0:
            logger.warning("No returns available")
            return self._empty_metrics()

        total_return = (self.equity_curve.iloc[-1] / self.initial_capital) - 1
        annual_return = ((1 + total_return) ** (252 / len(returns))) - 1

        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return / volatility) if volatility != 0 else 0
        max_drawdown = self._calculate_max_drawdown()

        pair_metrics = self._calculate_pair_metrics()

        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Pair Metrics': pair_metrics
        }

        return metrics

    def _empty_metrics(self) -> Dict[str, Union[float, Dict]]:
        """Return empty metrics structure."""
        return {
            'Total Return': 0.0,
            'Annual Return': 0.0,
            'Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0,
            'Pair Metrics': {}
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        peaks = self.equity_curve.expanding().max()
        drawdowns = (self.equity_curve - peaks) / peaks
        return abs(drawdowns.min())

    def _calculate_pair_metrics(self) -> Dict[str, Dict]:
        """Calculate performance metrics for each pair."""
        pair_metrics = {}

        for pair, trades in self.pair_performance.items():
            if not trades:
                continue

            trades_df = pd.DataFrame(trades)

            total_pnl = trades_df['PnL'].sum()
            avg_return = trades_df['Return'].mean()
            win_rate = (trades_df['PnL'] > 0).mean()

            holding_periods = (trades_df['Exit_Date'] -
                             trades_df['Entry_Date']).mean().days

            pair_metrics[f"{pair[0]}/{pair[1]}"] = {
                'Total PnL': total_pnl,
                'Average Return': avg_return,
                'Win Rate': win_rate,
                'Number of Trades': len(trades),
                'Average Holding Period': holding_periods
            }

        return pair_metrics

    def plot_results(self, show_drawdown: bool = True,
                    show_pair_returns: bool = True) -> None:
        """
        Plot comprehensive backtest results.

        Args:
            show_drawdown: Whether to show drawdown subplot
            show_pair_returns: Whether to show individual pair returns
        """
        n_rows = 1 + int(show_drawdown) + int(show_pair_returns)
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=self._get_subplot_titles(
                show_drawdown,
                show_pair_returns
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        if show_drawdown:
            drawdown = (self.equity_curve - self.equity_curve.cummax()) / \
                       self.equity_curve.cummax()
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red')
                ),
                row=2, col=1
            )

        if show_pair_returns and self.pair_performance:
            row = 2 + int(show_drawdown)
            self._add_pair_returns_plot(fig, row)

        fig.update_layout(
            title="Backtest Results",
            height=300 * n_rows,
            showlegend=True,
            template="plotly_white"
        )

        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        if show_pair_returns:
            fig.update_yaxes(
                title_text="Cumulative Return",
                row=2 + int(show_drawdown),
                col=1
            )

        fig.show()

    def _get_subplot_titles(self, show_drawdown: bool,
                           show_pair_returns: bool) -> List[str]:
        """Get subplot titles based on what's being shown."""
        titles = ["Portfolio Equity Curve"]
        if show_drawdown:
            titles.append("Portfolio Drawdown")
        if show_pair_returns:
            titles.append("Individual Pair Returns")
        return titles

    def _add_pair_returns_plot(self, fig: go.Figure, row: int) -> None:
        """Add individual pair returns to the plot."""
        for pair, trades in self.pair_performance.items():
            if not trades:
                continue

            trades_df = pd.DataFrame(trades)

            cum_returns = (1 + trades_df['Return']).cumprod()

            fig.add_trace(
                go.Scatter(
                    x=trades_df['Exit_Date'],
                    y=cum_returns,
                    mode='lines',
                    name=f"{pair[0]}/{pair[1]}",
                ),
                row=row, col=1
            )

    def generate_report(self, output_file: Optional[str] = None) -> dict[str, DataFrame | Any]:
        """
        Generate comprehensive backtest report.

        Args:
            output_file: Optional file path to save report

        Returns:
            DataFrame containing all metrics
        """
        metrics = self.calculate_performance_metrics()

        basic_metrics = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annual Return',
                'Volatility',
                'Sharpe Ratio',
                'Max Drawdown'
            ],
            'Value': [
                f"{metrics['Total Return']:.2%}",
                f"{metrics['Annual Return']:.2%}",
                f"{metrics['Volatility']:.2%}",
                f"{metrics['Sharpe Ratio']:.2f}",
                f"{metrics['Max Drawdown']:.2%}"
            ]
        })

        pair_metrics_list = []
        for pair, pair_metrics in metrics['Pair Metrics'].items():
            metrics_dict = {'Pair': pair}
            metrics_dict.update(pair_metrics)
            pair_metrics_list.append(metrics_dict)

        pair_metrics_df = pd.DataFrame(pair_metrics_list)

        trade_summary = self.trade_history.groupby('Pair').agg({
            'Date': ['count', 'min', 'max'],
            'Cost': 'sum'
        }).round(2)
        trade_summary.columns = ['Number of Trades', 'First Trade',
                               'Last Trade', 'Total Costs']

        report = {
            'Basic Metrics': basic_metrics,
            'Pair Metrics': pair_metrics_df,
            'Trade Summary': trade_summary
        }

        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                basic_metrics.to_excel(writer, sheet_name='Basic Metrics',
                                     index=False)
                pair_metrics_df.to_excel(writer, sheet_name='Pair Metrics',
                                       index=False)
                trade_summary.to_excel(writer, sheet_name='Trade Summary')
                self.trade_history.to_excel(writer, sheet_name='Trade History',
                                          index=False)

        return report