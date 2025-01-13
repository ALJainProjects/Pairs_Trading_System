import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.strategy.backtest import MultiPairBacktester
from src.strategy.risk import PairRiskManager
from src.strategy.pairs_strategy_SL import StatisticalPairsTrader
from src.strategy.pairs_strategy_ML import PairsTraderML
from src.strategy.pairs_strategy_DL import PairsTradingDL


class EnhancedStrategyBuilder:
    """Enhanced strategy building component with multiple strategy types."""

    def __init__(self):
        """Initialize the strategy builder with a risk manager."""
        self.risk_manager = PairRiskManager()

    def render(self):
        """Render the strategy builder interface."""
        st.header("Strategy Builder")

        if 'selected_pairs' not in st.session_state:
            st.warning("Please select pairs first in the Pair Analysis section.")
            return

        # Strategy selection and configuration
        strategy_type = self._render_strategy_selection()

        # Risk management configuration
        risk_params = self._render_risk_management()

        # Backtesting configuration
        backtest_params = self._render_backtest_config()

        # Create and run strategy
        if st.button("Run Backtest"):
            self._run_backtest(
                strategy_type=strategy_type,
                risk_params=risk_params,
                backtest_params=backtest_params
            )

        # Display results if available
        if 'backtest_results' in st.session_state:
            self._display_backtest_results()

    def _render_strategy_selection(self) -> Dict:
        """
        Render strategy selection and configuration interface.

        Returns:
            Dict: Strategy configuration parameters
        """
        st.subheader("Strategy Configuration")

        strategy_type = st.selectbox(
            "Strategy Type",
            ["Statistical", "Machine Learning", "Deep Learning"]
        )

        params = {}

        if strategy_type == "Statistical":
            col1, col2 = st.columns(2)
            with col1:
                params.update({
                    'entry_z_score': st.number_input(
                        "Entry Z-Score",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.0,
                        step=0.1
                    ),
                    'exit_z_score': st.number_input(
                        "Exit Z-Score",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.5,
                        step=0.1
                    )
                })
            with col2:
                params.update({
                    'lookback_window': st.number_input(
                        "Lookback Window (days)",
                        min_value=10,
                        max_value=252,
                        value=63
                    ),
                    'zscore_window': st.number_input(
                        "Z-Score Window (days)",
                        min_value=5,
                        max_value=126,
                        value=21
                    )
                })

        elif strategy_type == "Machine Learning":
            col1, col2 = st.columns(2)
            with col1:
                params.update({
                    'model_type': st.selectbox(
                        "ML Model Type",
                        ["RandomForest", "GradientBoosting", "LogisticRegression"]
                    ),
                    'feature_windows': st.multiselect(
                        "Feature Windows",
                        [5, 10, 21, 63, 126],
                        default=[21, 63]
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
                    'signal_threshold': st.slider(
                        "Signal Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.05
                    )
                })

        else:  # Deep Learning
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
                    )
                })
            with col2:
                params.update({
                    'hidden_units': st.number_input(
                        "Hidden Units",
                        min_value=16,
                        max_value=256,
                        value=64
                    ),
                    'dropout_rate': st.slider(
                        "Dropout Rate",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.2,
                        step=0.1
                    )
                })

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

        return {
            'max_position_size': max_position_size,
            'stop_loss': stop_loss,
            'max_drawdown': max_drawdown,
            'max_correlation': max_correlation,
            'leverage_limit': leverage_limit
        }

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

    def _run_backtest(self,
                      strategy_type: Dict,
                      risk_params: Dict,
                      backtest_params: Dict):
        """
        Run backtest with configured strategy and parameters.

        Args:
            strategy_type (Dict): Strategy type and parameters
            risk_params (Dict): Risk management parameters
            backtest_params (Dict): Backtest configuration parameters
        """
        try:
            with st.spinner("Running backtest..."):
                # Create strategy instance
                strategy = self._create_strategy(
                    strategy_type['type'],
                    strategy_type['params']
                )

                # Configure risk manager
                risk_manager = PairRiskManager(
                    max_position_size=risk_params['max_position_size'],
                    max_drawdown=risk_params['max_drawdown'],
                    stop_loss_threshold=risk_params['stop_loss'],
                    max_correlation=risk_params['max_correlation'],
                    leverage_limit=risk_params['leverage_limit']
                )

                # Create backtester
                backtester = MultiPairBacktester(
                    strategy=strategy,
                    returns=self._get_returns_data(),
                    initial_capital=backtest_params['initial_capital'],
                    risk_manager=risk_manager,
                    transaction_cost=backtest_params['transaction_cost'],
                    max_pairs=backtest_params['max_pairs']
                )

                # Run backtest
                equity_curve = backtester.run_backtest()

                # Store results
                st.session_state['backtest_results'] = {
                    'equity_curve': equity_curve,
                    'metrics': backtester.calculate_performance_metrics(),
                    'trades': backtester.trade_history,
                    'parameters': {
                        'strategy': strategy_type,
                        'risk': risk_params,
                        'backtest': backtest_params
                    }
                }

                st.success("Backtest completed successfully!")

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")

    def _create_strategy(self, strategy_type: str, params: Dict):
        """
        Create strategy instance based on type and parameters.

        Args:
            strategy_type (str): Type of strategy to create
            params (Dict): Strategy parameters

        Returns:
            Strategy instance based on type
        """
        if strategy_type == "Statistical":
            return StatisticalPairsTrader(
                lookback_window=params['lookback_window'],
                zscore_window=params['zscore_window'],
                entry_z_score=params['entry_z_score'],
                exit_z_score=params['exit_z_score']
            )

        elif strategy_type == "Machine Learning":
            return PairsTraderML(
                model_type=params['model_type'],
                feature_windows=params['feature_windows'],
                train_size=params['train_size'],
                signal_threshold=params['signal_threshold']
            )

        else:  # Deep Learning
            return PairsTradingDL(
                sequence_length=params['sequence_length'],
                prediction_horizon=params['prediction_horizon'],
                hidden_units=params['hidden_units'],
                dropout_rate=params['dropout_rate']
            )

    def _display_backtest_results(self):
        """Display comprehensive backtest results."""
        results = st.session_state['backtest_results']

        # Summary metrics
        self._display_summary_metrics(results['metrics'])

        # Performance charts
        self._display_performance_charts(
            results['equity_curve'],
            results['trades']
        )

        # Trade analysis
        self._display_trade_analysis(results['trades'])

        # Risk analysis
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
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Portfolio Value", "Drawdown"],
            vertical_spacing=0.12
        )

        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name="Portfolio Value",
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Add trade markers
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

        # Plot drawdown
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

        # Trade statistics by pair
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

        # Trade distribution plot
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

        # Calculate risk metrics
        returns = equity_curve.pct_change().dropna()
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        rolling_sharpe = (returns.rolling(window=63).mean() * 252) / \
                         (returns.rolling(window=63).std() * np.sqrt(252))
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Display risk metrics
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

        # Risk visualization
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

        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name="21d Volatility"
            ),
            row=1, col=1
        )

        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="63d Sharpe"
            ),
            row=1, col=2
        )

        # Return distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name="Returns",
                nbinsx=50
            ),
            row=2, col=1
        )

        # Position exposure
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

        # Pair performance metrics
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

        # Display pair metrics
        st.dataframe(pair_metrics)

        # Pair performance visualization
        fig = go.Figure()

        # Scatter plot of win rate vs average PnL
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

        # Calculate market regime
        market_returns = market_data.pct_change()
        market_vol = market_returns.rolling(window=21).std()

        conditions = pd.qcut(market_vol, q=3, labels=['Low Vol', 'Med Vol', 'High Vol'])
        trades['Market_Regime'] = trades['Date'].map(conditions)

        # Performance by market regime
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

        # Visualization
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "PnL by Market Regime",
                "Win Rate by Market Regime"
            ]
        )

        # PnL box plot
        fig.add_trace(
            go.Box(
                y=trades['PnL'],
                x=trades['Market_Regime'],
                name='PnL Distribution'
            ),
            row=1, col=1
        )

        # Win rate bar plot
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

        # Trading activity over time
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

        # Visualization
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

        # Trading activity
        fig.add_trace(
            go.Scatter(
                x=monthly_activity.index,
                y=monthly_activity['Number of Trades'],
                name='Number of Trades'
            ),
            row=1, col=1
        )

        # PnL vs Duration scatter
        fig.add_trace(
            go.Scatter(
                x=trades['Duration'],
                y=trades['PnL'],
                mode='markers',
                name='Trades'
            ),
            row=1, col=2
        )

        # Trade size distribution
        fig.add_trace(
            go.Histogram(
                x=trades['Quantity'].abs(),
                name='Trade Size'
            ),
            row=2, col=1
        )

        # Hour of day analysis
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

        # Create Excel writer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Write equity curve
            results['equity_curve'].to_excel(
                writer,
                sheet_name='Equity_Curve'
            )

            # Write trades
            results['trades'].to_excel(
                writer,
                sheet_name='Trades'
            )

            # Write metrics
            pd.DataFrame([results['metrics']]).to_excel(
                writer,
                sheet_name='Metrics'
            )

            # Write parameters
            pd.DataFrame([results['parameters']]).to_excel(
                writer,
                sheet_name='Parameters'
            )

        # Offer download
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
            index='date',
            columns='ticker',
            values='adj_close'
        )
        returns = prices.pct_change().dropna()
        return returns