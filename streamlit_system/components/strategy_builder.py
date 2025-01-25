import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Dict, List, Tuple

from config.logging_config import logger
from src.strategy.backtest import MultiPairBackTester
from src.strategy.risk import PairRiskManager
from src.strategy.pairs_strategy_SL import EnhancedStatPairsStrategy
from src.strategy.pairs_strategy_ML import MLPairsStrategy
from src.strategy.pairs_strategy_DL import PairsTradingDL
from streamlit_system.components.session_state_management import SessionStateManager


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
            ["Statistical", "Machine Learning", "Deep Learning"]
        )

        params = {}

        if strategy_type == "Statistical":
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

        else:
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

                strategy = self._create_strategy(
                    strategy_type['type'],
                    strategy_type['params']
                )

                if hasattr(strategy, 'set_tradeable_pairs'):
                    strategy.set_tradeable_pairs(pairs) # can be easily implemented but unecessary for now
                else:
                    strategy.pairs = pairs

                # print(prices.head(10))
                # print(prices.columns)
                # raise

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

        print(data.head(10))
        print(data.columns)

        if 'Date' not in data.columns or 'Symbol' not in data.columns or 'Adj_Close' not in data.columns:
            raise ValueError("Required columns missing in historical data")

        filled_data = data.groupby('Symbol').apply(
            lambda x: x.sort_values('Date').ffill().bfill()
        ).reset_index(drop=True)

        return filled_data.sort_values(['Symbol', 'Date'])

    def _get_selected_pairs(self) -> List[Tuple[str, str]]:
        """
        Get selected pairs from session state with strict type checking.

        Returns:
            List[Tuple[str, str]]: List of validated asset pairs
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
                # print(data)
                available_tickers = set(data['Symbol'].unique())

                valid_pairs = [
                    pair for pair in validated_pairs
                    if pair[0] in available_tickers and pair[1] in available_tickers
                ]

                if not valid_pairs:
                    raise ValueError("No valid pairs found in historical data")

                return valid_pairs

            # print(validated_pairs)
            return validated_pairs

        except Exception as e:
            raise ValueError(f"Error validating pairs: {str(e)}")

    def _create_strategy(self, strategy_type: str, params: Dict):
        """Create and initialize strategy with separate models for each pair."""
        try:
            train_data, test_data = self._get_train_test_data()
            pairs = self._get_selected_pairs()

            if strategy_type == "Statistical":
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

            else:
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
        if 'historical_data' not in st.session_state:
            raise ValueError("No historical data found in session state")

        data = st.session_state['historical_data']

        data['Date'] = pd.to_datetime(data['Date'])
        unique_tickers = data['Symbol'].unique()

        ticker_dates = {}
        for ticker in unique_tickers:
            ticker_data = data[data['Symbol'] == ticker]
            ticker_dates[ticker] = (ticker_data['Date'].min(), ticker_data['Date'].max())

        start_date = max(dates[0] for dates in ticker_dates.values())
        end_date = min(dates[1] for dates in ticker_dates.values())

        data = data[
            (data['Date'] >= start_date) &
            (data['Date'] <= end_date)
            ]

        prices = data.pivot(
            index='Date',
            columns='Symbol',
            values='Adj_Close'
        )

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