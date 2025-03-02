"""
Main Streamlit application for optimization component with baseline comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from streamlit_system.optimization_utilities.optimization_backend import OptimizationBackend, StrategyParameters, \
    OptimizationResult, MultiPairStrategyBridge
from streamlit_system.optimization_utilities.optimization_visualization import OptimizationVisualizer
from streamlit_system.optimization_utilities.optimization_util import load_strategy, setup_logger

# Import functions from strategy_builder
from src.strategy.strategy_builder import find_correlated_pairs, find_cointegrated_pairs, MultiPairTradingSystem
from src.strategy.random_baseline import RandomBaselineStrategy, RandomPairTradingSystem

logger = setup_logger()


class StreamlitOptimizationApp:
    """Streamlit interface for optimization component with baseline comparison."""

    def __init__(self):
        """Initialize the Streamlit application."""
        self.backend = OptimizationBackend()
        self.visualizer = OptimizationVisualizer()
        self.strategies = {
            'Statistical': 'StatisticalStrategy',
            'ML': 'MLStrategy',
            'DL': 'DLStrategy',
            'MultiPair': 'MultiPairSystem'
        }
        self.config = self._load_config()
        self.baseline_results = None  # Store baseline results for comparison

    def _load_config(self) -> Dict:
        """Load application configuration."""
        try:
            # with open('config/optimization_config.json', 'r') as f:
                # return json.load(f)
            raise
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Reverting to default.")
            return {
                'default_trials': 100,
                'max_trials': 500,
                'metrics': ['sharpe_ratio', 'sortino_ratio', 'total_return'],
                'default_metric': 'sharpe_ratio',
                'save_results': True,
                'output_dir': 'optimization_results'
            }

    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar controls and gather parameters."""
        st.sidebar.title("Optimization Settings")

        strategy_type = st.sidebar.selectbox(
            "Strategy Type",
            list(self.strategies.keys())
        )

        data_settings = self._render_data_settings()

        param_settings = self._render_parameter_settings(strategy_type)

        opt_settings = self._render_optimization_settings()

        # Add baseline comparison option
        st.sidebar.subheader("Baseline Comparison")
        baseline_settings = {
            'enable_baseline': st.sidebar.checkbox("Compare with Random Baseline", value=True),
            'baseline_seed': st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=9999),
            'baseline_holding_period': st.sidebar.slider("Avg Holding Period (days)", 5, 30, 10),
            'baseline_entry_probability': st.sidebar.slider("Entry Probability", 0.0, 0.3, 0.1, 0.01),
            'baseline_exit_probability': st.sidebar.slider("Exit Probability", 0.0, 0.3, 0.2, 0.01)
        }

        return {
            'strategy_type': strategy_type,
            'data_settings': data_settings,
            'param_settings': param_settings,
            'opt_settings': opt_settings,
            'baseline_settings': baseline_settings
        }

    def _render_data_settings(self) -> Dict:
        """Render data selection and preprocessing settings."""
        st.sidebar.subheader("Data Settings")

        uploaded_file = st.sidebar.file_uploader(
            "Upload Data (CSV)",
            type=['csv']
        )

        start_date = st.sidebar.date_input(
            "Start Date",
            pd.Timestamp.now() - pd.Timedelta(days=365)
        )

        end_date = st.sidebar.date_input(
            "End Date",
            pd.Timestamp.now()
        )

        return {
            'uploaded_file': uploaded_file,
            'start_date': start_date,
            'end_date': end_date
        }

    def _render_parameter_settings(self, strategy_type: str) -> Dict:
        """Render parameter range settings."""
        st.sidebar.subheader("Parameter Settings")

        if strategy_type == 'MultiPair':
            return self._render_multi_pair_parameter_settings()

        param_ranges = {}
        strategy = load_strategy(self.strategies[strategy_type])
        default_params = strategy.get_default_parameters()

        for param, default_value in default_params.items():
            col1, col2 = st.sidebar.columns(2)

            with col1:
                min_val = st.number_input(
                    f"Min {param}",
                    value=float(default_value) * 0.5,
                    step=float(default_value) * 0.1
                )
            with col2:
                max_val = st.number_input(
                    f"Max {param}",
                    value=float(default_value) * 1.5,
                    step=float(default_value) * 0.1
                )

            param_ranges[param] = (min_val, max_val)

            if st.sidebar.checkbox(f"Log scale for {param}", False):
                param_ranges[param] = (*param_ranges[param], 'log')

        return param_ranges

    def _render_multi_pair_parameter_settings(self) -> Dict:
        """Render parameter settings for MultiPairTradingSystem"""
        st.sidebar.subheader("Multi-Pair Strategy Parameters")

        param_ranges = {
            'window_size': (
                st.sidebar.slider("Window Size (days)", 30, 252, 90),
                st.sidebar.slider("Max Window Size (days)", 90, 300, 252)
            ),
            'threshold': (
                st.sidebar.slider("Min Threshold", 1.0, 2.0, 1.5, 0.1),
                st.sidebar.slider("Max Threshold", 2.0, 4.0, 2.5, 0.1)
            ),
            'stop_loss_pct': (
                st.sidebar.slider("Min Stop Loss (%)", 1.0, 5.0, 2.0, 0.5) / 100,
                st.sidebar.slider("Max Stop Loss (%)", 5.0, 15.0, 10.0, 0.5) / 100
            ),
            'profit_target_pct': (
                st.sidebar.slider("Min Profit Target (%)", 1.0, 5.0, 3.0, 0.5) / 100,
                st.sidebar.slider("Max Profit Target (%)", 5.0, 20.0, 10.0, 0.5) / 100
            ),
            'loss_limit_pct': (
                st.sidebar.slider("Min Loss Limit (%)", 1.0, 3.0, 1.5, 0.5) / 100,
                st.sidebar.slider("Max Loss Limit (%)", 3.0, 10.0, 5.0, 0.5) / 100
            ),
            'max_holding_period': (
                st.sidebar.slider("Min Holding Period (days)", 5, 30, 15),
                st.sidebar.slider("Max Holding Period (days)", 30, 90, 60)
            ),
            'capital_utilization': (
                st.sidebar.slider("Min Capital Utilization", 0.5, 0.7, 0.6, 0.05),
                st.sidebar.slider("Max Capital Utilization", 0.7, 1.0, 0.9, 0.05)
            )
        }

        # Add pair selection settings
        st.sidebar.subheader("Pair Selection")
        pair_selection = st.sidebar.radio(
            "Pair Selection Method",
            ["Use Selected Pairs", "Find Correlated Pairs", "Find Cointegrated Pairs"]
        )

        pair_settings = {
            'selection_method': pair_selection
        }

        if pair_selection == "Find Correlated Pairs":
            pair_settings.update({
                'correlation_threshold': st.sidebar.slider(
                    "Correlation Threshold",
                    0.5, 0.95, 0.7, 0.05
                ),
                'max_pairs': st.sidebar.slider(
                    "Maximum Number of Pairs",
                    1, 20, 5
                )
            })
        elif pair_selection == "Find Cointegrated Pairs":
            pair_settings.update({
                'p_value_threshold': st.sidebar.slider(
                    "Cointegration P-Value Threshold",
                    0.01, 0.15, 0.05, 0.01
                ),
                'max_pairs': st.sidebar.slider(
                    "Maximum Number of Pairs",
                    1, 20, 5
                )
            })

        return {
            'param_ranges': param_ranges,
            'pair_settings': pair_settings
        }

    def _render_optimization_settings(self) -> Dict:
        """Render optimization algorithm settings."""
        st.sidebar.subheader("Optimization Settings")

        n_trials = st.sidebar.slider(
            "Number of Trials",
            min_value=10,
            max_value=self.config['max_trials'],
            value=self.config['default_trials']
        )

        optimization_metric = st.sidebar.selectbox(
            "Optimization Metric",
            self.config['metrics'],
            index=self.config['metrics'].index(self.config['default_metric'])
        )

        advanced_settings = st.sidebar.expander("Advanced Settings")
        with advanced_settings:
            early_stopping = st.checkbox("Enable Early Stopping", True)
            save_results = st.checkbox("Save Results", self.config['save_results'])
            show_progress = st.checkbox("Show Progress", True)

        return {
            'n_trials': n_trials,
            'metric': optimization_metric,
            'early_stopping': early_stopping,
            'save_results': save_results,
            'show_progress': show_progress
        }

    def render_main_content(self, settings: Dict[str, Any]) -> None:
        """Render main content area."""
        st.title("Strategy Optimization")

        # Different optimization buttons for different strategy types
        if settings['strategy_type'] == 'MultiPair':
            if st.button("Start Multi-Pair Optimization"):
                # Run baseline first if enabled
                if settings['baseline_settings']['enable_baseline']:
                    with st.spinner("Running random baseline for comparison..."):
                        self._run_random_baseline(settings)
                # Then run optimization
                self._run_multi_pair_optimization(settings)
        else:
            if st.button("Start Optimization"):
                # Run baseline first if enabled
                if settings['baseline_settings']['enable_baseline']:
                    with st.spinner("Running random baseline for comparison..."):
                        self._run_random_baseline(settings)
                # Then run optimization
                self._run_optimization(settings)

    def _run_random_baseline(self, settings: Dict[str, Any]) -> None:
        """Run random baseline strategy for comparison."""
        try:
            # Get data
            if settings['data_settings']['uploaded_file']:
                # Handle CSV format with possibly different structures
                data = pd.read_csv(settings['data_settings']['uploaded_file'])

                # Check data format and convert to appropriate format
                if 'Date' in data.columns and 'Symbol' in data.columns and 'Adj_Close' in data.columns:
                    # Data is in long format, pivot it
                    data['Date'] = pd.to_datetime(data['Date'])
                    pivoted_data = data.pivot(
                        index='Date',
                        columns='Symbol',
                        values='Adj_Close'
                    )
                    data = pivoted_data
                else:
                    # Assume data is already in pivot format
                    if data.index.name != 'Date' and 'Date' in data.columns:
                        data = data.set_index('Date')
                    data.index = pd.to_datetime(data.index)

                # Apply date filter
                start_date = pd.to_datetime(settings['data_settings']['start_date'])
                end_date = pd.to_datetime(settings['data_settings']['end_date'])
                data = data[(data.index >= start_date) & (data.index <= end_date)]

                if data.empty:
                    raise ValueError("No data available for selected date range")
            else:
                raise ValueError("No data file uploaded")

            # Get pairs
            if settings['strategy_type'] == 'MultiPair':
                pair_settings = settings['param_settings']['pair_settings']
                if pair_settings['selection_method'] == "Find Correlated Pairs":
                    pairs = find_correlated_pairs(
                        prices_df=data,
                        correlation_threshold=pair_settings['correlation_threshold'],
                        max_pairs=pair_settings['max_pairs']
                    )
                elif pair_settings['selection_method'] == "Find Cointegrated Pairs":
                    pairs = find_cointegrated_pairs(
                        prices_df=data,
                        p_value_threshold=pair_settings['p_value_threshold'],
                        max_pairs=pair_settings['max_pairs']
                    )
                else:
                    # Use pairs from session state
                    if 'selected_pairs' not in st.session_state:
                        raise ValueError("No pairs selected. Please select pairs first.")

                    pairs_df = st.session_state['selected_pairs']
                    pairs = []

                    if 'Asset1' in pairs_df.columns and 'Asset2' in pairs_df.columns:
                        for _, row in pairs_df.iterrows():
                            pairs.append((row['Asset1'], row['Asset2']))
                    else:
                        raise ValueError("Selected pairs data format not recognized")
            else:
                # For other strategies, generate some random pairs
                symbols = data.columns.tolist()
                if len(symbols) < 2:
                    raise ValueError("Need at least 2 symbols in data")

                # Create up to 5 random pairs
                import random
                random.seed(settings['baseline_settings']['baseline_seed'])
                pairs = []
                for _ in range(min(5, len(symbols) // 2)):
                    s1, s2 = random.sample(symbols, 2)
                    pairs.append((s1, s2))

            # Create and run random baseline
            baseline_params = settings['baseline_settings']
            random_system = RandomPairTradingSystem(
                pairs=pairs,
                prices=data,
                initial_capital=1000000,
                transaction_cost_bps=1.0,
                seed=baseline_params['baseline_seed']
            )

            # Run backtest
            random_system.run_backtest()

            # Get metrics
            metrics = random_system.get_portfolio_metrics()

            # Create portfolio history DataFrame
            portfolio_df = pd.DataFrame(random_system.portfolio_history)
            portfolio_df.set_index('date', inplace=True)

            # Store baseline results for later comparison
            self.baseline_results = {
                'metrics': metrics,
                'equity_curve': portfolio_df['portfolio_value'],
                'system': random_system
            }

            # Success message
            st.success("Baseline strategy run completed")

        except Exception as e:
            st.error(f"Baseline comparison failed: {str(e)}")
            logger.error(f"Baseline error: {str(e)}", exc_info=True)

    def _convert_optimization_result(self, result: OptimizationResult) -> Dict[str, Any]:
        """Convert backend result to frontend format."""
        return {
            'best_score': result.score,
            'total_trials': result.trial_number,
            'time_taken': result.additional_info.get('time_taken', 0),
            'best_parameters': result.parameters,
            'optimization_history': [{
                'trial': i,
                'score': r.score,
                'params': r.parameters,
                'metrics': r.metrics
            } for i, r in enumerate(self.backend.results_history)],
            'settings': self.config
        }

    def _create_parameter_set(self, param_settings: Dict) -> StrategyParameters:
        """Convert frontend parameters to backend format."""
        param_types = {}
        for param, bounds in param_settings.items():
            if len(bounds) == 3 and bounds[2] == 'log':
                param_types[param] = 'log'
            elif isinstance(bounds[0], int):
                param_types[param] = 'int'
            else:
                param_types[param] = 'float'

        return StrategyParameters(
            values={},
            bounds={k: v[:2] for k, v in param_settings.items()},
            parameter_types=param_types
        )

    def _run_optimization(self, settings: Dict[str, Any]) -> None:
        """Run optimization with proper pipeline handling."""
        try:
            with st.spinner("Running optimization..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                if settings['data_settings']['uploaded_file']:
                    data = pd.read_csv(settings['data_settings']['uploaded_file'])
                    data.index = pd.to_datetime(data.index)
                    start_date = pd.to_datetime(settings['data_settings']['start_date'])
                    end_date = pd.to_datetime(settings['data_settings']['end_date'])
                    data = data[(data.index >= start_date) & (data.index <= end_date)]

                    if data.empty:
                        raise ValueError("No data available for selected date range")
                else:
                    raise ValueError("No data file uploaded")

                strategy_params = StrategyParameters(
                    values={},
                    bounds=settings['param_settings'],
                    parameter_types={k: 'float' for k in settings['param_settings'].keys()},
                    descriptions={}
                )

                strategy_class = load_strategy(self.strategies[settings['strategy_type']])
                strategy = strategy_class()

                self.backend.initialize(
                    strategy=strategy,
                    data=data,
                    parameters=strategy_params,
                    settings=settings['opt_settings']
                )

                results = self.backend.optimize(
                    method=settings['opt_settings'].get('method', 'bayesian'),
                    progress_callback=lambda p: progress_bar.progress(p),
                    status_callback=lambda s: status_text.text(s)
                )

                display_results = {
                    'best_score': results['best_score'],
                    'total_trials': results['n_trials'],
                    'time_taken': results['time_taken'],
                    'best_parameters': results['best_parameters'],
                    'optimization_history': results.get('optimization_history', []),
                    'settings': settings['opt_settings']
                }

                self._display_results(display_results)

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            logger.error(f"Optimization error: {str(e)}", exc_info=True)

    def _run_multi_pair_optimization(self, settings: Dict[str, Any]) -> None:
        """Run optimization for a MultiPairTradingSystem"""
        try:
            with st.spinner("Running Multi-Pair Strategy Optimization..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Load data
                if settings['data_settings']['uploaded_file']:
                    # Handle CSV format with possibly different structures
                    data = pd.read_csv(settings['data_settings']['uploaded_file'])

                    # Check data format and convert to appropriate format
                    if 'Date' in data.columns and 'Symbol' in data.columns and 'Adj_Close' in data.columns:
                        # Data is in long format, pivot it
                        data['Date'] = pd.to_datetime(data['Date'])
                        pivoted_data = data.pivot(
                            index='Date',
                            columns='Symbol',
                            values='Adj_Close'
                        )
                        data = pivoted_data
                    else:
                        # Assume data is already in pivot format
                        if data.index.name != 'Date' and 'Date' in data.columns:
                            data = data.set_index('Date')
                        data.index = pd.to_datetime(data.index)

                    # Apply date filter
                    start_date = pd.to_datetime(settings['data_settings']['start_date'])
                    end_date = pd.to_datetime(settings['data_settings']['end_date'])
                    data = data[(data.index >= start_date) & (data.index <= end_date)]

                    if data.empty:
                        raise ValueError("No data available for selected date range")
                else:
                    raise ValueError("No data file uploaded")

                # Get pairs based on selection method
                pair_settings = settings['param_settings']['pair_settings']
                if pair_settings['selection_method'] == "Find Correlated Pairs":
                    pairs = find_correlated_pairs(
                        prices_df=data,
                        correlation_threshold=pair_settings['correlation_threshold'],
                        max_pairs=pair_settings['max_pairs']
                    )
                    st.info(f"Found {len(pairs)} correlated pairs for optimization")
                elif pair_settings['selection_method'] == "Find Cointegrated Pairs":
                    pairs = find_cointegrated_pairs(
                        prices_df=data,
                        p_value_threshold=pair_settings['p_value_threshold'],
                        max_pairs=pair_settings['max_pairs']
                    )
                    st.info(f"Found {len(pairs)} cointegrated pairs for optimization")
                else:
                    # Use pairs from session state
                    if 'selected_pairs' not in st.session_state:
                        raise ValueError("No pairs selected. Please select pairs first.")

                    pairs_df = st.session_state['selected_pairs']
                    pairs = []

                    if 'Asset1' in pairs_df.columns and 'Asset2' in pairs_df.columns:
                        for _, row in pairs_df.iterrows():
                            pairs.append((row['Asset1'], row['Asset2']))
                    else:
                        raise ValueError("Selected pairs data format not recognized")

                # Run optimization with the bridge
                n_trials = settings['opt_settings']['n_trials']

                results = self.backend.optimize_multi_pair_strategy(
                    data=data,
                    pairs=pairs,
                    initial_capital=1000000,  # Default capital, could be made configurable
                    n_trials=n_trials,
                    progress_callback=lambda p: progress_bar.progress(p),
                    status_callback=lambda s: status_text.text(s)
                )

                self._display_multi_pair_results(results, settings['baseline_settings']['enable_baseline'])

        except Exception as e:
            st.error(f"Multi-pair optimization failed: {str(e)}")
            logger.error(f"Multi-pair optimization error: {str(e)}", exc_info=True)

    def _display_results(self, results: Dict[str, Any]) -> None:
        """Display optimization results."""
        st.header("Optimization Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Score", f"{results['best_score']:.4f}")
        with col2:
            st.metric("Total Trials", str(results['total_trials']))
        with col3:
            st.metric("Time Taken", f"{results['time_taken']:.1f}s")

        st.subheader("Best Parameters")
        st.json(results['best_parameters'])

        tab1, tab2, tab3 = st.tabs([
            "Optimization History",
            "Parameter Analysis",
            "Performance Analysis"
        ])

        with tab1:
            fig = self.visualizer.plot_optimization_history(results)
            st.plotly_chart(fig)

        with tab2:
            fig = self.visualizer.plot_parameter_analysis(results)
            st.plotly_chart(fig)

        with tab3:
            fig = self.visualizer.plot_performance_analysis(results)
            st.plotly_chart(fig)

        if results['settings']['save_results']:
            self._save_results(results)

    def _display_multi_pair_results(self, results: Dict[str, Any], show_baseline: bool = False) -> None:
        """Display results from MultiPairTradingSystem optimization."""
        st.header("Multi-Pair Strategy Optimization Results")

        # Display best parameters
        st.subheader("Optimized Parameters")
        st.json(results['best_parameters'])

        # Show key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Score", f"{results['best_score']:.4f}")
        with col2:
            if 'metrics' in results and 'sharpe_ratio' in results['metrics']:
                st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}")
            else:
                st.metric("Sharpe Ratio", "N/A")
        with col3:
            if 'metrics' in results and 'total_return' in results['metrics']:
                st.metric("Total Return", f"{results['metrics']['total_return']:.2f}%")
            else:
                st.metric("Total Return", "N/A")

        # Display optimization history
        st.subheader("Optimization Progress")
        history_df = pd.DataFrame(results['optimization_history'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(history_df))),
            y=history_df['score'],
            mode='markers',
            name='Trial Scores'
        ))

        # Add best score line
        best_scores = []
        best_so_far = float('-inf')
        for score in history_df['score']:
            if score > best_so_far:
                best_so_far = score
            best_scores.append(best_so_far)

        fig.add_trace(go.Scatter(
            x=list(range(len(history_df))),
            y=best_scores,
            mode='lines',
            name='Best Score'
        ))

        fig.update_layout(
            title="Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Score",
            height=500
        )
        st.plotly_chart(fig)

        # Display equity curve with baseline comparison if enabled
        if 'metrics' in results and 'equity_curve' in results['metrics']:
            st.subheader("Equity Curve")
            equity_curve = results['metrics']['equity_curve']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Optimized Strategy'
            ))

            # Add baseline comparison if available
            if show_baseline and self.baseline_results is not None:
                baseline_equity = self.baseline_results['equity_curve']
                # Ensure the same timeframe
                common_dates = equity_curve.index.intersection(baseline_equity.index)
                if not common_dates.empty:
                    # Normalize to start from the same point
                    baseline_normalized = baseline_equity[common_dates] * (equity_curve.iloc[0] / baseline_equity.iloc[0])
                    fig.add_trace(go.Scatter(
                        x=common_dates,
                        y=baseline_normalized.values,
                        mode='lines',
                        name='Random Baseline',
                        line=dict(dash='dash', color='red')
                    ))

                    # Add outperformance comparison
                    if st.checkbox("Show Outperformance vs Baseline"):
                        # Calculate outperformance
                        outperformance = equity_curve[common_dates] - baseline_normalized
                        fig.add_trace(go.Scatter(
                            x=common_dates,
                            y=outperformance.values,
                            mode='lines',
                            name='Outperformance',
                            line=dict(color='green'),
                            yaxis="y2"
                        ))
                        # Add second y-axis for outperformance
                        fig.update_layout(
                            yaxis2=dict(
                                title="Outperformance ($)",
                                overlaying="y",
                                side="right"
                            )
                        )

            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
            )
            st.plotly_chart(fig)

            # Calculate drawdown
            peak = equity_curve.cummax()
            drawdown = (equity_curve - peak) / peak * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Optimized Strategy Drawdown',
                fill='tozeroy',
                line=dict(color='blue')
            ))

            # Add baseline drawdown if available
            if show_baseline and self.baseline_results is not None:
                baseline_equity = self.baseline_results['equity_curve']
                baseline_peak = baseline_equity.cummax()
                baseline_drawdown = (baseline_equity - baseline_peak) / baseline_peak * 100

                # Filter to common dates
                common_dates = drawdown.index.intersection(baseline_drawdown.index)
                if not common_dates.empty:
                    fig.add_trace(go.Scatter(
                        x=common_dates,
                        y=baseline_drawdown[common_dates].values,
                        mode='lines',
                        name='Baseline Drawdown',
                        line=dict(color='red', dash='dash')
                    ))

            fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300
            )
            st.plotly_chart(fig)

            # Add performance comparison table
            if show_baseline and self.baseline_results is not None:
                st.subheader("Performance Comparison")

                # Get metrics
                opt_metrics = results['metrics']
                baseline_metrics = self.baseline_results['metrics']['Portfolio Metrics']

                # Create comparison dataframe
                comparison_data = {
                    'Metric': [
                        'Total Return (%)',
                        'Sharpe Ratio',
                        'Max Drawdown (%)',
                        'Annual Volatility (%)'
                    ],
                    'Optimized Strategy': [
                        opt_metrics.get('total_return', 0),
                        opt_metrics.get('sharpe_ratio', 0),
                        opt_metrics.get('max_drawdown', 0),
                        opt_metrics.get('annual_vol', 0)
                    ],
                    'Random Baseline': [
                        baseline_metrics.get('Total Return (%)', 0),
                        baseline_metrics.get('Sharpe Ratio', 0),
                        baseline_metrics.get('Max Drawdown (%)', 0),
                        baseline_metrics.get('Annual Volatility (%)', 0)
                    ]
                }

                # Calculate outperformance
                comparison_data['Difference'] = [
                    comparison_data['Optimized Strategy'][0] - comparison_data['Random Baseline'][0],
                    comparison_data['Optimized Strategy'][1] - comparison_data['Random Baseline'][1],
                    comparison_data['Random Baseline'][2] - comparison_data['Optimized Strategy'][2],  # Drawdown is better if lower
                    comparison_data['Random Baseline'][3] - comparison_data['Optimized Strategy'][3]   # Volatility is better if lower
                ]

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)

        # Show system dashboard if available
        if 'system' in results and results['system'] is not None:
            st.subheader("Strategy Performance Dashboard")
            try:
                system_fig = results['system'].plot_portfolio_overview()
                st.plotly_chart(system_fig)

                # Add option to view individual pair analyses
                if st.checkbox("Show Individual Pair Analyses"):
                    for pair, model in results['system'].pair_models.items():
                        with st.expander(f"Pair: {pair[0]}-{pair[1]}"):
                            try:
                                pair_fig = model.plot_pair_analysis()
                                st.plotly_chart(pair_fig)
                            except Exception as e:
                                st.warning(f"Could not generate analysis for pair {pair}: {str(e)}")
            except Exception as e:
                st.warning(f"Could not generate system dashboard: {str(e)}")

        # Save results
        if results['settings']['save_results']:
            self._save_multi_pair_results(results)

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results."""
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"optimization_results_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(results, f, indent=4, default=str)

            st.success(f"Results saved to {filename}")

        except Exception as e:
            st.warning(f"Failed to save results: {str(e)}")
            logger.error(f"Error saving results: {str(e)}", exc_info=True)

    def _save_multi_pair_results(self, results: Dict[str, Any]) -> None:
        """Save multi-pair optimization results."""
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"multi_pair_results_{timestamp}.json"

            # Extract what can be serialized
            serializable_results = {
                'best_parameters': results['best_parameters'],
                'best_score': results['best_score'],
                'optimization_history': results['optimization_history'],
                'time_taken': results['time_taken'],
            }

            # Add metrics if available
            if 'metrics' in results:
                serializable_results['metrics'] = {
                    k: v for k, v in results['metrics'].items()
                    if k != 'equity_curve'  # Skip equity curve as it's not JSON serializable
                }

            # Add baseline comparison if available
            if self.baseline_results is not None:
                serializable_results['baseline_comparison'] = {
                    'metrics': self.baseline_results['metrics']['Portfolio Metrics']
                }

            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=str)

            # Save equity curve separately if available
            if 'metrics' in results and 'equity_curve' in results['metrics']:
                equity_filename = output_dir / f"multi_pair_equity_{timestamp}.csv"
                results['metrics']['equity_curve'].to_csv(equity_filename)

            # Save baseline equity curve if available
            if self.baseline_results is not None and 'equity_curve' in self.baseline_results:
                baseline_filename = output_dir / f"baseline_equity_{timestamp}.csv"
                self.baseline_results['equity_curve'].to_csv(baseline_filename)

            st.success(f"Results saved to {filename}")

        except Exception as e:
            st.warning(f"Failed to save multi-pair results: {str(e)}")
            logger.error(f"Error saving multi-pair results: {str(e)}", exc_info=True)

    def _get_price_data(self) -> pd.DataFrame:
        """Get price data from session state or from a file."""
        if 'pivot_prices' in st.session_state:
            return st.session_state['pivot_prices']

        if 'historical_data' in st.session_state:
            data = st.session_state['historical_data']

            # Check if data is already in the right format
            if 'Date' in data.columns and 'Symbol' in data.columns and 'Adj_Close' in data.columns:
                # Data is in long format, pivot it
                data['Date'] = pd.to_datetime(data['Date'])
                pivoted_data = data.pivot(
                    index='Date',
                    columns='Symbol',
                    values='Adj_Close'
                )
                return pivoted_data

            return data

        raise ValueError("No price data available")

    def _get_selected_pairs(self) -> List[Tuple[str, str]]:
        """Get selected pairs from session state."""
        if 'selected_pairs' not in st.session_state:
            raise ValueError("No pairs selected. Please select pairs first.")

        pairs_df = st.session_state['selected_pairs']
        pairs = []

        if 'Asset1' in pairs_df.columns and 'Asset2' in pairs_df.columns:
            for _, row in pairs_df.iterrows():
                pairs.append((row['Asset1'], row['Asset2']))
        else:
            raise ValueError("Selected pairs data format not recognized")

        return pairs

    def run(self) -> None:
        """Run the Streamlit application."""
        st.title("Strategy Optimization")

        tabs = st.tabs(["General Optimization", "Multi-Pair Optimization"])

        with tabs[0]:
            settings = self.render_sidebar()
            if settings['strategy_type'] != 'MultiPair':
                if st.button("Start General Optimization"):
                    # Run baseline first if enabled
                    if settings['baseline_settings']['enable_baseline']:
                        with st.spinner("Running random baseline for comparison..."):
                            self._run_random_baseline(settings)
                    # Then run optimization
                    self._run_optimization(settings)
            else:
                st.info("For Multi-Pair optimization, please use the Multi-Pair Optimization tab")

        with tabs[1]:
            multi_pair_settings = self._render_multi_pair_optimization_settings()
            if st.button("Start Multi-Pair Optimization"):
                # Run baseline first if enabled
                if multi_pair_settings['baseline_settings']['enable_baseline']:
                    with st.spinner("Running random baseline for comparison..."):
                        self._run_random_baseline(multi_pair_settings)
                # Then run optimization
                self._run_multi_pair_optimization(multi_pair_settings)

    def _render_multi_pair_optimization_settings(self) -> Dict[str, Any]:
        """Render settings specific to MultiPair optimization."""
        st.sidebar.title("Multi-Pair Optimization Settings")

        # Data settings
        data_settings = self._render_data_settings()

        # Parameter settings for multi-pair strategy
        st.sidebar.subheader("Strategy Parameters")
        param_settings = {
            'window_size': st.sidebar.slider("Window Size (days)", 30, 252, 90),
            'threshold': st.sidebar.slider("Threshold", 1.0, 4.0, 2.0, 0.1),
            'stop_loss_pct': st.sidebar.slider("Stop Loss (%)", 1.0, 15.0, 5.0, 0.5) / 100,
            'profit_target_pct': st.sidebar.slider("Profit Target (%)", 1.0, 20.0, 5.0, 0.5) / 100,
            'loss_limit_pct': st.sidebar.slider("Loss Limit (%)", 1.0, 10.0, 3.0, 0.5) / 100,
            'max_holding_period': st.sidebar.slider("Max Holding Period (days)", 5, 90, 30),
            'capital_utilization': st.sidebar.slider("Capital Utilization", 0.5, 1.0, 0.8, 0.05)
        }

        # Pair selection settings
        st.sidebar.subheader("Pair Selection")
        pair_selection = st.sidebar.radio(
            "Pair Selection Method",
            ["Use Selected Pairs", "Find Correlated Pairs", "Find Cointegrated Pairs"]
        )

        pair_settings = {
            'selection_method': pair_selection
        }

        if pair_selection == "Find Correlated Pairs":
            pair_settings.update({
                'correlation_threshold': st.sidebar.slider(
                    "Correlation Threshold",
                    0.5, 0.95, 0.7, 0.05
                ),
                'max_pairs': st.sidebar.slider(
                    "Maximum Number of Pairs",
                    1, 20, 5
                )
            })
        elif pair_selection == "Find Cointegrated Pairs":
            pair_settings.update({
                'p_value_threshold': st.sidebar.slider(
                    "Cointegration P-Value Threshold",
                    0.01, 0.15, 0.05, 0.01
                ),
                'max_pairs': st.sidebar.slider(
                    "Maximum Number of Pairs",
                    1, 20, 5
                )
            })

        # Optimization settings
        opt_settings = self._render_optimization_settings()

        # Baseline comparison settings
        st.sidebar.subheader("Baseline Comparison")
        baseline_settings = {
            'enable_baseline': st.sidebar.checkbox("Compare with Random Baseline", value=True),
            'baseline_seed': st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=9999),
            'baseline_holding_period': st.sidebar.slider("Avg Holding Period (days)", 5, 30, 10),
            'baseline_entry_probability': st.sidebar.slider("Entry Probability", 0.0, 0.3, 0.1, 0.01),
            'baseline_exit_probability': st.sidebar.slider("Exit Probability", 0.0, 0.3, 0.2, 0.01)
        }

        return {
            'data_settings': data_settings,
            'param_settings': param_settings,
            'pair_settings': pair_settings,
            'opt_settings': opt_settings,
            'baseline_settings': baseline_settings
        }


if __name__ == "__main__":
    app = StreamlitOptimizationApp()
    app.run()