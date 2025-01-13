import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Callable
import io
from datetime import datetime

from src.strategy.optimization import StrategyOptimizer
from src.strategy.backtest import MultiPairBacktester
from src.strategy.risk import PairRiskManager
from src.utils.metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio


class EnhancedOptimizationComponent:
    """Enhanced optimization component with comprehensive parameter tuning."""

    def __init__(self):
        self.risk_manager = PairRiskManager()

    def render(self):
        """Render the optimization interface."""
        st.header("Strategy Optimization")

        if 'selected_pairs' not in st.session_state:
            st.warning("Please select pairs first in the Pair Analysis section.")
            return

        # Strategy selection and parameter ranges
        strategy_config = self._render_strategy_selection()

        # Optimization configuration
        optimization_config = self._render_optimization_config()

        # Risk constraints
        risk_constraints = self._render_risk_constraints()

        # Run optimization
        if st.button("Run Optimization"):
            self._run_optimization(
                strategy_config=strategy_config,
                optimization_config=optimization_config,
                risk_constraints=risk_constraints
            )

        # Display results if available
        if 'optimization_results' in st.session_state:
            self._display_optimization_results()

    def _render_strategy_selection(self) -> Dict:
        """Render strategy selection and parameter range configuration."""
        st.subheader("Strategy Configuration")

        # Strategy type selection
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Statistical", "Machine Learning", "Deep Learning"]
        )

        # Parameter ranges based on strategy type
        param_ranges = {}
        if strategy_type == "Statistical":
            param_ranges = self._render_statistical_params()
        elif strategy_type == "Machine Learning":
            param_ranges = self._render_ml_params()
        else:  # Deep Learning
            param_ranges = self._render_dl_params()

        return {
            'type': strategy_type,
            'param_ranges': param_ranges
        }

    def _render_statistical_params(self) -> Dict:
        """Configure parameter ranges for statistical strategy."""
        st.subheader("Statistical Strategy Parameters")

        col1, col2 = st.columns(2)
        with col1:
            entry_z_range = {
                'min': st.number_input("Min Entry Z-Score", value=1.0, step=0.1),
                'max': st.number_input("Max Entry Z-Score", value=3.0, step=0.1),
                'steps': st.number_input("Entry Z-Score Steps", value=5, min_value=2)
            }

            lookback_range = {
                'min': st.number_input("Min Lookback (days)", value=20, step=5),
                'max': st.number_input("Max Lookback (days)", value=126, step=5),
                'steps': st.number_input("Lookback Steps", value=5, min_value=2)
            }

        with col2:
            exit_z_range = {
                'min': st.number_input("Min Exit Z-Score", value=0.0, step=0.1),
                'max': st.number_input("Max Exit Z-Score", value=1.0, step=0.1),
                'steps': st.number_input("Exit Z-Score Steps", value=5, min_value=2)
            }

            zscore_window_range = {
                'min': st.number_input("Min Z-Score Window", value=10, step=5),
                'max': st.number_input("Max Z-Score Window", value=63, step=5),
                'steps': st.number_input("Window Steps", value=4, min_value=2)
            }

        return {
            'entry_zscore': entry_z_range,
            'exit_zscore': exit_z_range,
            'lookback': lookback_range,
            'zscore_window': zscore_window_range
        }

    def _render_ml_params(self) -> Dict:
        """Configure parameter ranges for ML strategy."""
        st.subheader("Machine Learning Strategy Parameters")

        col1, col2 = st.columns(2)
        with col1:
            model_types = st.multiselect(
                "Model Types",
                ["RandomForest", "GradientBoosting", "LogisticRegression"],
                default=["RandomForest"]
            )

            feature_windows = st.multiselect(
                "Feature Windows",
                [5, 10, 21, 63, 126],
                default=[21, 63]
            )

        with col2:
            signal_threshold_range = {
                'min': st.number_input("Min Signal Threshold", value=0.5, step=0.05),
                'max': st.number_input("Max Signal Threshold", value=0.9, step=0.05),
                'steps': st.number_input("Threshold Steps", value=5, min_value=2)
            }

            train_size_range = {
                'min': st.number_input("Min Training Days", value=126, step=21),
                'max': st.number_input("Max Training Days", value=252, step=21),
                'steps': st.number_input("Training Size Steps", value=3, min_value=2)
            }

        return {
            'model_types': model_types,
            'feature_windows': feature_windows,
            'signal_threshold': signal_threshold_range,
            'train_size': train_size_range
        }

    def _render_dl_params(self) -> Dict:
        """Configure parameter ranges for DL strategy."""
        st.subheader("Deep Learning Strategy Parameters")

        col1, col2 = st.columns(2)
        with col1:
            sequence_length_range = {
                'min': st.number_input("Min Sequence Length", value=10, step=5),
                'max': st.number_input("Max Sequence Length", value=50, step=5),
                'steps': st.number_input("Sequence Steps", value=5, min_value=2)
            }

            hidden_units_range = {
                'min': st.number_input("Min Hidden Units", value=32, step=16),
                'max': st.number_input("Max Hidden Units", value=128, step=16),
                'steps': st.number_input("Hidden Unit Steps", value=4, min_value=2)
            }

        with col2:
            dropout_range = {
                'min': st.number_input("Min Dropout Rate", value=0.1, step=0.1),
                'max': st.number_input("Max Dropout Rate", value=0.5, step=0.1),
                'steps': st.number_input("Dropout Steps", value=5, min_value=2)
            }

            prediction_horizon_range = {
                'min': st.number_input("Min Prediction Horizon", value=1, step=1),
                'max': st.number_input("Max Prediction Horizon", value=5, step=1),
                'steps': st.number_input("Horizon Steps", value=5, min_value=2)
            }

        return {
            'sequence_length': sequence_length_range,
            'hidden_units': hidden_units_range,
            'dropout_rate': dropout_range,
            'prediction_horizon': prediction_horizon_range
        }

    def _render_optimization_config(self) -> Dict:
        """Configure optimization settings."""
        st.subheader("Optimization Configuration")

        col1, col2 = st.columns(2)
        with col1:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Grid Search", "Bayesian Optimization"]
            )

            objective_metric = st.selectbox(
                "Objective Metric",
                ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]
            )

        with col2:
            if optimization_method == "Grid Search":
                max_combinations = st.number_input(
                    "Maximum Parameter Combinations",
                    min_value=10,
                    max_value=1000,
                    value=100
                )
            else:
                n_trials = st.number_input(
                    "Number of Trials",
                    min_value=10,
                    max_value=200,
                    value=50
                )

            n_jobs = st.number_input(
                "Number of Parallel Jobs",
                min_value=1,
                max_value=8,
                value=4
            )

        return {
            'method': optimization_method,
            'objective': objective_metric,
            'n_trials': n_trials if optimization_method == "Bayesian Optimization" else None,
            'max_combinations': max_combinations if optimization_method == "Grid Search" else None,
            'n_jobs': n_jobs
        }

    def _render_risk_constraints(self) -> Dict:
        """Configure risk constraints for optimization."""
        st.subheader("Risk Constraints")

        col1, col2, col3 = st.columns(3)
        with col1:
            max_drawdown = st.number_input(
                "Maximum Drawdown (%)",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=1.0
            ) / 100

            stop_loss = st.number_input(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=10.0,
                step=1.0
            ) / 100

        with col2:
            max_leverage = st.number_input(
                "Maximum Leverage",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )

            position_size = st.number_input(
                "Maximum Position Size (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=1.0
            ) / 100

        with col3:
            min_sharpe = st.number_input(
                "Minimum Sharpe Ratio",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1
            )

        return {
            'max_drawdown': max_drawdown,
            'stop_loss': stop_loss,
            'max_leverage': max_leverage,
            'position_size': position_size,
            'min_sharpe': min_sharpe
        }

    def _run_optimization(self,
                          strategy_config: Dict,
                          optimization_config: Dict,
                          risk_constraints: Dict):
        """Run optimization with configured parameters."""
        try:
            with st.spinner("Running optimization..."):
                # 1. Create parameter grid
                param_grid = self._create_param_grid(
                    strategy_config['type'],
                    strategy_config['param_ranges'],
                    optimization_config
                )

                # 2. Define objective function
                objective = self._create_objective_function(
                    optimization_config['objective'],
                    risk_constraints
                )

                # 3. Create optimizer
                optimizer = StrategyOptimizer(
                    strategy=self._create_base_strategy(strategy_config['type']),
                    objective_metric=objective,
                    parameter_grid=param_grid,
                    initial_capital=100000,  # Could make configurable
                    transaction_cost=0.001,
                    max_pairs=5  # Could make configurable
                )

                # 4. Run optimization
                if optimization_config['method'] == "Grid Search":
                    best_params, best_score = optimizer.grid_search_optimize()
                else:
                    best_params, best_score = optimizer.bayesian_optimize(
                        n_trials=optimization_config['n_trials']
                    )

                # 5. Store results
                st.session_state['optimization_results'] = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'all_results': optimizer.optimization_results,
                    'configuration': {
                        'strategy': strategy_config,
                        'optimization': optimization_config,
                        'risk': risk_constraints
                    }
                }

                st.success("Optimization completed successfully!")

        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")

    def _create_param_grid(self,
                           strategy_type: str,
                           param_ranges: Dict,
                           opt_config: Dict) -> Dict:
        """Create parameter grid from ranges."""
        param_grid = {}

        if strategy_type == "Statistical":
            param_grid = {
                'entry_zscore': np.linspace(
                    param_ranges['entry_zscore']['min'],
                    param_ranges['entry_zscore']['max'],
                    param_ranges['entry_zscore']['steps']
                ),
                'exit_zscore': np.linspace(
                    param_ranges['exit_zscore']['min'],
                    param_ranges['exit_zscore']['max'],
                    param_ranges['exit_zscore']['steps']
                ),
                'lookback': np.linspace(
                    param_ranges['lookback']['min'],
                    param_ranges['lookback']['max'],
                    param_ranges['lookback']['steps'],
                    dtype=int
                ),
                'zscore_window': np.linspace(
                    param_ranges['zscore_window']['min'],
                    param_ranges['zscore_window']['max'],
                    param_ranges['zscore_window']['steps'],
                    dtype=int
                )
            }

        elif strategy_type == "Machine Learning":
            param_grid = {
                'model_type': param_ranges['model_types'],
                'feature_windows': [param_ranges['feature_windows']],
                'signal_threshold': np.linspace(
                    param_ranges['signal_threshold']['min'],
                    param_ranges['signal_threshold']['max'],
                    param_ranges['signal_threshold']['steps']
                ),
                'train_size': np.linspace(
                    param_ranges['train_size']['min'],
                    param_ranges['train_size']['max'],
                    param_ranges['train_size']['steps'],
                    dtype=int
                )
            }

        else:  # Deep Learning
            param_grid = {
                'sequence_length': np.linspace(
                    param_ranges['sequence_length']['min'],
                    param_ranges['sequence_length']['max'],
                    param_ranges['sequence_length']['steps'],
                    dtype=int
                ),
                'hidden_units': np.linspace(
                    param_ranges['hidden_units']['min'],
                    param_ranges['hidden_units']['max'],
                    param_ranges['hidden_units']['steps'],
                    dtype=int
                ),
                'dropout_rate': np.linspace(
                    param_ranges['dropout_rate']['min'],
                    param_ranges['dropout_rate']['max'],
                    param_ranges['dropout_rate']['steps']
                ),
                'prediction_horizon': np.linspace(
                    param_ranges['prediction_horizon']['min'],
                    param_ranges['prediction_horizon']['max'],
                    param_ranges['prediction_horizon']['steps'],
                    dtype=int
                )
            }

        return param_grid

    def _create_objective_function(self,
                                   metric: str,
                                   risk_constraints: Dict) -> Callable:
        """Create objective function with risk constraints."""

        def objective(equity_curve: pd.Series, returns: pd.DataFrame) -> float:
            # Check risk constraints
            drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
            max_dd = abs(drawdown.min())

            if max_dd > risk_constraints['max_drawdown']:
                return -np.inf

            # Calculate performance metric
            if metric == "Sharpe Ratio":
                score = calculate_sharpe_ratio(equity_curve)
            elif metric == "Sortino Ratio":
                score = calculate_sortino_ratio(equity_curve)
            else:  # Calmar Ratio
                score = calculate_calmar_ratio(equity_curve)

            # Apply risk penalties
            if score < risk_constraints['min_sharpe']:
                score = -np.inf

            return score

        return objective

    def _display_optimization_results(self):
        """Display comprehensive optimization results."""
        results = st.session_state['optimization_results']

        # 1. Best Parameters
        st.subheader("Best Parameters Found")
        self._display_best_parameters(results['best_params'], results['best_score'])

        # 2. Parameter Search Visualization
        st.subheader("Parameter Search Analysis")
        self._display_parameter_search_analysis(results['all_results'])

        # 3. Performance Analysis
        st.subheader("Performance Analysis")
        self._display_performance_analysis(results)

        # 4. Hyperparameter Importance
        st.subheader("Hyperparameter Importance")
        self._display_hyperparameter_importance(results['all_results'])

        # 5. Export Results
        self.export_results(results)

    def _display_best_parameters(self, params: Dict, score: float):
        """Display best parameters found."""
        col1, col2 = st.columns([2, 1])

        with col1:
            # Create parameter table
            param_df = pd.DataFrame([params]).T
            param_df.columns = ['Value']
            param_df.index.name = 'Parameter'
            st.dataframe(param_df)

        with col2:
            # Show best score
            st.metric("Best Score", f"{score:.4f}")

            # Apply parameters button
            if st.button("Apply Best Parameters"):
                st.session_state['strategy_params'] = params
                st.success("Parameters applied! Go to Strategy Builder to run backtest.")

    def _display_parameter_search_analysis(self, all_results: List[Dict]):
        """Visualize parameter search process."""
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)

        # Create visualization based on parameter types
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Score Distribution",
                "Parameter Combinations",
                "Score Evolution",
                "Parameter Correlations"
            ]
        )

        # Score distribution
        fig.add_trace(
            go.Histogram(
                x=results_df['score'],
                name="Score Distribution"
            ),
            row=1, col=1
        )

        # Parameter combinations (for most important parameters)
        param_cols = [col for col in results_df.columns if col != 'score']
        if len(param_cols) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=results_df[param_cols[0]],
                    y=results_df[param_cols[1]],
                    mode='markers',
                    marker=dict(
                        color=results_df['score'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name="Parameter Space"
                ),
                row=1, col=2
            )

        # Score evolution
        fig.add_trace(
            go.Scatter(
                y=results_df['score'].cummax(),
                name="Best Score"
            ),
            row=2, col=1
        )

        # Parameter correlations
        corr_matrix = results_df.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                name="Correlations"
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig)

    def _display_performance_analysis(self, results: Dict):
        """Analyze and display performance characteristics."""
        # Get top N parameter sets
        top_n = 5
        results_df = pd.DataFrame(results['all_results'])
        top_results = results_df.nlargest(top_n, 'score')

        # Run backtests for top parameter sets
        performance_comparison = []

        with st.spinner(f"Analyzing top {top_n} parameter sets..."):
            for _, row in top_results.iterrows():
                backtest_results = self._run_validation_backtest(dict(row))
                performance_comparison.append({
                    'parameters': dict(row),
                    'metrics': backtest_results
                })

        # Display comparison
        comparison_df = pd.DataFrame([
            {
                'Parameter Set': i + 1,
                'Score': result['parameters']['score'],
                'Sharpe': result['metrics']['Sharpe Ratio'],
                'Max DD': result['metrics']['Max Drawdown'],
                'Win Rate': result['metrics']['Win Rate']
            }
            for i, result in enumerate(performance_comparison)
        ])

        st.dataframe(comparison_df)

        # Visualization
        self._plot_performance_comparison(performance_comparison)

    def _display_hyperparameter_importance(self, all_results: List[Dict]):
        """Analyze and display hyperparameter importance."""
        results_df = pd.DataFrame(all_results)
        param_cols = [col for col in results_df.columns if col != 'score']

        # Calculate importance scores
        importance_scores = {}
        for param in param_cols:
            correlation = abs(results_df[param].corr(results_df['score']))
            unique_values = len(results_df[param].unique())
            importance_scores[param] = correlation * np.log(unique_values)

        # Normalize scores
        total = sum(importance_scores.values())
        importance_scores = {k: v / total for k, v in importance_scores.items()}

        # Visualization
        fig = go.Figure(go.Bar(
            x=list(importance_scores.keys()),
            y=list(importance_scores.values()),
            text=[f"{v:.2%}" for v in importance_scores.values()],
            textposition='auto'
        ))

        fig.update_layout(
            title="Hyperparameter Importance",
            xaxis_title="Parameter",
            yaxis_title="Importance Score",
            height=400
        )

        st.plotly_chart(fig)

    def _run_validation_backtest(self, parameters: Dict) -> Dict:
        """Run validation backtest for a parameter set."""
        try:
            strategy = self._create_strategy_from_params(parameters)
            backtester = MultiPairBacktester(
                strategy=strategy,
                returns=self._get_returns_data(),
                initial_capital=100000,
                risk_manager=self.risk_manager
            )

            equity_curve = backtester.run_backtest()
            metrics = backtester.calculate_performance_metrics()

            return metrics

        except Exception as e:
            st.error(f"Error in validation backtest: {str(e)}")
            return {}

    def _plot_performance_comparison(self, performance_comparison: List[Dict]):
        """Plot performance comparison of top parameter sets."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Return Distribution",
                "Risk Metrics",
                "Trading Activity",
                "Parameter Comparison"
            ]
        )

        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for i, perf in enumerate(performance_comparison):
            metrics = perf['metrics']
            params = perf['parameters']

            # Return distribution
            if 'returns' in metrics:
                fig.add_trace(
                    go.Histogram(
                        x=metrics['returns'],
                        name=f"Set {i + 1}",
                        opacity=0.7,
                        marker_color=colors[i]
                    ),
                    row=1, col=1
                )

            # Risk metrics
            fig.add_trace(
                go.Scatter(
                    x=['Sharpe', 'Sortino', 'Calmar'],
                    y=[
                        metrics['Sharpe Ratio'],
                        metrics['Sortino Ratio'],
                        metrics['Calmar Ratio']
                    ],
                    name=f"Set {i + 1}",
                    mode='lines+markers',
                    marker_color=colors[i]
                ),
                row=1, col=2
            )

            # Additional visualizations for trading activity and parameter comparison
            # can be added to remaining subplots

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig)

    def export_results(self, results: Dict):
        """Export optimization results."""
        if st.button("Export Results"):
            try:
                # Convert results to DataFrames
                results_df = pd.DataFrame(results['all_results'])
                config_df = pd.DataFrame([results['configuration']])

                # Create Excel writer
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='All_Results')
                    config_df.to_excel(writer, sheet_name='Configuration')

                    # Add best parameters
                    pd.DataFrame([results['best_params']]).to_excel(
                        writer,
                        sheet_name='Best_Parameters'
                    )

                buffer.seek(0)
                st.download_button(
                    label="Download Results",
                    data=buffer,
                    file_name="optimization_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Error exporting results: {str(e)}")

    @staticmethod
    def _get_returns_data() -> pd.DataFrame:
        """Get returns data from session state."""
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