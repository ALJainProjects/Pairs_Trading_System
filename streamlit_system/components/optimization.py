"""
Main Streamlit application for optimization component with proper frontend handling.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import json

from streamlit_system.optimization_utilities.optimization_backend import OptimizationBackend, StrategyParameters, \
    OptimizationResult
from streamlit_system.optimization_utilities.optimization_visualization import OptimizationVisualizer
from streamlit_system.optimization_utilities.optimization_util import load_strategy, setup_logger

logger = setup_logger()


class StreamlitOptimizationApp:
    """Streamlit interface for optimization component."""

    def __init__(self):
        """Initialize the Streamlit application."""
        self.backend = OptimizationBackend()
        self.visualizer = OptimizationVisualizer()
        self.strategies = {
            'Statistical': 'StatisticalStrategy',
            'ML': 'MLStrategy',
            'DL': 'DLStrategy'
        }
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load application configuration."""
        try:
            with open('config/optimization_config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
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

        return {
            'strategy_type': strategy_type,
            'data_settings': data_settings,
            'param_settings': param_settings,
            'opt_settings': opt_settings
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

        if st.button("Start Optimization"):
            self._run_optimization(settings)

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

    def run(self) -> None:
        """Run the Streamlit application."""
        settings = self.render_sidebar()
        self.render_main_content(settings)


if __name__ == "__main__":
    app = StreamlitOptimizationApp()
    app.run()