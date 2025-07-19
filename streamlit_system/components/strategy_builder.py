import streamlit as st
import pandas as pd
import numpy as np  # For numerical operations, e.g., ranges

from src.strategy.backtest import Backtester
from src.strategy.static_pairs_strategy import StaticPairsStrategy
from src.strategy.dynamic_pairs_strategy import DynamicPairsStrategy
from src.strategy.advanced_ml_strategy import AdvancedMLStrategy
from src.strategy.meta_strategy import MetaStrategy
from src.utils.visualization import PlotlyVisualizer
from typing import Dict, Any, Type, Optional, List, Tuple

# Define strategy registry with their classes and default parameter UI ranges/types.
# These ranges should align with what's expected by OptunaOptimizer's param_space logic,
# and how each strategy's __init__ expects its parameters.
STRATEGY_REGISTRY = {
    "Static Statistical Pairs": {
        "class": StaticPairsStrategy,
        "default_params": {
            'lookback_window': 60, 'zscore_entry': 2.0, 'zscore_exit': 0.5,
            'coint_p_threshold': 0.05, 'coint_min_half_life': 10, 'coint_max_half_life': 120,
            'coint_integration_test_min_obs': 60, 'coint_adf_regression_type': 'c'
        },
        "ui_elements": {
            'lookback_window': {'type': 'slider', 'min': 20, 'max': 252, 'step': 1,
                                'help': "Number of periods for rolling calculations."},
            'zscore_entry': {'type': 'slider', 'min': 1.0, 'max': 3.0, 'step': 0.1,
                             'help': "Z-score to enter a new trade."},
            'zscore_exit': {'type': 'slider', 'min': 0.0, 'max': 1.5, 'step': 0.1,
                            'help': "Absolute Z-score to exit an open trade."},
            'coint_p_threshold': {'type': 'slider', 'min': 0.01, 'max': 0.1, 'step': 0.01, 'format': "%.2f",
                                  'help': "P-value threshold for cointegration test."},
            'coint_min_half_life': {'type': 'slider', 'min': 5, 'max': 90, 'step': 1,
                                    'help': "Minimum acceptable half-life of spread mean-reversion."},
            'coint_max_half_life': {'type': 'slider', 'min': 90, 'max': 252, 'step': 1,
                                    'help': "Maximum acceptable half-life of spread mean-reversion."},
            'coint_integration_test_min_obs': {'type': 'slider', 'min': 20, 'max': 252, 'step': 1,
                                               'help': "Minimum observations for stationarity/cointegration tests."},
            'coint_adf_regression_type': {'type': 'selectbox', 'options': ['c', 'ct', 'nc'],
                                          'help': "ADF test regression type (constant, constant+trend, no constant/trend)."},
        }
    },
    "Dynamic Statistical Pairs": {
        "class": DynamicPairsStrategy,
        "default_params": {
            'reselection_period': 60, 'lookback_window': 60, 'zscore_entry': 2.0, 'zscore_exit': 0.5, 'max_pairs': 10,
            'coint_p_threshold': 0.05, 'coint_min_half_life': 10, 'coint_max_half_life': 120,
            'coint_integration_test_min_obs': 60, 'coint_adf_regression_type': 'c'
        },
        "ui_elements": {
            'reselection_period': {'type': 'slider', 'min': 20, 'max': 120, 'step': 1,
                                   'help': "Days between pair re-evaluation."},
            'lookback_window': {'type': 'slider', 'min': 20, 'max': 252, 'step': 1,
                                'help': "Number of periods for rolling calculations."},
            'zscore_entry': {'type': 'slider', 'min': 1.0, 'max': 3.0, 'step': 0.1,
                             'help': "Z-score to enter a new trade."},
            'zscore_exit': {'type': 'slider', 'min': 0.0, 'max': 1.5, 'step': 0.1,
                            'help': "Absolute Z-score to exit an open trade."},
            'max_pairs': {'type': 'number_input', 'min': 1, 'max': 50,
                          'help': "Maximum number of concurrent pairs to trade."},
            'coint_p_threshold': {'type': 'slider', 'min': 0.01, 'max': 0.1, 'step': 0.01, 'format': "%.2f",
                                  'help': "P-value threshold for cointegration test during reselection."},
            'coint_min_half_life': {'type': 'slider', 'min': 5, 'max': 90, 'step': 1,
                                    'help': "Min half-life for pair reselection."},
            'coint_max_half_life': {'type': 'slider', 'min': 90, 'max': 252, 'step': 1,
                                    'help': "Max half-life for pair reselection."},
            'coint_integration_test_min_obs': {'type': 'slider', 'min': 20, 'max': 252, 'step': 1,
                                               'help': "Min observations for reselection tests."},
            'coint_adf_regression_type': {'type': 'selectbox', 'options': ['c', 'ct', 'nc'],
                                          'help': "ADF test regression type for reselection."},
        }
    },
    "Advanced ML (TimesNet) Strategy": {
        "class": AdvancedMLStrategy,
        "default_params": {
            'lookback_window': 60, 'sequence_length': 30, 'max_pairs': 5,
            'timesnet_epochs': 50, 'timesnet_batch_size': 32, 'timesnet_n_filters': 64,
            'timesnet_kernel_size': 3, 'timesnet_stack_size': 2,
            'garch_p': 1, 'garch_q': 1
        },
        "ui_elements": {
            'lookback_window': {'type': 'slider', 'min': 30, 'max': 90, 'step': 1,
                                'help': "Window for spread series in prediction."},
            'sequence_length': {'type': 'slider', 'min': 10, 'max': 60, 'step': 1,
                                'help': "Input sequence length for TimesNet."},
            'max_pairs': {'type': 'number_input', 'min': 1, 'max': 10,
                          'help': "Max concurrent pairs (hardcoded in this demo)."},
            "---TimesNet Model Parameters---": {'type': 'markdown',
                                                'help': "Parameters for the underlying TimesNet neural network model."},
            'timesnet_epochs': {'type': 'slider', 'min': 20, 'max': 100, 'step': 1,
                                'help': "Epochs for TimesNet training."},
            'timesnet_batch_size': {'type': 'slider', 'min': 16, 'max': 64, 'step': 1,
                                    'help': "Batch size for TimesNet training."},
            'timesnet_n_filters': {'type': 'slider', 'min': 32, 'max': 128, 'step': 1,
                                   'help': "Number of filters in TCN residual blocks."},
            'timesnet_kernel_size': {'type': 'slider', 'min': 2, 'max': 5, 'step': 1,
                                     'help': "Kernel size for TCN convolutions."},
            'timesnet_stack_size': {'type': 'slider', 'min': 1, 'max': 3, 'step': 1,
                                    'help': "Number of TCN residual block stacks."},
            "---GARCH Model Parameters---": {'type': 'markdown', 'help': "Parameters for the GARCH volatility model."},
            'garch_p': {'type': 'slider', 'min': 1, 'max': 2, 'step': 1,
                        'help': "Order of the AR term in GARCH model."},
            'garch_q': {'type': 'slider', 'min': 1, 'max': 2, 'step': 1,
                        'help': "Order of the MA term in GARCH model."},
        }
    },
    "Meta Strategy (Regime Switching)": {
        "class": MetaStrategy,
        "default_params": {
            'market_symbol': 'SPY',  # Default market index
            # Regime Detector Params
            'regime_detector_params__lookback_period': 60,
            'regime_detector_params__long_trend_period': 120,
            'regime_detector_params__volatility_quantile_threshold': 0.75,
            'regime_detector_params__trend_slope_threshold': 0.0001,
            # Sub-strategy params (example for Ranging_Quiet and Ranging_Volatile)
            'sub_strategy_params__Ranging_Quiet__zscore_entry': 2.0,
            'sub_strategy_params__Ranging_Quiet__lookback_window': 60,
            'sub_strategy_params__Ranging_Volatile__zscore_entry': 2.5,
            'sub_strategy_params__Ranging_Volatile__lookback_window': 60,
        },
        "ui_elements": {
            'market_symbol': {'type': 'text_input',
                              'help': "Market index symbol for regime detection (must be in loaded data)."},
            "---Regime Detector Parameters---": {'type': 'markdown',
                                                 'help': "Parameters for classifying market into regimes."},
            'regime_detector_params__lookback_period': {'type': 'slider', 'min': 30, 'max': 120, 'step': 1,
                                                        'help': "Short-term lookback for trend/volatility."},
            'regime_detector_params__long_trend_period': {'type': 'slider', 'min': 60, 'max': 250, 'step': 1,
                                                          'help': "Long-term lookback for trend detection."},
            'regime_detector_params__volatility_quantile_threshold': {'type': 'slider', 'min': 0.5, 'max': 0.95,
                                                                      'step': 0.01, 'format': "%.2f",
                                                                      'help': "Quantile threshold for volatile vs quiet regimes."},
            'regime_detector_params__trend_slope_threshold': {'type': 'slider', 'min': 0.00001, 'max': 0.001,
                                                              'step': 0.00001, 'format': "%.5f",
                                                              'help': "Min MA slope for 'Trending' regime."},
            "---Sub-Strategy Parameters (Override Defaults)---": {'type': 'markdown',
                                                                  'help': "Customize parameters for sub-strategies within MetaStrategy. Only common params are exposed here."},
            'sub_strategy_params__Ranging_Quiet__zscore_entry': {'type': 'slider', 'min': 1.0, 'max': 3.0, 'step': 0.1,
                                                                 'help': "Entry Z-score for 'Ranging_Quiet' regime."},
            'sub_strategy_params__Ranging_Quiet__lookback_window': {'type': 'slider', 'min': 20, 'max': 120, 'step': 1,
                                                                    'help': "Lookback window for 'Ranging_Quiet' strategy."},
            'sub_strategy_params__Ranging_Volatile__zscore_entry': {'type': 'slider', 'min': 1.0, 'max': 3.0,
                                                                    'step': 0.1,
                                                                    'help': "Entry Z-score for 'Ranging_Volatile' regime."},
            'sub_strategy_params__Ranging_Volatile__lookback_window': {'type': 'slider', 'min': 20, 'max': 120,
                                                                       'step': 1,
                                                                       'help': "Lookback window for 'Ranging_Volatile' strategy."},
        }
    }
}


def render_strategy_builder_page():
    """Renders the UI for selecting, configuring, and backtesting a strategy."""
    st.title("🛠️ Strategy Builder & Backtester")

    # --- Data and Pair Availability Check ---
    if st.session_state.pivot_prices.empty:
        st.warning("Please load data in the 'Data Loader' page first.")
        return

    if not st.session_state.get('selected_pairs', []):
        st.warning("Please select trading pairs in the 'Pair Analyzer' page first.")
        return

    st.subheader("1. Configure Strategy")

    strategy_name = st.selectbox(
        "Select a Strategy",
        list(STRATEGY_REGISTRY.keys()),
        help="Choose one of the predefined trading strategies to configure and backtest."
    )
    strategy_info = STRATEGY_REGISTRY[strategy_name]
    strategy_class = strategy_info["class"]
    default_params = strategy_info["default_params"]
    ui_elements = strategy_info["ui_elements"]

    # Display Strategy-Specific Hyperparameters
    with st.expander("Strategy Hyperparameters", expanded=True):
        params = {}
        for param_name, ui_config in ui_elements.items():
            # Handle special cases for Markdown text
            if param_name.startswith("---"):
                st.markdown(param_name.replace("---", "##### "))  # Use H5 for sub-headers in expander
                continue

            param_type = ui_config['type']
            default_value = default_params.get(param_name)
            help_text = ui_config.get('help', '')

            # Use Streamlit widgets based on parameter type
            if param_type == 'slider':
                params[param_name] = st.slider(
                    param_name.replace('__', ' ').replace('_', ' ').title(),  # Better display name
                    min_value=ui_config['min'],
                    max_value=ui_config['max'],
                    value=default_value,
                    step=ui_config.get('step', 1),
                    format=ui_config.get('format', None),
                    help=help_text,
                    key=f"param_{strategy_name}_{param_name}"  # Unique key for Streamlit
                )
            elif param_type == 'number_input':
                params[param_name] = st.number_input(
                    param_name.replace('__', ' ').replace('_', ' ').title(),
                    min_value=ui_config['min'],
                    max_value=ui_config['max'],
                    value=default_value,
                    step=ui_config.get('step', 1),
                    format=ui_config.get('format', None),
                    help=help_text,
                    key=f"param_{strategy_name}_{param_name}"
                )
            elif param_type == 'selectbox':
                params[param_name] = st.selectbox(
                    param_name.replace('__', ' ').replace('_', ' ').title(),
                    options=ui_config['options'],
                    index=ui_config['options'].index(default_value) if default_value in ui_config['options'] else 0,
                    help=help_text,
                    key=f"param_{strategy_name}_{param_name}"
                )
            elif param_type == 'text_input':
                params[param_name] = st.text_input(
                    param_name.replace('__', ' ').replace('_', ' ').title(),
                    value=default_value,
                    help=help_text,
                    key=f"param_{strategy_name}_{param_name}"
                )
            # Add other widget types as needed (e.g., checkbox)

        # Special handling for max_pairs, which is dynamic based on selected_pairs
        # This will override the max_pairs from UI elements for these specific strategies
        if "Static" in strategy_name:
            # For Static, max_pairs is usually fixed to selected_pairs length unless explicitly overridden by user
            params['max_pairs'] = st.number_input(
                "Max Concurrent Pairs (for Static Strategy, usually based on selected pairs)",
                min_value=1,
                max_value=len(st.session_state.selected_pairs),  # Max possible is number of selected pairs
                value=len(st.session_state.selected_pairs),
                help="For Static Strategy, this is set to the number of pairs selected in 'Pair Analyzer'."
            )

        elif "Dynamic" in strategy_name:
            # max_pairs is already taken care of by the the ui_elements for DynamicPairsStrategy
            pass

    st.subheader("2. Configure Backtest")
    col1, col2, col3 = st.columns(3)
    initial_capital = col1.number_input("Initial Capital ($)", 10000, 10000000, 100000, step=10000,
                                        help="Starting capital for the portfolio.")
    cost_pct = col2.number_input("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.01, format="%.2f",
                                 help="Percentage cost per trade (e.g., 0.1% = 0.001).") / 100
    slippage_pct = col3.number_input("Slippage (%)", 0.0, 1.0, 0.05, 0.01, format="%.2f",
                                     help="Percentage slippage applied per trade (e.g., 0.05% = 0.0005).") / 100

    if st.button("🚀 Run Backtest", use_container_width=True):
        with st.spinner("Instantiating strategy and running backtest... This may take a while for ML strategies."):
            try:
                # Helper function to parse flat Streamlit params into nested dicts for MetaStrategy
                def _parse_meta_strategy_params(flat_params: Dict[str, Any]) -> Dict[str, Any]:
                    parsed_params = {}
                    regime_detector_params = {}
                    sub_strategy_params = {}

                    for key, value in flat_params.items():
                        if key.startswith('regime_detector_params__'):
                            param_name = key.replace('regime_detector_params__', '')
                            regime_detector_params[param_name] = value
                        elif key.startswith('sub_strategy_params__'):
                            parts = key.split('__')
                            if len(parts) >= 3:
                                regime_key = parts[1]
                                sub_param_name = parts[2]
                                if regime_key not in sub_strategy_params:
                                    sub_strategy_params[regime_key] = {}
                                sub_strategy_params[regime_key][sub_param_name] = value
                        else:
                            parsed_params[key] = value  # Pass through non-nested params

                    if regime_detector_params:
                        parsed_params['regime_detector_params'] = regime_detector_params
                    if sub_strategy_params:
                        parsed_params['sub_strategy_params'] = sub_strategy_params

                    return parsed_params

                # Pass parameters to strategy, parsing for MetaStrategy if applicable
                final_strategy_params = params
                if strategy_name == "Meta Strategy (Regime Switching)":
                    final_strategy_params = _parse_meta_strategy_params(params)

                strategy = strategy_class(name=strategy_name, **final_strategy_params)

                # For Static and Dynamic strategies, explicitly pass selected_pairs
                # AdvancedMLStrategy has hardcoded pairs for this demo, MetaStrategy manages its own
                if "Static" in strategy_name or "Dynamic" in strategy_name:
                    strategy.tradeable_pairs = st.session_state.selected_pairs

                    # Ensure historical_data is the pivoted data, as expected by Backtester
                backtester = Backtester(
                    strategy=strategy,
                    historical_data=st.session_state.pivot_prices,  # Use the pivoted price data
                    initial_capital=initial_capital,
                    transaction_cost_pct=cost_pct,
                    slippage_pct=slippage_pct
                )

                results = backtester.run()
                st.session_state.backtest_results = results
                st.success("Backtest completed successfully!")

            except Exception as e:
                st.error(f"An error occurred during the backtest: {e}")
                st.warning("Please check your input data, selected pairs, and strategy parameters.")
                import traceback
                st.exception(traceback.format_exc())  # Display full traceback in Streamlit as well for dev

    # --- Display Backtest Results ---
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        st.subheader("📈 Backtest Results")
        results = st.session_state.backtest_results
        metrics = results['metrics']

        # Display key metrics using st.metric
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{metrics.get('Total Return', 0):.2%}")
        col2.metric("Annualized Return", f"{metrics.get('Annualized Return', 0):.2%}")
        col3.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
        col4.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Calmar Ratio", f"{metrics.get('Calmar Ratio', 0):.2f}")
        col6.metric("Total Trades", f"{metrics.get('Total Trades', 0):.0f}")
        col7.metric("Win Rate", f"{metrics.get('Win Rate (%)', 0):.2f}%")
        col8.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")

        viz = PlotlyVisualizer()

        st.markdown("#### Equity Curve and Drawdown")
        st.plotly_chart(viz.plot_equity_curve(results['equity_curve']), use_container_width=True)

        st.markdown("#### Underwater Plot")
        st.plotly_chart(viz.plot_underwater(results['equity_curve']), use_container_width=True)

        with st.expander("Detailed Metrics Table & Trade Log"):
            st.markdown("##### Full Performance Metrics")
            metrics_df_display = pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})

            # Apply custom formatting to the display DataFrame for clarity
            def format_metric_display_value(metric_name: str, value: Any) -> str:
                if isinstance(value, (float, np.floating)):
                    if any(s in metric_name for s in ['Return', 'Drawdown', 'Rate', 'Pct', 'Volatility']):
                        return f'{value:.2%}'  # Percentage
                    elif any(s in metric_name for s in ['Ratio', 'Factor', 'Score']):
                        return f'{value:.2f}'  # Ratio/Factor/Score
                    elif any(s in metric_name for s in ['$', 'Profit', 'Loss', 'Cost', 'Capital']):
                        return f'${value:,.2f}'  # Currency
                    elif 'Trades' in metric_name:
                        return f'{int(value)}'  # Integer for trade counts
                    return f'{value:,.4f}'  # Default for other floats
                return str(value)

            metrics_df_display['Formatted Value'] = metrics_df_display.apply(
                lambda row: format_metric_display_value(row.name, row['Value']), axis=1)
            st.dataframe(metrics_df_display[['Formatted Value']])

            st.markdown("##### Trade Log")
            st.dataframe(pd.DataFrame(results['trades']))