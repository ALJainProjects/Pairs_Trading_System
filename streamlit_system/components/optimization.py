from typing import Type, Optional

import streamlit as st
import pandas as pd
import numpy as np  # For np.array for sensitivity ranges, if re-added
import optuna
import plotly.graph_objects as go  # For direct plotly plots

# Import all necessary optimizers and strategies
from src.strategy.static_pairs_strategy import StaticPairsStrategy
from src.strategy.dynamic_pairs_strategy import DynamicPairsStrategy
from src.strategy.advanced_ml_strategy import AdvancedMLStrategy
from src.strategy.meta_strategy import MetaStrategy  # Assuming MetaStrategy is desired for optimization
from src.strategy.base import BaseStrategy  # For type hinting

from src.strategy.optimization import (  # Corrected import for optimization classes
    OptunaOptimizer,
    WalkForwardOptimizer,
    CrossValidatedOptimizer,
    TransactionCostOptimizer,
    ParameterSensitivityAnalyzer, BaseStrategyEvaluator  # Also include for potential future use or display
)

from src.strategy.backtest import Backtester  # Backtester is used internally by optimizers

# --- Configuration for Strategies and their Parameter Spaces ---
# Define typical parameter ranges for each strategy
STRATEGIES_CONFIG = {
    "StaticPairsStrategy": {
        "class": StaticPairsStrategy,
        "params_space": {
            'lookback_window': {'type': 'int', 'low': 30, 'high': 120},
            'zscore_entry': {'type': 'float', 'low': 1.0, 'high': 3.0},
            'zscore_exit': {'type': 'float', 'low': 0.1, 'high': 1.5},
            # Add cointegration params if you want to optimize them
            'coint_p_threshold': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'coint_min_half_life': {'type': 'int', 'low': 5, 'high': 60},
        }
    },
    "DynamicPairsStrategy": {
        "class": DynamicPairsStrategy,
        "params_space": {
            'reselection_period': {'type': 'int', 'low': 30, 'high': 180},
            'lookback_window': {'type': 'int', 'low': 30, 'high': 120},
            'zscore_entry': {'type': 'float', 'low': 1.0, 'high': 3.0},
            'zscore_exit': {'type': 'float', 'low': 0.1, 'high': 1.5}
            # Add cointegration params if desired
        }
    },
    "AdvancedMLStrategy": {
        "class": AdvancedMLStrategy,
        "params_space": {
            'lookback_window': {'type': 'int', 'low': 30, 'high': 90},
            'sequence_length': {'type': 'int', 'low': 10, 'high': 60},
        }
    },
    "MetaStrategy": {
        "class": MetaStrategy,
        "params_space": {
            # MetaStrategy params (e.g., regime_detector_params)
            # Optimizing sub-strategy params through MetaStrategy requires careful structuring.
            # For simplicity, optimize MetaStrategy's *own* params.
            # To optimize sub-strategy params, you'd define nested optimization in Optuna for each sub-strategy
            # or optimize each sub-strategy separately.
            # Here, we'll expose a simplified MetaStrategy param for demonstration.
            'regime_detector_params__lookback_period': {'type': 'int', 'low': 30, 'high': 90},
            # Example: how long to look back for regime
            'regime_detector_params__volatility_quantile_threshold': {'type': 'float', 'low': 0.5, 'high': 0.9}
        }
    }
}


def render_optimization_page():
    """Renders the UI for strategy parameter optimization."""
    st.title("⚙️ Strategy Optimization")

    # --- Data and Pair Availability Check ---
    if st.session_state.pivot_prices.empty:
        st.warning("Please load data in the 'Data Loader' page first.")
        return

    # If using strategies that require pre-selected pairs (like StaticPairsStrategy),
    # ensure pairs are selected if that's the current requirement.
    # For a general optimization page, we assume the data `pivot_prices` is sufficient.
    # The strategies will internally select/use pairs from this data.

    st.info("Optimize your strategy's parameters using various methods to find the best performance.")

    # --- Strategy Selection ---
    strategy_name = st.selectbox(
        "Select Strategy to Optimize",
        list(STRATEGIES_CONFIG.keys())
    )
    selected_strategy_config = STRATEGIES_CONFIG[strategy_name]
    strategy_class = selected_strategy_config["class"]
    param_space = selected_strategy_config["params_space"]

    st.subheader("Optimization Parameters")
    st.write(f"Optimizing parameters for **{strategy_name}**: (Range)")
    # Display the parameter space for the user
    for param, space in param_space.items():
        st.write(
            f"- **{param}**: {space['type']} from {space.get('low', 'N/A')} to {space.get('high', 'N/A')}{f' (choices: {space.get("choices")})' if space['type'] == 'categorical' else ''}")

    # --- Optimization Mode Selection ---
    optimization_mode = st.radio(
        "Select Optimization Mode",
        ["Single Backtest Optimization", "Walk-Forward Optimization (WFO)", "Time-Series Cross-Validation (TSCV)"],
        index=0  # Default to Single Backtest
    )

    # --- Optimization Specific Parameters ---
    n_trials = st.slider("Number of Optuna Trials", 10, 500, 50)

    # WFO/TSCV specific parameters
    if optimization_mode in ["Walk-Forward Optimization (WFO)", "Time-Series Cross-Validation (TSCV)"]:
        st.subheader("Fold Configuration")
        if optimization_mode == "Walk-Forward Optimization (WFO)":
            col_wfo_1, col_wfo_2, col_wfo_3 = st.columns(3)
            train_size_ratio = col_wfo_1.slider("Train Size Ratio", 0.1, 0.8, 0.6, 0.05)
            test_size_ratio = col_wfo_2.slider("Test Size Ratio", 0.05, 0.4, 0.2, 0.05)
            step_size_ratio = col_wfo_3.slider("Step Size Ratio", 0.01, 0.2, 0.1, 0.01)
        elif optimization_mode == "Time-Series Cross-Validation (TSCV)":
            n_splits = st.slider("Number of Splits (Folds)", 2, 10, 5)

    # --- Transaction Cost Optimization Option ---
    include_transaction_costs = st.checkbox("Include Market Impact & Transaction Costs in Optimization Objective",
                                            value=False)

    # --- Start Optimization Button ---
    if st.button("🏁 Start Optimization", use_container_width=True):
        st.session_state.optimization_results = None  # Clear previous results

        # Determine the optimizer class based on user selection
        optimizer_class: Type[Optional[BaseStrategyEvaluator, OptunaOptimizer]]
        if include_transaction_costs:
            optimizer_class = TransactionCostOptimizer
        else:
            optimizer_class = OptunaOptimizer  # Default to basic Optuna

        try:
            # The data for optimization will always be the full pivoted prices
            # The optimizers (WFO/TSCV) will handle splitting internally.
            data_for_optimization = st.session_state.pivot_prices

            if optimization_mode == "Single Backtest Optimization":
                optimizer = optimizer_class(strategy_class, param_space)
                with st.spinner(f"Running Bayesian optimization for {strategy_name} on single backtest..."):
                    best_params = optimizer.run_optimization(data_for_optimization, n_trials=n_trials)
                    # For single backtest, we can run a final backtest with best params to get full metrics
                    best_strategy = strategy_class(**best_params)
                    final_backtester = Backtester(best_strategy, data_for_optimization)
                    final_results = final_backtester.run()
                    st.session_state.optimization_results = {
                        'mode': 'Single Backtest',
                        'best_params': best_params,
                        'metrics': final_results['metrics'],
                        'equity_curve': final_results['equity_curve'],
                        'trades': final_results['trades']  # Also store trades for analysis
                    }

            elif optimization_mode == "Walk-Forward Optimization (WFO)":
                wfo_optimizer = WalkForwardOptimizer(data_for_optimization, train_size_ratio, test_size_ratio,
                                                     step_size_ratio)
                with st.spinner(f"Running Walk-Forward Optimization for {strategy_name}... This may take a while."):
                    wfo_results = wfo_optimizer.run(strategy_class, param_space, n_trials_per_fold=n_trials)
                    st.session_state.optimization_results = {
                        'mode': 'WFO',
                        'results': wfo_results  # This dict already contains fold_results, overall_metrics etc.
                    }

            elif optimization_mode == "Time-Series Cross-Validation (TSCV)":
                tscv_optimizer = CrossValidatedOptimizer(data_for_optimization, n_splits=n_splits)
                with st.spinner(f"Running Time-Series Cross-Validation for {strategy_name}... This may take a while."):
                    tscv_results = tscv_optimizer.run(strategy_class, param_space, n_trials_per_fold=n_trials)
                    st.session_state.optimization_results = {
                        'mode': 'TSCV',
                        'results': tscv_results  # This dict already contains fold_results, overall_metrics etc.
                    }

            st.success("Optimization complete!")

        except Exception as e:
            st.error(f"Optimization failed: {e}. Check logs for details.")
            st.session_state.optimization_results = None  # Clear results on failure

    # --- Display Optimization Results ---
    if 'optimization_results' in st.session_state and st.session_state.optimization_results is not None:
        results = st.session_state.optimization_results
        st.subheader("Optimization Results")

        if results['mode'] == 'Single Backtest':
            st.metric("Best Sharpe Ratio", f"{results['metrics'].get('Sharpe Ratio', 0.0):.3f}")
            st.write("Best Parameters:")
            st.json(results['best_params'])

            # Show single backtest equity curve
            st.subheader("Best Strategy Equity Curve")
            fig_equity = go.Figure(
                data=go.Scatter(x=results['equity_curve'].index, y=results['equity_curve'], mode='lines',
                                name='Equity'))
            fig_equity.update_layout(title="Equity Curve for Best Parameters (Single Backtest)", xaxis_title="Date",
                                     yaxis_title="Portfolio Value ($)")
            st.plotly_chart(fig_equity, use_container_width=True)

        elif results['mode'] in ['WFO', 'TSCV']:
            overall_metrics = results['results']['overall_metrics']
            st.write(f"### Overall Aggregated Metrics ({results['mode']})")

            # Display overall metrics in a structured way (e.g., DataFrame or table)
            metrics_df = pd.DataFrame([overall_metrics]).T.rename(columns={0: 'Value'})
            st.dataframe(metrics_df)

            st.write(f"### Parameter Stability ({results['mode']})")
            param_stability_df = pd.DataFrame(results['results']['parameter_stability']).T
            st.dataframe(param_stability_df.applymap(
                lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x))  # Format floats

            st.write(f"### Performance Stability Across Folds ({results['mode']})")
            perf_stability_df = pd.DataFrame(results['results']['performance_stability']).T
            st.dataframe(perf_stability_df.applymap(
                lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x))  # Format floats

            # Plot combined equity curve for WFO/TSCV
            st.subheader(f"Combined Equity Curve ({results['mode']})")
            if 'combined_equity_curve' in results['results'] and not results['results']['combined_equity_curve'].empty:
                fig_combined_equity = go.Figure(data=go.Scatter(x=results['results']['combined_equity_curve'].index,
                                                                y=results['results']['combined_equity_curve'],
                                                                mode='lines', name='Combined Equity'))
                fig_combined_equity.update_layout(title=f"Combined Equity Curve ({results['mode']})",
                                                  xaxis_title="Date", yaxis_title="Equity (Normalized)")
                st.plotly_chart(fig_combined_equity, use_container_width=True)
            else:
                st.warning("Combined equity curve not available for plotting.")

            # Optional: Plot individual fold equity curves if desired
            if st.checkbox(f"Show Individual Fold Equity Curves ({results['mode']})"):
                fig_individual_folds = go.Figure()
                for i, fold_res in enumerate(results['results']['fold_results']):
                    if 'equity_curve' in fold_res and not fold_res['equity_curve'].empty:
                        # Normalize each fold for comparison
                        normalized_fold_equity = fold_res['equity_curve'] / fold_res['equity_curve'].iloc[0]
                        fig_individual_folds.add_trace(
                            go.Scatter(x=normalized_fold_equity.index, y=normalized_fold_equity,
                                       mode='lines', name=f'Fold {i + 1}'))
                fig_individual_folds.update_layout(title=f"Individual Fold Equity Curves ({results['mode']})",
                                                   xaxis_title="Date", yaxis_title="Equity (Normalized)")
                st.plotly_chart(fig_individual_folds, use_container_width=True)

        # Common Optuna Plots (only if it's a simple study, or aggregate for WFO/TSCV)
        # These plots typically work best on a single Optuna study object.
        # For WFO/TSCV, you'd need to pick one representative study or aggregate results manually.
        # For simplicity, we'll only show them for 'Single Backtest' mode, or if you were to store Optuna study per fold.
        # Given `plot_optimization_history` and `plot_param_importances` operate on `optuna.study.Study`
        # and we don't save the full study object per fold in WFO/TSCV results, we can only plot if we store it.
        # For this version, let's just make it available for single backtest for now.

        if results['mode'] == 'Single Backtest' and 'optimization_study' in st.session_state:
            study = st.session_state.optimization_study  # This would be the single study object

            st.subheader("Optuna Visualization (Single Backtest)")
            if st.checkbox("Show Optimization History Plot (Single Backtest)"):
                fig = optuna.visualization.plot_optimization_history(study)
                st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Show Parameter Importance Plot (Single Backtest)"):
                try:
                    fig = optuna.visualization.plot_param_importances(study)
                    st.plotly_chart(fig, use_container_width=True)
                except (ValueError, ZeroDivisionError) as e:
                    st.warning(
                        f"Could not generate parameter importance plot: {e}. (Often occurs with too few trials or constant parameters)")

            if st.checkbox("Show Contour Plot (Single Backtest)"):
                # Contour plot requires at least two parameters and numerical params
                numeric_params = [p for p, space in param_space.items() if space['type'] in ['int', 'float']]
                if len(numeric_params) >= 2:
                    # Select top 2 important params if available, otherwise just first 2 numeric
                    try:
                        param_importances = optuna.importance.get_param_importances(study)
                        top_params = [p for p, _ in param_importances.items() if p in numeric_params][:2]
                        if len(top_params) < 2: top_params = numeric_params[:2]  # Fallback
                    except Exception:
                        top_params = numeric_params[:2]  # Fallback if importance calculation fails

                    fig = optuna.visualization.plot_contour(study, params=top_params)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Contour plot requires at least two numerical parameters to optimize.")