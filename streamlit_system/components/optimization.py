import streamlit as st
import optuna
from src.strategy.static_pairs_strategy import StaticPairsStrategy
from src.strategy.backtest import Backtester

def render_optimization_page():
    """Renders the UI for strategy parameter optimization."""
    st.title("⚙️ Strategy Optimization")

    if st.session_state.pivot_prices.empty or not st.session_state.selected_pairs:
        st.warning("Please load data and select pairs first.")
        return

    st.info("This page demonstrates Bayesian optimization using Optuna for the Static Pairs Strategy.")

    n_trials = st.slider("Number of Optimization Trials", 10, 200, 50)

    if st.button("🏁 Start Optimization", use_container_width=True):

        def objective(trial):
            params = {
                'lookback_window': trial.suggest_int('lookback_window', 20, 252),
                'zscore_entry': trial.suggest_float('zscore_entry', 1.0, 3.0),
                'zscore_exit': trial.suggest_float('zscore_exit', 0.1, 1.5),
            }
            if params['zscore_exit'] >= params['zscore_entry']:
                raise optuna.exceptions.TrialPruned()

            try:
                strategy = StaticPairsStrategy(**params, max_pairs=len(st.session_state.selected_pairs))
                strategy.tradeable_pairs = st.session_state.selected_pairs

                backtester = Backtester(
                    strategy=strategy,
                    historical_data=st.session_state.historical_data,
                    initial_capital=100000
                )
                results = backtester.run()
                sharpe = results['metrics'].get('Sharpe Ratio', 0.0)

                return sharpe if pd.notna(sharpe) and np.isfinite(sharpe) else -1
            except Exception:
                return -1

        with st.spinner("Running Bayesian optimization... This may take a while."):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            st.session_state.optimization_study = study

        st.success("Optimization complete!")

    if 'optimization_study' in st.session_state:
        study = st.session_state.optimization_study
        st.subheader("Optimization Results")

        st.metric("Best Sharpe Ratio", f"{study.best_value:.3f}")
        st.write("Best Parameters:")
        st.json(study.best_params)

        if st.checkbox("Show Optimization History Plot"):
            fig = optuna.visualization.plot_optimization_history(study)
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Parameter Importance Plot"):
            try:
                fig = optuna.visualization.plot_param_importances(study)
                st.plotly_chart(fig, use_container_width=True)
            except (ValueError, ZeroDivisionError):
                st.warning("Could not generate parameter importance plot.")

        if st.checkbox("Show Contour Plot"):
            params = list(study.best_params.keys())
            if len(params) >= 2:
                fig = optuna.visualization.plot_contour(study, params=params[:2])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Contour plot requires at least two parameters to optimize.")