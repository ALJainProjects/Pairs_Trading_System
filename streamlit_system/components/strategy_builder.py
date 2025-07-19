import streamlit as st
import pandas as pd
from src.strategy.backtest import Backtester
from src.strategy.static_pairs_strategy import StaticPairsStrategy
from src.strategy.dynamic_pairs_strategy import DynamicPairsStrategy
from src.strategy.advanced_ml_strategy import AdvancedMLStrategy
from src.strategy.meta_strategy import MetaStrategy
from src.utils.visualization import PlotlyVisualizer

STRATEGY_REGISTRY = {
    "Static Statistical Pairs": StaticPairsStrategy,
    "Dynamic Statistical Pairs": DynamicPairsStrategy,
    "Advanced ML (TimesNet) Strategy": AdvancedMLStrategy,
    "Meta Strategy (Regime Switching)": MetaStrategy,
}


def render_strategy_builder_page():
    """Renders the UI for selecting, configuring, and backtesting a strategy."""
    st.title("🛠️ Strategy Builder & Backtester")

    if st.session_state.historical_data.empty or not st.session_state.selected_pairs:
        st.warning("Please load data and select pairs in the preceding steps.")
        return

    st.subheader("1. Configure Strategy")

    strategy_name = st.selectbox("Select a Strategy", list(STRATEGY_REGISTRY.keys()))
    strategy_class = STRATEGY_REGISTRY[strategy_name]

    with st.expander("Strategy Hyperparameters", expanded=True):
        params = {}
        if "Static" in strategy_name:
            params['lookback_window'] = st.slider("Lookback Window", 20, 252, 60)
            params['zscore_entry'] = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
            params['zscore_exit'] = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.1)
            params['max_pairs'] = len(st.session_state.selected_pairs)

        elif "Dynamic" in strategy_name:
            params['reselection_period'] = st.slider("Pair Reselection Period (days)", 20, 120, 60)
            params['lookback_window'] = st.slider("Lookback Window", 20, 252, 60)
            params['zscore_entry'] = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
            params['zscore_exit'] = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.1)
            params['max_pairs'] = st.number_input("Max Concurrent Pairs", 1, 50, 10)
        else:
            st.info("Advanced strategies use pre-configured parameters for this demo.")

    st.subheader("2. Configure Backtest")
    col1, col2, col3 = st.columns(3)
    initial_capital = col1.number_input("Initial Capital", 10000, 10000000, 100000)
    cost_pct = col2.number_input("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.01) / 100
    slippage_pct = col3.number_input("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100

    if st.button("🚀 Run Backtest", use_container_width=True):
        with st.spinner("Instantiating strategy and running backtest..."):
            try:
                strategy = strategy_class(name=strategy_name, **params)
                if "Static" in strategy_name:
                    strategy.tradeable_pairs = st.session_state.selected_pairs

                backtester = Backtester(
                    strategy=strategy,
                    historical_data=st.session_state.historical_data,
                    initial_capital=initial_capital,
                    transaction_cost_pct=cost_pct,
                    slippage_pct=slippage_pct
                )

                results = backtester.run()
                st.session_state.backtest_results = results
                st.success("Backtest completed successfully!")

            except Exception as e:
                st.error(f"An error occurred during the backtest: {e}")

    if st.session_state.backtest_results:
        st.subheader("📈 Backtest Results")
        results = st.session_state.backtest_results
        metrics = results['metrics']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{metrics.get('Total Return', 0):.2%}")
        c2.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
        c3.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")
        c4.metric("Calmar Ratio", f"{metrics.get('Calmar Ratio', 0):.2f}")

        viz = PlotlyVisualizer()
        st.plotly_chart(viz.plot_equity_curve(results['equity_curve']), use_container_width=True)
        st.plotly_chart(viz.plot_underwater(results['equity_curve']), use_container_width=True)

        with st.expander("Detailed Metrics & Trade Log"):
            st.dataframe(pd.DataFrame([metrics]))
            st.dataframe(pd.DataFrame(results['trades']))