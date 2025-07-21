import streamlit as st
import pandas as pd

# Import the refactored UI components
from streamlit_system.components.data_loader import render_data_loader_page
from streamlit_system.components.pair_analyzer import render_pair_analyzer_page
from streamlit_system.components.strategy_builder import render_strategy_builder_page
from streamlit_system.components.optimization import render_optimization_page


def initialize_session_state():
    """Initializes all required keys in Streamlit's session state."""
    state_defaults = {
        'historical_data': pd.DataFrame(),
        'pivot_prices': pd.DataFrame(),
        'selected_pairs': [],
        'backtest_results': None,
        'latest_analysis': {}
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Renders the sidebar navigation and status display."""
    with st.sidebar:
        st.title("📈 Quant Platform")

        page = st.radio(
            "Navigation",
            ["Data Loader", "Pair Analyzer", "Strategy & Backtest", "Optimization"],
            key='sidebar_nav'
        )
        st.markdown("---")

        # Status Display
        st.subheader("System Status")
        if not st.session_state.historical_data.empty:
            symbols = st.session_state.historical_data['Symbol'].nunique()
            st.success(f"✅ Data Loaded ({symbols} symbols)")
        else:
            st.warning("❌ No Data Loaded")

        if st.session_state.selected_pairs:
            st.success(f"✅ Pairs Selected ({len(st.session_state.selected_pairs)})")
        else:
            st.warning("❌ No Pairs Selected")

        if st.session_state.backtest_results:
            st.success("✅ Backtest Complete")
        else:
            st.warning("❌ No Backtest Results")

    return page


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Pairs Trading System", layout="wide")

    initialize_session_state()
    page = render_sidebar()

    if page == "Data Loader":
        render_data_loader_page()
    elif page == "Pair Analyzer":
        render_pair_analyzer_page()
    elif page == "Strategy & Backtest":
        render_strategy_builder_page()
    elif page == "Optimization":
        render_optimization_page()


if __name__ == "__main__":
    main()