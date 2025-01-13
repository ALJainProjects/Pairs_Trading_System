import streamlit as st
from typing import Dict, Optional
import pandas as pd
from datetime import datetime

from src.data.database import DatabaseManager
from components.data_loader import EnhancedDataLoader
from components.pair_analyzer import EnhancedPairAnalyzer
from components.strategy_builder import EnhancedStrategyBuilder
from components.optimization import EnhancedOptimizationComponent


class PairTradingApp:
    """Main application class for the pair trading system."""

    def __init__(self):
        """Initialize app components and database connection."""
        self.db_manager = DatabaseManager()
        self.data_loader = EnhancedDataLoader()
        self.pair_analyzer = EnhancedPairAnalyzer()
        self.strategy_builder = EnhancedStrategyBuilder()
        self.optimization = EnhancedOptimizationComponent()

    def run(self):
        """Run the main application."""
        self._configure_page()
        self._render_sidebar()
        self._render_main_content()

    def _configure_page(self):
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="Pair Trading Strategy Builder",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _render_sidebar(self):
        """Render the sidebar navigation and status."""
        with st.sidebar:
            # Navigation
            st.title("Navigation")
            page = st.radio(
                "Select Section",
                ["Data Loading", "Pair Analysis", "Strategy Builder", "Optimization"]
            )

            # Session state indicators
            st.markdown("---")
            st.markdown("### Session Status")
            self._display_session_status()

            # Help section
            st.markdown("---")
            st.markdown("### Help")
            self._render_help_section(page)

        return page

    def _display_session_status(self):
        """Display the current session state status."""
        # Data status
        if 'historical_data' in st.session_state:
            data = st.session_state['historical_data']
            st.success(f"✅ Data loaded: {len(data['ticker'].unique())} tickers")
            st.info(f"Date range: {data['date'].min():%Y-%m-%d} to {data['date'].max():%Y-%m-%d}")
        else:
            st.warning("❌ No data loaded")

        # Pairs status
        if 'selected_pairs' in st.session_state:
            pairs = st.session_state['selected_pairs']
            st.success(f"✅ {len(pairs)} pairs selected")
        else:
            st.warning("❌ No pairs selected")

        # Strategy status
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            st.success("✅ Strategy backtested")
            if 'metrics' in results:
                metrics = results['metrics']
                st.info(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
                st.info(f"Total Return: {metrics.get('Total Return', 0):.1%}")
        else:
            st.warning("❌ No backtest results")

    def _render_help_section(self, current_page: str):
        """Render context-sensitive help based on current page."""
        help_text = {
            "Data Loading": """
            **Data Loading Help**
            1. Upload your own CSV files
            2. Download market data directly
            3. Query from database

            Required columns: date, ticker, open, high, low, close, volume
            """,

            "Pair Analysis": """
            **Pair Analysis Help**
            1. Select analysis method
            2. Configure parameters
            3. Review and select pairs

            Tips:
            - Look for high correlation
            - Check cointegration
            - Review historical performance
            """,

            "Strategy Builder": """
            **Strategy Builder Help**
            1. Choose strategy type
            2. Configure parameters
            3. Set risk management rules
            4. Run backtest

            Available strategies:
            - Statistical
            - Machine Learning
            - Deep Learning
            """,

            "Optimization": """
            **Optimization Help**
            1. Define parameter ranges
            2. Select objective function
            3. Choose optimization method
            4. Review results

            Methods:
            - Grid Search
            - Bayesian Optimization
            """
        }

        st.markdown(help_text[current_page])

    def _render_main_content(self):
        """Render the main content based on selected page."""
        page = self._render_sidebar()

        # Clear any temporary error messages
        if 'error' in st.session_state:
            del st.session_state['error']

        # Render selected page
        if page == "Data Loading":
            self.data_loader.render()
        elif page == "Pair Analysis":
            self.pair_analyzer.render()
        elif page == "Strategy Builder":
            self.strategy_builder.render()
        else:  # Optimization
            self.optimization.render()

    def _check_dependencies(self, page: str) -> bool:
        """Check if required session state exists for current page."""
        dependencies = {
            "Pair Analysis": ["historical_data"],
            "Strategy Builder": ["historical_data", "selected_pairs"],
            "Optimization": ["historical_data", "selected_pairs"]
        }

        if page in dependencies:
            missing = [dep for dep in dependencies[page]
                       if dep not in st.session_state]
            if missing:
                st.warning(f"Please complete previous steps: {', '.join(missing)}")
                return False
        return True


def main():
    """Main entry point for the application."""
    app = PairTradingApp()
    app.run()


if __name__ == "__main__":
    main()