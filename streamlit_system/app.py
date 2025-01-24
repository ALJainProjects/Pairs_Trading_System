import streamlit as st

from src.data.database import DatabaseManager
from components.data_loader import EnhancedDataLoader
from components.pair_analyzer import EnhancedPairAnalyzer
from components.strategy_builder import EnhancedStrategyBuilder
from components.optimization import StreamlitOptimizationApp


class PairTradingApp:
    """Main application class for the pair trading system."""

    def __init__(self):
        """Initialize app components and database connection."""
        self.db_manager = DatabaseManager()
        self.data_loader = EnhancedDataLoader()
        self.pair_analyzer = EnhancedPairAnalyzer()
        self.strategy_builder = EnhancedStrategyBuilder()
        self.optimization = StreamlitOptimizationApp()

        if 'backtest_results' not in st.session_state:
            st.session_state['backtest_results'] = None

    def run(self):
        """Run the main application."""
        self._configure_page()
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
            st.title("Navigation")
            page = st.radio(
                "Select Section",
                ["Data Loading", "Pair Analysis", "Strategy Builder", "Optimization"],
                key='sidebar_main_radio'
            )

            st.markdown("---")
            st.markdown("### Session Status")
            self._display_session_status()

            st.markdown("---")
            st.markdown("### Help")
            self._render_help_section(page)

        return page

    def _display_session_status(self):
        """Display the current session state status with standardized column names."""
        if 'historical_data' in st.session_state:
            data = st.session_state['historical_data']
            if not data.empty:
                data.columns = [col for col in data.columns]

                if 'Symbol' not in data.columns and 'ticker' in data.columns:
                    data['Symbol'] = data['ticker']

                if 'Symbol' in data.columns:
                    st.success(f"✅ Data loaded: {len(data['Symbol'].unique())} tickers")
                    if 'Date' in data.columns:
                        st.info(f"Date range: {data['Date'].min():%Y-%m-%d} to {data['Date'].max():%Y-%m-%d}")
                else:
                    st.warning("❌ Data loaded but ticker/symbol column not found")
            else:
                st.warning("❌ Empty dataset loaded")
        else:
            st.warning("❌ No data loaded")

        if 'selected_pairs' in st.session_state:
            pairs = st.session_state['selected_pairs']
            if not pairs.empty:
                st.success(f"✅ {len(pairs)} pairs selected")
            else:
                st.warning("❌ No pairs selected")
        else:
            st.warning("❌ No pairs selected")

        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            if results is not None and isinstance(results, dict) and 'metrics' in results:
                st.success("✅ Strategy backtested")
                metrics = results['metrics']
                st.info(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
                st.info(f"Total Return: {metrics.get('Total Return', 0):.1%}")
            else:
                st.warning("❌ Invalid or incomplete backtest results")
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

        if 'error' in st.session_state:
            del st.session_state['error']

        if page == "Data Loading":
            self.data_loader.render()
        elif page == "Pair Analysis":
            self.pair_analyzer.render()
        elif page == "Strategy Builder":
            self.strategy_builder.render()
        else:
            self.optimization.run()

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