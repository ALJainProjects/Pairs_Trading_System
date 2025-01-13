import streamlit as st
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import plotly.graph_objects as go

from src.data.downloader import DataDownloader
from src.data.preprocessor import Preprocessor
from src.data.database import DatabaseManager
from src.utils.validation import validate_dataframe


class EnhancedDataLoader:
    """Enhanced data loading component with comprehensive functionality."""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.db_manager = DatabaseManager()
        self.downloader = DataDownloader()

    def render(self):
        """Render the enhanced data loading interface."""
        st.header("Data Loading and Preprocessing")

        # Create tabs for different data loading methods
        tab1, tab2, tab3 = st.tabs(["File Upload", "Direct Download", "Database Query"])

        with tab1:
            self._render_file_upload()

        with tab2:
            self._render_direct_download()

        with tab3:
            self._render_database_query()

    def _render_file_upload(self):
        """Handle file uploads."""
        st.subheader("Upload Data Files")

        uploaded_files = st.file_uploader(
            "Upload CSV files (one per ticker)",
            type="csv",
            accept_multiple_files=True
        )

        if uploaded_files:
            try:
                with st.spinner("Processing uploaded files..."):
                    # Process each file
                    dfs = []
                    for file in uploaded_files:
                        df = pd.read_csv(file)
                        df['ticker'] = file.name.split('.')[0]  # Use filename as ticker
                        dfs.append(df)

                    # Combine and process
                    combined_df = pd.concat(dfs, ignore_index=True)
                    processed_df = self._process_data(combined_df)

                    # Store in session and database
                    self._store_data(processed_df)

                    # Display summary
                    self._display_data_summary(processed_df)

            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

    def _render_direct_download(self):
        """Handle direct data downloads."""
        st.subheader("Download Market Data")

        col1, col2 = st.columns([2, 1])
        with col1:
            # Allow manual input, CSV upload, and predefined lists
            ticker_input_method = st.radio(
                "Ticker Input Method",
                ["Manual Input", "CSV Upload", "NASDAQ 100", "S&P 500"]
            )

            if ticker_input_method == "Manual Input":
                tickers_input = st.text_input(
                    "Enter ticker symbols (comma-separated)",
                    "AAPL,MSFT,GOOGL"
                )
                tickers = [t.strip() for t in tickers_input.split(",")]

            elif ticker_input_method == "CSV Upload":
                ticker_file = st.file_uploader(
                    "Upload CSV file with ticker symbols",
                    type="csv",
                    help="CSV should have a column named 'ticker' or 'symbol'"
                )
                if ticker_file:
                    try:
                        ticker_df = pd.read_csv(ticker_file)
                        # Look for ticker column with various possible names
                        ticker_col = next(
                            (col for col in ticker_df.columns
                             if col.lower() in ['ticker', 'symbol', 'tickers', 'symbols']),
                            None
                        )
                        if ticker_col:
                            tickers = ticker_df[ticker_col].dropna().unique().tolist()
                            st.success(f"Loaded {len(tickers)} tickers from CSV")

                            # Show preview of loaded tickers
                            with st.expander("Preview Loaded Tickers"):
                                st.dataframe(
                                    pd.DataFrame(tickers, columns=['Ticker']).head(10)
                                )
                        else:
                            st.error(
                                "No ticker column found in CSV. Please ensure column is named 'ticker' or 'symbol'")
                            tickers = []
                    except Exception as e:
                        st.error(f"Error reading ticker CSV: {str(e)}")
                        tickers = []
                else:
                    tickers = []
            else:
                # Get predefined list
                tickers = self.downloader.get_nasdaq100_components() \
                    if ticker_input_method == "NASDAQ 100" \
                    else self._get_sp500_tickers()

            # Display selected tickers
            if tickers:
                with st.expander("Selected Tickers Summary"):
                    st.write(f"Number of tickers: {len(tickers)}")
                    st.write("Sample tickers:", ", ".join(tickers[:10]))
                    if len(tickers) > 10:
                        st.write("... and", len(tickers) - 10, "more")

        with col2:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365 * 2)
            )
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )

        # Download controls
        if tickers:
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input(
                    "Download Batch Size",
                    min_value=1,
                    max_value=100,
                    value=min(50, len(tickers))
                )
            with col2:
                include_validation = st.checkbox(
                    "Include Data Validation",
                    value=True
                )

        # Download button with progress
        if st.button("Download Data"):
            if not tickers:
                st.warning("Please select tickers before downloading.")
                return

            try:
                with st.spinner("Downloading market data..."):
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(i, total, message):
                        progress_bar.progress(i / total)
                        status_text.text(message)

                    # Download with progress updates
                    df = self.downloader.download_historical_data(
                        tickers,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                        progress_callback=update_progress
                    )

                    if include_validation:
                        processed_df = self._process_data(df)
                    else:
                        processed_df = df

                    self._store_data(processed_df)
                    self._display_data_summary(processed_df)

            except Exception as e:
                st.error(f"Download error: {str(e)}")

    def _render_database_query(self):
        """Handle database queries."""
        st.subheader("Query Historical Data")

        # Query parameters
        col1, col2 = st.columns(2)
        with col1:
            ticker_filter = st.text_input(
                "Filter by tickers (comma-separated, empty for all)",
                ""
            )
            min_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365)
            )

        with col2:
            max_date = st.date_input(
                "End Date",
                datetime.now()
            )

        if st.button("Query Database"):
            try:
                with st.spinner("Querying database..."):
                    # Parse ticker filter
                    tickers = [t.strip() for t in ticker_filter.split(",")] \
                        if ticker_filter else None

                    # Query data
                    df = self.db_manager.fetch_historical_data(
                        tickers=tickers,
                        start_date=min_date,
                        end_date=max_date
                    )

                    if df.empty:
                        st.warning("No data found for the specified criteria.")
                        return

                    processed_df = self._process_data(df)
                    self._store_data(processed_df)
                    self._display_data_summary(processed_df)

            except Exception as e:
                st.error(f"Database query error: {str(e)}")

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with validation and cleaning."""
        # Validate input
        if not validate_dataframe(df, methods=['missing', 'dtype']):
            st.warning("Data validation warnings detected. See logs for details.")

        # Clean and process
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.normalize_data(df, method="z-score")
        df = self.preprocessor.handle_outliers(df)

        return df

    def _store_data(self, df: pd.DataFrame):
        """Store processed data in session state and database."""
        # Store in session
        st.session_state['historical_data'] = df

        # Store in database
        try:
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker]
                self.db_manager.insert_historical_data(ticker, ticker_data)
        except Exception as e:
            st.warning(f"Database storage warning: {str(e)}")

    def _display_data_summary(self, df: pd.DataFrame):
        """Display comprehensive data summary with visualizations."""
        st.subheader("Data Summary")

        # Basic info
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Tickers", len(df['ticker'].unique()))
        col2.metric("Date Range", f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
        col3.metric("Total Records", len(df))

        # Sample price chart
        fig = go.Figure()
        for ticker in df['ticker'].unique()[:5]:  # First 5 tickers
            ticker_data = df[df['ticker'] == ticker]
            fig.add_trace(go.Scatter(
                x=ticker_data['date'],
                y=ticker_data['adj_close'],
                name=ticker,
                mode='lines'
            ))

        fig.update_layout(
            title="Sample Price Data",
            xaxis_title="Date",
            yaxis_title="Adjusted Close Price",
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # Data quality metrics
        st.subheader("Data Quality Metrics")
        quality_metrics = self._calculate_quality_metrics(df)
        st.dataframe(quality_metrics)

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate data quality metrics per ticker."""
        metrics = []
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]
            metrics.append({
                'Ticker': ticker,
                'Missing Values (%)': (ticker_data.isnull().sum().sum() / len(ticker_data)) * 100,
                'Trading Days': len(ticker_data),
                'Price Range': f"{ticker_data['adj_close'].min():.2f} - {ticker_data['adj_close'].max():.2f}",
                'Avg Daily Volume': ticker_data['volume'].mean()
            })
        return pd.DataFrame(metrics)

    @staticmethod
    def _get_sp500_tickers() -> List[str]:
        """Get S&P 500 tickers (placeholder)."""
        # Implement actual S&P 500 ticker retrieval
        return ["AAPL", "MSFT", "GOOGL"]  # Placeholder