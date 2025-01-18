import traceback
import pandas as pd
from typing import List
from datetime import datetime, timedelta
import plotly.graph_objects as go
import streamlit as st

from src.data.downloader import DataDownloader
from src.data.preprocessor import Preprocessor
from src.data.database import DatabaseManager
from src.utils.validation import validate_dataframe


class EnhancedDataLoader:
    """Enhanced data loading component with comprehensive session state management."""

    def __init__(self):
        """Initialize the data loader with required components."""
        self.preprocessor = Preprocessor()
        self.db_manager = DatabaseManager()
        self.downloader = DataDownloader()
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize required session state variables."""
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = pd.DataFrame()
        if 'data_loading_error' not in st.session_state:
            st.session_state.data_loading_error = None
        if 'data_loading_success' not in st.session_state:
            st.session_state.data_loading_success = False

    def render(self):
        """Render the enhanced data loading interface."""
        st.header("Data Loading and Preprocessing")

        tab1, tab2, tab3 = st.tabs(["File Upload", "Direct Download", "Database Query"])

        with tab1:
            self._render_file_upload()

        with tab2:
            self._render_direct_download()

        with tab3:
            self._render_database_query()

        if st.session_state.data_loading_error:
            st.error(st.session_state.data_loading_error)
        if st.session_state.data_loading_success:
            st.success("Data loaded successfully!")

    def _render_file_upload(self):
        """Handle file uploads with improved error handling and state management."""
        st.subheader("Upload Data Files")

        uploaded_files = st.file_uploader(
            "Upload CSV files (one per ticker)",
            type="csv",
            accept_multiple_files=True
        )

        if uploaded_files:
            try:
                with st.spinner("Processing uploaded files..."):
                    dfs = []
                    for file in uploaded_files:
                        df = pd.read_csv(file)
                        df['ticker'] = file.name.split('.')[0]
                        dfs.append(df)

                    combined_df = pd.concat(dfs, ignore_index=True)
                    processed_df = self._process_data(combined_df)

                    self._store_data(processed_df)

                    st.session_state.data_loading_success = True
                    st.session_state.data_loading_error = None

                    self._display_data_summary(processed_df)

            except Exception as e:
                st.session_state.data_loading_error = f"Error processing files: {str(e)}"
                st.session_state.data_loading_success = False

    def _render_direct_download(self):
        """Handle direct data downloads with improved state management."""
        st.subheader("Download Market Data")

        ticker_input_method = st.radio(
            "Ticker Input Method",
            ["Manual Input", "CSV Upload", "NASDAQ 100", "S&P 500"]
        )

        tickers = self._handle_ticker_input(ticker_input_method)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365 * 2)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )

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

            if st.button("Download Data"):
                self._handle_data_download(
                    tickers,
                    start_date,
                    end_date,
                    batch_size,
                    include_validation
                )

    @staticmethod
    def _get_sp500_tickers() -> List[str]:
        """Get S&P 500 tickers from SSGA website."""
        try:
            url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
            holdings = pd.read_excel(url, engine='openpyxl', skiprows=4)
            return holdings['Ticker'].dropna().tolist()
        except Exception as e:
            st.error(f"Error fetching S&P 500 components: {str(e)}")
            return []

    def _handle_ticker_input(self, input_method: str) -> List[str]:
        """Handle different ticker input methods."""
        tickers = []

        if input_method == "Manual Input":
            tickers_input = st.text_input(
                "Enter ticker symbols (comma-separated)",
                "AAPL,MSFT,GOOGL"
            )
            tickers = [t.strip() for t in tickers_input.split(",")]

        elif input_method == "CSV Upload":
            ticker_file = st.file_uploader(
                "Upload CSV file with ticker symbols",
                type="csv"
            )
            if ticker_file:
                try:
                    ticker_df = pd.read_csv(ticker_file)
                    ticker_col = next(
                        (col for col in ticker_df.columns
                         if col.lower() in ['ticker', 'symbol', 'tickers', 'symbols']),
                        None
                    )
                    if ticker_col:
                        tickers = ticker_df[ticker_col].dropna().unique().tolist()
                except Exception as e:
                    st.error(f"Error reading ticker CSV: {str(e)}")

        elif input_method == "NASDAQ 100":
            try:
                with st.spinner("Fetching NASDAQ 100 components..."):
                    tickers = self.downloader.get_nasdaq100_components()
            except Exception as e:
                st.error(f"Error fetching NASDAQ 100 components: {str(e)}")

        else:
            try:
                with st.spinner("Fetching S&P 500 components..."):
                    tickers = self._get_sp500_tickers()
            except Exception as e:
                st.error(f"Error fetching S&P 500 components: {str(e)}")

        return tickers

    def _handle_data_download(
            self,
            tickers: List[str],
            start_date: datetime,
            end_date: datetime,
            batch_size: int,
            include_validation: bool
    ):
        """Handle the data download process with proper state management."""
        try:
            with st.spinner("Downloading market data..."):
                df = self.downloader.download_historical_data(
                    tickers,
                    start_date=start_date,
                    end_date=end_date,
                )

                if df is not None and not df.empty:
                    if include_validation:
                        df = self._process_data(df)

                    self._store_data(df)

                    st.session_state.data_loading_success = True
                    st.session_state.data_loading_error = None

                    self._display_data_summary(df)
                else:
                    st.session_state.data_loading_error = "No data was downloaded. Please check your inputs."
                    st.session_state.data_loading_success = False

        except Exception as e:
            st.session_state.data_loading_error = f"Error downloading data: {str(e)}"
            st.session_state.data_loading_success = False
            st.error(traceback.format_exc())

    def _render_database_query(self):
        """Handle database queries with improved state management."""
        st.subheader("Query Historical Data")

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
                    tickers = [t.strip() for t in ticker_filter.split(",")] if ticker_filter else None

                    df = pd.DataFrame()
                    for ticker in (tickers or [None]):
                        ticker_df = self.db_manager.fetch_historical_data(
                            ticker=ticker,
                            start_date=min_date,
                            end_date=max_date
                        )
                        if not ticker_df.empty:
                            df = pd.concat([df, ticker_df])

                    if not df.empty:
                        processed_df = self._process_data(df)
                        self._store_data(processed_df)
                        self._display_data_summary(processed_df)

                        st.session_state.data_loading_success = True
                        st.session_state.data_loading_error = None
                    else:
                        st.session_state.data_loading_error = "No data found for the specified criteria."
                        st.session_state.data_loading_success = False

            except Exception as e:
                st.session_state.data_loading_error = f"Database query error: {str(e)}"
                st.session_state.data_loading_success = False

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with validation and cleaning."""
        if not validate_dataframe(df, methods=['missing', 'dtype']):
            st.warning("Data validation warnings detected. See logs for details.")

        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.normalize_data(df, method="z-score")
        df = self.preprocessor.handle_outliers(df)

        return df

    def _store_data(self, df: pd.DataFrame):
        """Store processed data in session state and database."""
        if df is None or df.empty:
            raise ValueError("Cannot store empty DataFrame")

        required_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                if col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df['Adj_Close']
                elif col == 'Volume':
                    df[col] = 0

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(['Symbol', 'Date'])

        st.session_state.historical_data = df.copy()

        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            self.db_manager.insert_historical_data(ticker, ticker_data)

    def _display_data_summary(self, df: pd.DataFrame):
        """Display comprehensive data summary with visualizations."""
        st.subheader("Data Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Tickers", len(df['ticker'].unique()))
        col2.metric("Date Range", f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
        col3.metric("Total Records", len(df))

        fig = go.Figure()
        for ticker in df['ticker'].unique()[:5]:
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
