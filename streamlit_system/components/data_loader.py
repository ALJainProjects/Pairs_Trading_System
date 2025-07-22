import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.data.downloader import DataDownloader
from src.data.preprocessor import Preprocessor
# Changed import from DatabaseManager to HistoricalDatabaseManager
from src.data.database import HistoricalDatabaseManager
from src.utils.validation import validate_dataframe  # Import the validation function

# Configure Streamlit's logger if it's not already set up by config/logging_config.py
# This ensures Streamlit's internal logs show up in the console.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _validate_and_store_data(df: pd.DataFrame, source_name: str) -> bool:
    """
    Validates the DataFrame and stores it in session state if valid.
    This helper function ensures consistency across all data loading methods.

    Args:
        df (pd.DataFrame): The DataFrame to validate and store.
        source_name (str): A string indicating the source of the data (e.g., "API", "Uploaded File").

    Returns:
        bool: True if validation passed and data was stored, False otherwise.
    """
    if df.empty:
        st.error(f"No data to process from {source_name}. DataFrame is empty.")
        return False

    # Ensure 'Date' column is in datetime format and sort
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates_count = df['Date'].isna().sum()
            if invalid_dates_count > 0:
                st.warning(
                    f"Found {invalid_dates_count} rows with invalid dates in '{source_name}'. These rows will be dropped during validation.")
                df.dropna(subset=['Date'], inplace=True)  # Drop rows with NaT
            df = df.sort_values(by=['Date', 'Symbol']).reset_index(drop=True)
        except Exception as e:
            st.error(
                f"Failed to convert 'Date' column to datetime. Please check date format in your {source_name} data. Error: {e}")
            logger.error(f"Date conversion error for {source_name}: {e}")
            return False
    else:
        st.error(f"Missing 'Date' column in {source_name} data. Cannot proceed.")
        logger.error(f"Missing 'Date' column in {source_name}.")
        return False

    if 'Symbol' not in df.columns:
        st.error(f"Missing 'Symbol' column in {source_name} data. Cannot proceed.")
        logger.error(f"Missing 'Symbol' column in {source_name}.")
        return False

    # Perform data quality validation using the dedicated module
    st.info(f"Running data quality checks on data from {source_name}...")
    # `validate_dataframe` will log details about missing, dtype issues, and outliers
    if validate_dataframe(df, methods=['missing', 'dtype', 'outlier']):
        st.session_state.historical_data = df
        st.success(f"Successfully loaded and validated {len(df)} rows from {source_name}.")
        st.dataframe(df.head())  # Show head of the raw, validated data
        logger.info(f"Data from {source_name} successfully loaded and passed initial validation.")
        return True
    else:
        st.error(
            f"Data from {source_name} failed validation. Please check the logs in your terminal for details and adjust your data/parameters.")
        logger.error(f"Data from {source_name} failed validation.")
        return False


def render_data_loader_page():
    """Renders the UI for loading, downloading, and preprocessing data."""
    st.title("📂 Data Loader")

    # Initialize session state variables if they don't exist
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    if 'pivot_prices' not in st.session_state:
        st.session_state.pivot_prices = pd.DataFrame()

    preprocessor = Preprocessor()
    # Instantiate HistoricalDatabaseManager
    db_manager = HistoricalDatabaseManager()

    # --- Global Parallelization settings for DataDownloader (in sidebar) ---
    st.sidebar.subheader("Download Settings")
    use_parallel_download = st.sidebar.checkbox("Enable Parallel Downloading", value=True,
                                                help="Download multiple symbols concurrently.")
    num_parallel_workers = st.sidebar.slider("Number of Parallel Workers", 1, 20, 5,
                                             help="Number of concurrent threads/processes for downloading. More workers might hit API rate limits faster.")

    # Initialize DataDownloader with selected parallelization settings
    # This instance will be used across all download sections
    data_downloader = DataDownloader(
        source='yfinance',  # Default source, can be overridden per section
        use_parallel=use_parallel_download,
        num_parallel_workers=num_parallel_workers
    )

    source_type = st.radio("Select Data Source Type",
                           ["API Download", "Index Components Download", "File Upload", "Load from Database"])

    # --- API Download Section (Individual Symbols) ---
    if source_type == "API Download":
        st.subheader("Download Individual Symbols via API")

        api_source = st.selectbox("Select API Provider", ["Yahoo Finance", "Alpaca", "Polygon.io"],
                                  key="api_provider_individual")
        st.info(f"💡 Ensure your API keys for {api_source} are configured in your `.env` file "
                f"(e.g., `ALPACA_API_KEY`, `POLYGON_API_KEY`).")

        symbols_input = st.text_input("Enter symbols (comma-separated, e.g., AAPL,MSFT)", "AAPL,MSFT,GOOG,GOOGL",
                                      key="symbols_input_individual")

        col1, col2 = st.columns(2)
        # Set sensible default dates for download
        default_end_date = datetime.now().date()
        default_start_date = default_end_date - timedelta(days=5 * 365)  # 5 years of data
        start_date = col1.date_input("Start Date", default_start_date, key="api_download_start_date")
        end_date = col2.date_input("End Date", default_end_date, key="api_download_end_date")

        if st.button(f"Download Selected Symbols from {api_source}"):
            # Clear previous data from session state to avoid mixing data
            st.session_state.historical_data = pd.DataFrame()
            st.session_state.pivot_prices = pd.DataFrame()

            provider_map = {
                "Yahoo Finance": "yfinance",
                "Alpaca": "alpaca",
                "Polygon.io": "polygon"
            }
            # Set the provider for the downloader based on selection
            data_downloader.provider = data_downloader._get_provider(provider_map[api_source])
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

            if symbols:
                with st.spinner(
                        f"Downloading data from {api_source} and storing to DB... This may take a while for Polygon due to rate limits."):
                    try:
                        df = data_downloader.download_and_store_data(symbols, start_date, end_date)
                        _validate_and_store_data(df, f"API ({api_source})")
                    except Exception as e:
                        st.error(
                            f"Failed to download data from {api_source}. Error: {e}. Check symbols, date range, and API keys.")
                        logger.error(f"API Download error from {api_source}: {e}")
            else:
                st.warning("Please enter at least one symbol to download.")

        st.markdown("---")
        st.write("For a **full 20-year history** (Polygon.io only, requires premium API key):")
        if st.button("Download Full 20-Year History (Polygon Only)"):
            # Clear previous data
            st.session_state.historical_data = pd.DataFrame()
            st.session_state.pivot_prices = pd.DataFrame()

            data_downloader.provider = data_downloader._get_provider("polygon")  # Set provider to Polygon
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            start_20_years = datetime.now().date() - timedelta(days=20 * 365)
            end_20_years = datetime.now().date()

            if symbols:
                with st.spinner(
                        "Downloading 20 years of data from Polygon... This will take a long time due to API rate limits."):
                    try:
                        df = data_downloader.download_and_store_data(symbols, start_20_years, end_20_years)
                        _validate_and_store_data(df, "Polygon 20-Year API")
                    except Exception as e:
                        st.error(f"Failed to download 20-year history from Polygon. Error: {e}.")
                        logger.error(f"Polygon 20-year download error: {e}")
            else:
                st.warning("Please enter at least one symbol to download 20-year history.")

    # --- NEW: Index Components Download Section ---
    elif source_type == "Index Components Download":
        st.subheader("Download Index Components Data")
        st.info("💡 Select an index to download historical data for all its constituent symbols. "
                "The system will first try to load components from local CSV files provided, then fall back to web scraping if not found.")

        index_options = {
            "NASDAQ 100": data_downloader.get_nasdaq100_components,
            "S&P 500": data_downloader.get_sp500_components,
            "Russell 2000": data_downloader.get_russell2000_components
        }

        selected_index_name = st.selectbox(
            "Choose an Index to download components:",
            list(index_options.keys()),
            key="index_selection"
        )

        api_source_index = st.selectbox(
            "Select API Provider for Index Components Data:",
            ["Yahoo Finance", "Alpaca", "Polygon.io"],
            key="api_provider_index_components"
        )
        st.info(
            f"💡 The chosen API provider ({api_source_index}) will be used to download the historical data for the selected index's constituents.")

        col1_idx, col2_idx = st.columns(2)
        default_end_date_idx = datetime.now().date()
        default_start_date_idx = default_end_date_idx - timedelta(days=5 * 365)
        start_date_idx = col1_idx.date_input("Start Date", default_start_date_idx, key="idx_download_start_date")
        end_date_idx = col2_idx.date_input("End Date", default_end_date_idx, key="idx_download_end_date")

        if st.button(f"Download {selected_index_name} Components"):
            if start_date_idx >= end_date_idx:
                st.error("Start Date must be before End Date.")
            else:
                st.write(
                    f"Downloading historical data for {selected_index_name} components using {api_source_index}...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Set the downloader's provider based on the selected API source for index components
                    provider_map = {
                        "Yahoo Finance": "yfinance",
                        "Alpaca": "alpaca",
                        "Polygon.io": "polygon"
                    }
                    data_downloader.provider = data_downloader._get_provider(provider_map[api_source_index])

                    # Get the components list (this part still prioritizes local CSVs, then web scraping)
                    get_components_func = index_options[selected_index_name]
                    symbols_to_download = get_components_func()

                    if not symbols_to_download:
                        st.warning(f"No symbols found for {selected_index_name}. Download aborted.")
                        logger.warning(f"No symbols retrieved for {selected_index_name}.")
                        progress_bar.empty()
                        status_text.empty()
                        return

                    status_text.text(
                        f"Found {len(symbols_to_download)} symbols in {selected_index_name}. Starting download via {api_source_index}.")
                    logger.info(
                        f"Attempting to download data for {len(symbols_to_download)} symbols from {selected_index_name} via {api_source_index}.")

                    # Clear previous data from session state before starting new download
                    st.session_state.historical_data = pd.DataFrame()
                    st.session_state.pivot_prices = pd.DataFrame()

                    # The download_and_store_data method now handles parallelization internally
                    combined_df = data_downloader.download_and_store_data(
                        symbols=symbols_to_download,
                        start_date=start_date_idx,
                        end_date=end_date_idx
                    )

                    # Update progress bar based on actual downloaded symbols count (if available from downloader)
                    # For now, a simplified update to 100% on completion.
                    if not combined_df.empty:
                        total_symbols_downloaded_this_session = combined_df['Symbol'].nunique()
                    else:
                        total_symbols_downloaded_this_session = 0

                    progress_bar.progress(1.0)  # Assume 100% if download completes without unhandled exception

                    if not combined_df.empty:
                        _validate_and_store_data(combined_df,
                                                 f"{selected_index_name} Components via {api_source_index}")
                        st.success(
                            f"Successfully downloaded and stored historical data for {total_symbols_downloaded_this_session} unique symbols from {selected_index_name} into the historical database using {api_source_index}!")
                        logger.info(
                            f"Completed download for {selected_index_name}: {total_symbols_downloaded_this_session} unique symbols via {api_source_index}.")
                    else:
                        st.warning(f"No data was downloaded for {selected_index_name} components.")
                        logger.warning(f"No data downloaded for {selected_index_name} components.")

                except Exception as e:
                    st.error(f"An error occurred during index components download: {e}")
                    logger.exception(f"Error downloading {selected_index_name} components' data.")
                finally:
                    progress_bar.empty()
                    status_text.empty()
                    # download_status_container was removed in the previous step, so no change needed here.


    # --- File Upload Section ---
    elif source_type == "File Upload":
        st.subheader("Upload Historical Data (CSV)")
        st.info(
            "💡 Upload one or more CSV files. Each file should have 'Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume' columns. "
            "If 'Symbol' column is missing, the filename (e.g., AAPL.csv -> AAPL) will be used as the symbol.")
        uploaded_files = st.file_uploader("Upload one or more CSV files", accept_multiple_files=True, type="csv")
        if uploaded_files:
            # Clear previous data
            st.session_state.historical_data = pd.DataFrame()
            st.session_state.pivot_prices = pd.DataFrame()

            dfs = []
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    # Infer Symbol if missing
                    if 'Symbol' not in df.columns and 'symbol' not in df.columns:
                        df['Symbol'] = file.name.split('.')[0].upper()
                    # Normalize common column names to expected case (e.g., 'date' -> 'Date', 'adj close' -> 'Adj_Close')
                    df.columns = [
                        col.capitalize() if col.lower() in ['date', 'open', 'high', 'low', 'close', 'volume'] else col
                        for col in df.columns]
                    df.columns = [
                        col.replace('Adj close', 'Adj_Close').replace('adj close', 'Adj_Close').replace('Adj_close',
                                                                                                        'Adj_Close').replace('Close', 'Adj_Close') for
                        col in df.columns]  # Specific for Adj_Close
                    dfs.append(df)
                    st.success(f"Successfully read {file.name}.")
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}. Please ensure it's a valid CSV with correct headers.")
                    logger.error(f"File upload error for {file.name}: {e}")

            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                _validate_and_store_data(combined_df, f"Uploaded File(s) ({len(uploaded_files)} files)")
            else:
                st.warning("No valid files were uploaded.")

    # --- Load from Database Section ---
    elif source_type == "Load from Database":
        st.subheader("Load Data from Local Database")
        if st.button("Load All Data from DB"):
            # Clear previous data
            st.session_state.historical_data = pd.DataFrame()
            st.session_state.pivot_prices = pd.DataFrame()

            with st.spinner("Querying database..."):
                try:
                    # fetch_historical_data without arguments loads all data
                    df = db_manager.fetch_historical_data()
                    if not df.empty:
                        _validate_and_store_data(df, "Database")
                    else:
                        st.warning("No data found in the database. Try downloading some first.")
                        logger.info("Database returned no data.")
                except Exception as e:
                    st.error(f"Error loading data from database: {e}. Check database connection and if data exists.")
                    logger.error(f"Database load error: {e}")

    # --- Data Preprocessing Section ---
    # Only show if there's data in session state (meaning raw data has been loaded/validated)
    if not st.session_state.historical_data.empty:
        st.markdown("---")
        st.subheader("📊 Data Preprocessing")
        st.info(
            f"Current raw data loaded: {len(st.session_state.historical_data)} rows across {st.session_state.historical_data['Symbol'].nunique()} symbols. Ready for cleaning.")

        if st.button("Process and Prepare Data for Analysis"):
            with st.spinner("Cleaning data, handling outliers, and preparing for analysis..."):
                df_to_process = st.session_state.historical_data.copy()  # Always work on a copy

                try:
                    # --- Preprocessing Pipeline ---
                    # 1. Clean data (e.g., sort, handle duplicates, ensure datetime)
                    df_processed = preprocessor.clean_data(df_to_process)

                    # 2. Handle missing values (e.g., ffill, bfill, interpolate) - often done before outliers if using statistical methods
                    df_processed = preprocessor.handle_missing_values(df_processed, method="ffill", groupby="Symbol")

                    # 3. Handle outliers (e.g., winsorization)
                    df_processed = preprocessor.handle_outliers(df_processed, method='zscore', threshold=3.0,
                                                                groupby="Symbol")

                    # 4. Final validation after all preprocessing steps
                    st.info("Running final data quality checks after all preprocessing steps...")
                    if validate_dataframe(df_processed,
                                          methods=['missing', 'dtype', 'outlier']):  # Rerun full validation
                        st.session_state.historical_data = df_processed  # Update session state with processed data
                        st.success(
                            "Data has been successfully cleaned, handled, and validated. Displaying head of processed data:")
                        st.dataframe(df_processed.head())  # Show head of the *processed* DataFrame
                        logger.info("Processed data passed final validation.")
                    else:
                        st.error(
                            "Processed data failed final validation. Review the cleaning/outlier handling steps and logs.")
                        logger.error("Processed data failed final validation.")
                        return  # Stop here if post-processing validation fails

                    # Create and cache the pivoted price data used by analysis and backtesting
                    st.info("Creating pivoted price data (Date x Symbol) for analysis and backtesting.")
                    pivot_prices = df_processed.pivot(index='Date', columns='Symbol', values='Adj_Close')
                    # Apply ffill/bfill to the pivoted data to handle any remaining NaNs from gaps after pivot
                    pivot_prices = pivot_prices.ffill().bfill()

                    if pivot_prices.empty:
                        st.error(
                            "Pivoted price data is empty after preprocessing. Check input data or preprocessing steps.")
                        logger.error("Pivoted price data is empty.")
                    else:
                        st.session_state.pivot_prices = pivot_prices
                        st.success(
                            f"Pivoted price data created: {pivot_prices.shape[0]} rows, {pivot_prices.shape[1]} columns. Displaying head:")
                        st.dataframe(pivot_prices.head())
                        logger.info(f"Pivoted price data created with shape: {pivot_prices.shape}.")

                except Exception as e:
                    st.error(
                        f"Error during data preprocessing: {e}. Please check your data and preprocessing parameters.")
                    logger.error(f"Data preprocessing error: {e}")

    # --- Sidebar Data Status Display ---
    if not st.session_state.historical_data.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Loaded Data Status")
        st.sidebar.write(f"**Total Rows:** {len(st.session_state.historical_data):,}")
        st.sidebar.write(f"**Unique Symbols:** {st.session_state.historical_data['Symbol'].nunique()}")
        if 'Date' in st.session_state.historical_data.columns:
            st.sidebar.write(
                f"**Date Range:** {st.session_state.historical_data['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.historical_data['Date'].max().strftime('%Y-%m-%d')}")

    if not st.session_state.pivot_prices.empty:
        st.sidebar.write(f"**Pivoted Data Shape:** {st.session_state.pivot_prices.shape}")
        st.sidebar.write(
            f"**Pivoted Date Range:** {st.session_state.pivot_prices.index.min().strftime('%Y-%m-%d')} to {st.session_state.pivot_prices.index.max().strftime('%Y-%m-%d')}")