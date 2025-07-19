import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.data.downloader import DataDownloader
from src.data.preprocessor import Preprocessor
from src.data.database import DatabaseManager


def render_data_loader_page():
    """Renders the UI for loading, downloading, and preprocessing data."""
    st.title("📂 Data Loader")

    downloader = DataDownloader()
    preprocessor = Preprocessor()
    db_manager = DatabaseManager()

    source = st.radio("Select Data Source", ["Download from API", "Upload CSV", "Load from Database"])

    if source == "Download from API":
        st.subheader("Download Market Data via API")

        symbols_input = st.text_input("Enter symbols (comma-separated)", "AAPL,MSFT,GOOG,GOOGL,NVDA,TSLA")

        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=5 * 365))
        end_date = col2.date_input("End Date", datetime.now())

        if st.button("Download Data"):
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            if symbols:
                with st.spinner(f"Downloading data for {len(symbols)} symbols..."):
                    df = downloader.download_historical_data(symbols, start_date, end_date)
                    if not df.empty:
                        st.session_state.historical_data = df
                        st.success(f"Successfully downloaded {len(df)} rows.")
                        st.dataframe(df.head())
                    else:
                        st.error("Failed to download data. Check symbols and date range.")

    elif source == "Upload CSV":
        st.subheader("Upload Historical Data")
        uploaded_files = st.file_uploader("Upload one or more CSV files", accept_multiple_files=True, type="csv")
        if uploaded_files:
            dfs = []
            for file in uploaded_files:
                df = pd.read_csv(file)
                if 'Symbol' not in df.columns and 'symbol' not in df.columns:
                    df['Symbol'] = file.name.split('.')[0].upper()
                dfs.append(df)

            combined_df = pd.concat(dfs, ignore_index=True)
            st.session_state.historical_data = combined_df
            st.success(f"Successfully loaded {len(combined_df)} rows from {len(uploaded_files)} files.")
            st.dataframe(combined_df.head())

    elif source == "Load from Database":
        st.subheader("Load Data from Local Database")
        if st.button("Load All Data"):
            with st.spinner("Querying database..."):
                df = db_manager.fetch_historical_data()
                if not df.empty:
                    st.session_state.historical_data = df
                    st.success(f"Successfully loaded {len(df)} rows from the database.")
                    st.dataframe(df.head())
                else:
                    st.warning("No data found in the database.")

    if not st.session_state.historical_data.empty:
        st.markdown("---")
        st.subheader("Data Preprocessing")

        with st.spinner("Cleaning data..."):
            cleaned_df = preprocessor.clean_data(st.session_state.historical_data)

        st.write("Outlier Handling (Winsorizing):")
        if st.checkbox("Apply Outlier Handling", value=True):
            with st.spinner("Handling outliers..."):
                cleaned_df = preprocessor.handle_outliers(cleaned_df)

        if st.button("Save Processed Data to Session"):
            st.session_state.historical_data = cleaned_df

            pivot_prices = cleaned_df.pivot(index='Date', columns='Symbol', values='Adj_Close').ffill().bfill()
            st.session_state.pivot_prices = pivot_prices

            st.success("Data has been cleaned and is ready for analysis.")