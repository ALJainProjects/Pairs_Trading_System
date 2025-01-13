"""
Enhanced Live Data Module

This module handles the retrieval and management of live intraday data feeds with:
1. Thread-safe operations
2. Memory management
3. Clean shutdown capabilities
4. Proper timestamp handling
5. Duplicate prevention
6. Data archiving
"""

import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import threading
from typing import Dict, Optional, List, Union
from queue import Queue
import signal
import pytz
from pathlib import Path
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import logger


class LiveDataHandler:
    """LiveDataHandler with enhanced thread safety and memory management."""

    # Class constants
    VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m']
    MAX_RETRIES = 3
    ARCHIVE_THRESHOLD = 10000  # Rows before archiving

    def __init__(self,
                 tickers: List[str],
                 update_interval: int = 60,
                 data_interval: str = '1m',
                 max_stored_rows: int = 5000,
                 archive_data: bool = True):
        """
        Initialize the LiveDataHandler.

        Args:
            tickers: List of ticker symbols
            update_interval: Seconds between updates
            data_interval: Data granularity ('1m', '5m', etc.)
            max_stored_rows: Maximum rows to keep in memory
            archive_data: Whether to archive old data
        """
        # Validate inputs
        if data_interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of {self.VALID_INTERVALS}")

        self.tickers = tickers
        self.update_interval = max(update_interval, 60)  # Minimum 60s
        self.data_interval = data_interval
        self.max_stored_rows = max_stored_rows
        self.archive_data = archive_data

        # Thread control
        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()
        self._archive_lock = threading.Lock()

        # Data storage
        self.live_data = {ticker: pd.DataFrame() for ticker in tickers}
        self._error_queue = Queue()

        # Setup directories
        self._setup_directories()
        logger.info(
            f"Initialized LiveDataHandler for {len(tickers)} tickers "
            f"with {data_interval} interval"
        )

    def _setup_directories(self) -> None:
        """Setup necessary directories."""
        self.live_data_dir = Path(PROCESSED_DATA_DIR) / "live_data"
        self.archive_dir = self.live_data_dir / "archive"

        for directory in [self.live_data_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def fetch_live_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch latest live data with retries.

        Returns:
            DataFrame of latest data or None on failure
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                data = yf.download(
                    tickers=self.tickers,
                    period='1d',
                    interval=self.data_interval,
                    progress=False,
                    prepost=True  # Include pre/post market
                )

                if data.empty:
                    raise ValueError("Empty data received")

                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [
                        '_'.join(col).strip()
                        for col in data.columns.values
                    ]

                # Ensure datetime index is timezone-aware
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')

                return data

            except Exception as e:
                logger.warning(
                    f"Fetch attempt {attempt + 1}/{self.MAX_RETRIES} "
                    f"failed: {str(e)}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self._error_queue.put(
                        f"Failed to fetch data after {self.MAX_RETRIES} attempts"
                    )
                    return None

    def update_live_data(self) -> None:
        """Update live data with thread safety and memory management."""
        logger.info("Starting live data update loop")

        while not self._stop_event.is_set():
            try:
                data = self.fetch_live_data()
                if data is None:
                    time.sleep(self.update_interval)
                    continue

                fetch_time = datetime.now(pytz.UTC)

                with self._data_lock:
                    for ticker in self.tickers:
                        # Extract and clean ticker data
                        ticker_data = self._process_ticker_data(
                            data, ticker, fetch_time
                        )

                        if ticker_data is not None:
                            self._update_ticker_data(ticker, ticker_data)

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                self._error_queue.put(str(e))
                time.sleep(self.update_interval)

    def _process_ticker_data(self,
                             data: pd.DataFrame,
                             ticker: str,
                             fetch_time: datetime) -> Optional[pd.DataFrame]:
        """Process data for a single ticker."""
        try:
            # Extract relevant columns
            ticker_cols = [col for col in data.columns if ticker in col]
            ticker_data = data[ticker_cols].copy()

            if ticker_data.empty:
                return None

            # Clean column names
            ticker_data.columns = [
                col.replace(f"{ticker}_", "")
                for col in ticker_data.columns
            ]

            # Add metadata
            ticker_data['ticker'] = ticker
            ticker_data['fetch_time'] = fetch_time

            return ticker_data

        except Exception as e:
            logger.error(f"Error processing {ticker} data: {str(e)}")
            return None

    def _update_ticker_data(self,
                            ticker: str,
                            new_data: pd.DataFrame) -> None:
        """Update data for a single ticker with memory management."""
        current_data = self.live_data[ticker]

        # Combine data and remove duplicates
        updated_data = pd.concat(
            [current_data, new_data],
            ignore_index=False
        ).drop_duplicates(
            subset=['ticker', new_data.index.name],
            keep='last'
        )

        # Sort by timestamp
        updated_data = updated_data.sort_index()

        # Archive if necessary
        if (self.archive_data and
                len(updated_data) >= self.ARCHIVE_THRESHOLD):
            self._archive_ticker_data(ticker, updated_data)

        # Trim to max rows
        if len(updated_data) > self.max_stored_rows:
            updated_data = updated_data.iloc[-self.max_stored_rows:]

        self.live_data[ticker] = updated_data

    def _archive_ticker_data(self,
                             ticker: str,
                             data: pd.DataFrame) -> None:
        """Archive old data to disk."""
        with self._archive_lock:
            # Create archive filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{ticker}_archive_{timestamp}.parquet"
            filepath = self.archive_dir / filename

            try:
                # Keep only older data for archiving
                cutoff_idx = len(data) - self.max_stored_rows
                archive_data = data.iloc[:cutoff_idx]
                archive_data.to_parquet(filepath)
                logger.info(f"Archived {len(archive_data)} rows for {ticker}")

            except Exception as e:
                logger.error(f"Error archiving {ticker} data: {str(e)}")
                self._error_queue.put(f"Archive error for {ticker}: {str(e)}")

    def start_live_updates(self) -> None:
        """Start the live data update thread."""
        self._stop_event.clear()
        self.update_thread = threading.Thread(
            target=self.update_live_data,
            daemon=True
        )
        self.update_thread.start()
        logger.info("Live data update thread started")

    def stop_live_updates(self, timeout: int = 10) -> None:
        """
        Stop the live data update thread.

        Args:
            timeout: Seconds to wait for thread to stop
        """
        logger.info("Stopping live data updates...")
        self._stop_event.set()

        if hasattr(self, 'update_thread'):
            self.update_thread.join(timeout=timeout)
            if self.update_thread.is_alive():
                logger.warning("Update thread did not stop cleanly")

        # Final archive
        if self.archive_data:
            for ticker, data in self.live_data.items():
                if not data.empty:
                    self._archive_ticker_data(ticker, data)

        logger.info("Live data updates stopped")

    def get_latest_data(self,
                        tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get latest data for specified tickers.

        Args:
            tickers: List of tickers (None for all)

        Returns:
            DataFrame of latest data
        """
        with self._data_lock:
            tickers = tickers or self.tickers
            latest_data = []

            for ticker in tickers:
                if ticker in self.live_data and not self.live_data[ticker].empty:
                    latest = self.live_data[ticker].iloc[-1].to_frame().T
                    latest_data.append(latest)

            if latest_data:
                return pd.concat(latest_data, ignore_index=True)
            else:
                logger.warning("No live data available")
                return pd.DataFrame()

    def get_error_messages(self) -> List[str]:
        """Get any error messages from the update thread."""
        messages = []
        while not self._error_queue.empty():
            messages.append(self._error_queue.get())
        return messages


def main():
    """Example usage of LiveDataHandler."""
    # Example tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Initialize handler
    handler = LiveDataHandler(
        tickers=tickers,
        update_interval=60,
        data_interval='1m',
        max_stored_rows=5000,
        archive_data=True
    )

    try:
        # Start updates
        handler.start_live_updates()

        # Run for a while
        for _ in range(5):
            time.sleep(60)

            # Check for errors
            errors = handler.get_error_messages()
            if errors:
                logger.error("Errors occurred:", errors)

            # Get latest data
            latest = handler.get_latest_data()
            if not latest.empty:
                print("\nLatest Data:")
                print(latest)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        # Clean shutdown
        handler.stop_live_updates()


if __name__ == "__main__":
    main()