import time
import pandas as pd
import yfinance as yf
from datetime import datetime
import threading
from typing import Optional, List
from queue import Queue
import pytz
from pathlib import Path
from config.settings import PROCESSED_DATA_DIR
from config.logging_config import logger

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

class LiveDataHandler:
    """LiveDataHandler that maintains separate files for each ticker."""

    VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m']
    MAX_RETRIES = 3

    def __init__(self,
                 tickers: List[str],
                 update_interval: int = 60,
                 data_interval: str = '1m',
                 max_stored_rows: int = 5000,
                 archive_data: bool = True,
                 archive_threshold: int = None):
        """
        Initialize the LiveDataHandler.

        Args:
            tickers: List of ticker symbols
            update_interval: Seconds between updates
            data_interval: Data granularity ('1m', '5m', etc.)
            max_stored_rows: Maximum rows to keep in memory
            archive_data: Whether to archive old data
        """
        if data_interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of {self.VALID_INTERVALS}")

        self.tickers = tickers
        self.update_interval = max(update_interval, 60)
        self.data_interval = data_interval
        self.max_stored_rows = max_stored_rows
        self.archive_data = archive_data
        self.archive_threshold = archive_threshold or (max_stored_rows * 2)

        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()
        self._archive_lock = threading.Lock()

        self.live_data = {ticker: pd.DataFrame() for ticker in tickers}
        self._error_queue = Queue()

        self._setup_directories()
        logger.info(
            f"Initialized LiveDataHandler for {len(tickers)} tickers "
            f"with {data_interval} interval"
        )

    def _setup_directories(self) -> None:
        """Setup necessary directories."""
        self.live_data_dir = Path(PROCESSED_DATA_DIR.replace(r'\config', '')) / "live_data"
        self.archive_dir = self.live_data_dir / "archive"

        for directory in [self.live_data_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def fetch_live_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch latest live data with improved verification."""
        for attempt in range(self.MAX_RETRIES):
            try:
                data = yf.download(
                    tickers=ticker,
                    period='1d',
                    interval=self.data_interval,
                    progress=False,
                    prepost=True
                )

                if data.empty:
                    raise ValueError(f"Empty data received for {ticker}")

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data['Symbol'] = ticker
                fetch_time = datetime.now(pytz.UTC)
                data['fetch_time'] = fetch_time

                data.index = pd.to_datetime(data.index)
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                else:
                    data.index = data.index.tz_convert('UTC')

                data = data.reset_index()
                data.rename(columns={'index': 'Date', 'Datetime': 'Date'}, inplace=True)

                if len(data) > 0:
                    latest_time = data['Date'].max()
                    current_time = pd.Timestamp.now(tz='UTC')
                    time_diff = current_time - latest_time
                    print(f"\n{ticker} latest data time: {latest_time}")
                    print(f"Time difference from now: {time_diff}")

                data = data[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'fetch_time']]

                return data

            except Exception as e:
                wait_time = min(2 ** attempt, 30)
                logger.warning(
                    f"Fetch attempt {attempt + 1}/{self.MAX_RETRIES} "
                    f"failed for {ticker}. Waiting {wait_time}s. Error: {str(e)}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    self._error_queue.put(
                        f"Failed to fetch data for {ticker} after {self.MAX_RETRIES} attempts"
                    )
                    return None

    def update_live_data(self) -> None:
        """Update live data for each ticker independently."""
        logger.info("Starting live data update loop")

        while not self._stop_event.is_set():
            try:
                fetch_time = datetime.now(pytz.UTC)

                for ticker in self.tickers:
                    data = self.fetch_live_data(ticker)
                    if data is not None:
                        with self._data_lock:
                            self._update_ticker_data(ticker, data)

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                self._error_queue.put(str(e))
                time.sleep(self.update_interval)

    def _update_ticker_data(self, ticker: str, new_data: pd.DataFrame) -> None:
        """Update data for a single ticker and save to file with verification."""
        try:
            ticker_file = self.live_data_dir / f"{ticker}.csv"

            current_rows = 0
            if ticker_file.exists():
                current_data = pd.read_csv(ticker_file)
                current_rows = len(current_data)
                print(f"\nBefore update - {ticker} file has {current_rows} rows")

                current_data['Date'] = pd.to_datetime(current_data['Date'])
                if current_data['Date'].dt.tz is None:
                    current_data['Date'] = current_data['Date'].dt.tz_localize('UTC')
                else:
                    current_data['Date'] = current_data['Date'].dt.tz_convert('UTC')
            else:
                current_data = pd.DataFrame()
                print(f"\nCreating new file for {ticker}")

            print(f"New data rows for {ticker}: {len(new_data)}")

            new_data['Date'] = pd.to_datetime(new_data['Date'])

            if not current_data.empty:
                latest_current = current_data['Date'].max()
                new_records = new_data[new_data['Date'] > latest_current]
                print(f"Number of new records: {len(new_records)}")
            else:
                new_records = new_data

            if len(new_records) > 0:
                updated_data = pd.concat(
                    [current_data, new_records],
                    ignore_index=True
                ).drop_duplicates(
                    subset=['Date', 'Symbol'],
                    keep='last'
                )

                updated_data = updated_data.sort_values('Date')

                if self.archive_data and len(updated_data) >= self.archive_threshold:
                    self._archive_ticker_data(ticker, updated_data)
                    updated_data = updated_data.iloc[-self.max_stored_rows:]

                updated_data.to_csv(ticker_file, index=False)

                if ticker_file.exists():
                    verify_data = pd.read_csv(ticker_file)
                    print(f"After update - {ticker} file has {len(verify_data)} rows")
                    print(f"Added {len(verify_data) - current_rows} new rows")

                    self.live_data[ticker] = updated_data.iloc[-self.max_stored_rows:]
                else:
                    raise IOError(f"Failed to write data to {ticker_file}")
            else:
                print(f"No new data to add for {ticker}")

        except Exception as e:
            logger.error(f"Error updating {ticker} data: {str(e)}")
            self._error_queue.put(f"Update error for {ticker}: {str(e)}")
            raise

    def _archive_ticker_data(self, ticker: str, data: pd.DataFrame) -> None:
        """Archive old data to disk."""
        with self._archive_lock:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{ticker}_archive_{timestamp}.csv"
            filepath = self.archive_dir / filename

            try:
                cutoff_idx = len(data) - self.max_stored_rows
                archive_data = data.iloc[:cutoff_idx]
                archive_data.to_csv(filepath, index=False)
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

        logger.info("Live data updates stopped")

    def get_latest_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get latest data for specified tickers.

        Args:
            tickers: List of tickers (None for all)

        Returns:
            DataFrame of latest data
        """
        tickers = tickers or self.tickers
        latest_data = []

        for ticker in tickers:
            ticker_file = self.live_data_dir / f"{ticker}.csv"
            if ticker_file.exists():
                try:
                    data = pd.read_csv(ticker_file)
                    if not data.empty:
                        latest = data.iloc[-1:].copy()
                        latest_data.append(latest)
                except Exception as e:
                    logger.error(f"Error reading latest data for {ticker}: {str(e)}")

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
    """Example usage with frequent checks."""
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    handler = LiveDataHandler(
        tickers=tickers,
        update_interval=30,
        data_interval='1m',
        max_stored_rows=5000,
        archive_data=True
    )

    try:
        handler.start_live_updates()
        print("Live updates started...", flush=True)

        for i in range(12):
            time.sleep(10)
            print(f"\nCheck {i+1}/12:", flush=True)

            for ticker in tickers:
                file_path = handler.live_data_dir / f"{ticker}.csv"
                if file_path.exists():
                    data = pd.read_csv(file_path)
                    print(f"{ticker} file size: {len(data)} rows", flush=True)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal", flush=True)
    finally:
        handler.stop_live_updates()
        print("Live updates stopped", flush=True)

if __name__=="__main__":
    main()