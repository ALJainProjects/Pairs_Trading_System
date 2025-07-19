import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
import threading
from typing import Optional, List
from queue import Queue
import pytz
from pathlib import Path

# Removed PROCESSED_DATA_DIR as we will rely on DB for storage
from config.settings import REALTIME_DATABASE_URL, DATABASE_URL # Import both URLs
from config.logging_config import logger
# Import both database managers
from src.data.database import RealtimeDatabaseManager, HistoricalDatabaseManager

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

class LiveDataHandler:
    """LiveDataHandler that fetches live data, stores it in a real-time database,
    and archives historical portions to a separate historical database."""

    VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m']
    MAX_RETRIES = 3

    def __init__(self,
                 tickers: List[str],
                 update_interval: int = 60,
                 data_interval: str = '1m',
                 archive_hour_utc: int = 4, # e.g., 4 AM UTC, often after major market closes
                 archive_minute_utc: int = 0):
        """
        Initialize the LiveDataHandler.

        Args:
            tickers: List of ticker symbols
            update_interval: Seconds between updates
            data_interval: Data granularity ('1m', '5m', etc.)
            archive_hour_utc: Hour (UTC) at which to attempt daily archiving.
            archive_minute_utc: Minute (UTC) at which to attempt daily archiving.
        """
        if data_interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of {self.VALID_INTERVALS}")

        self.tickers = tickers
        self.update_interval = max(update_interval, 60)
        self.data_interval = data_interval

        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()
        self._archive_lock = threading.Lock() # New lock for archiving
        self._error_queue = Queue()

        self.realtime_db_manager = RealtimeDatabaseManager(db_uri=REALTIME_DATABASE_URL)
        self.historical_db_manager = HistoricalDatabaseManager(db_uri=DATABASE_URL) # Initialize Historical DB Manager

        self.archive_time_utc = dt_time(archive_hour_utc, archive_minute_utc, tzinfo=pytz.UTC)
        self.last_archive_date = None # Track last successful archive date (UTC) for daily trigger

        logger.info(
            f"Initialized LiveDataHandler for {len(tickers)} tickers "
            f"with {data_interval} interval. Real-time data to '{REALTIME_DATABASE_URL.split('///')[-1]}', "
            f"historical archive to '{DATABASE_URL.split('///')[-1]}'."
        )

    def fetch_live_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch latest live data with improved verification."""
        for attempt in range(self.MAX_RETRIES):
            try:
                # yfinance returns UTC aware datetime if tz=None and data.index is localized
                data = yf.download(
                    tickers=ticker,
                    period='1d', # Fetch enough data to potentially get new bars
                    interval=self.data_interval,
                    progress=False,
                    prepost=True
                )

                if data.empty:
                    raise ValueError(f"Empty data received for {ticker}")

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data['Symbol'] = ticker
                fetch_time = datetime.now(pytz.UTC) # Capture fetch time in UTC
                data['fetch_time'] = fetch_time

                data.index = pd.to_datetime(data.index)
                if data.index.tz is None:
                    # If yfinance returned naive, assume UTC as per standard practice for market data
                    data.index = data.index.tz_localize('UTC')
                else:
                    data.index = data.index.tz_convert('UTC')

                data = data.reset_index()
                # Rename 'index' or 'Datetime' to 'Date' to match our standardized format
                data.rename(columns={'index': 'Date', 'Datetime': 'Date'}, inplace=True)


                # Ensure required columns are present and in order
                required_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'fetch_time']
                for col in required_cols:
                    if col not in data.columns:
                        data[col] = pd.NA # Use pandas NA for missing values

                data = data[required_cols]

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

    def _perform_daily_archive(self) -> None:
        """
        Archives data from the real-time database to the historical database for the previous day.
        This method should be called once per day, typically after market close or at a specific UTC time.
        """
        with self._archive_lock:
            now_utc = datetime.now(pytz.UTC)
            today_utc_date = now_utc.date()

            # Determine the date to archive (the day *before* the current archive attempt)
            # If current time is past archive_time_utc, we archive yesterday's data.
            # If it's before, we might still be in yesterday for archiving purposes if the
            # last archive was two days ago (e.g. if script was off).
            archive_candidate_date = None
            if now_utc.time() >= self.archive_time_utc:
                archive_candidate_date = today_utc_date # Archive "today's" completed data (which means data from yesterday's trading session)
            else:
                archive_candidate_date = today_utc_date - timedelta(days=1) # Archive "yesterday's" data

            # Check if we already archived this date
            if self.last_archive_date == archive_candidate_date:
                logger.debug(f"Data for {archive_candidate_date} already archived. Skipping.")
                return

            # Define start and end datetimes for the data to be archived (the whole previous day)
            # We want data from `archive_candidate_date` (UTC start of day) to `archive_candidate_date` (UTC end of day)
            start_dt_to_archive = datetime.combine(archive_candidate_date, dt_time.min).replace(tzinfo=pytz.UTC)
            end_dt_to_archive = datetime.combine(archive_candidate_date, dt_time.max).replace(tzinfo=pytz.UTC)


            logger.info(f"Attempting to archive data for {archive_candidate_date} (UTC) from real-time DB to historical DB...")

            archived_any_data = False
            for ticker in self.tickers:
                try:
                    # Fetch all data for the specific day from the real-time database
                    daily_realtime_data = self.realtime_db_manager.fetch_realtime_data(
                        ticker=ticker,
                        start_datetime=start_dt_to_archive.strftime('%Y-%m-%d %H:%M:%S'),
                        end_datetime=end_dt_to_archive.strftime('%Y-%m-%d %H:%M:%S')
                    )

                    if not daily_realtime_data.empty:
                        # Prepare data for historical database upsert
                        # The historical DB uses 'Date' (Python date object) and 'Adj_Close'
                        # Real-time DB has 'datetime' and potentially no 'adj_close'
                        df_to_archive = daily_realtime_data.copy()
                        df_to_archive.rename(columns={'datetime': 'Date'}, inplace=True)

                        # Convert 'Date' column from datetime (with time) to date only
                        df_to_archive['Date'] = df_to_archive['Date'].dt.date

                        # Ensure 'Adj_Close' exists. For real-time, often 'Close' is used.
                        if 'Adj_Close' not in df_to_archive.columns:
                            df_to_archive['Adj_Close'] = df_to_archive['close']

                        # Group by Symbol and Date, then aggregate to daily bars
                        # This ensures we get one entry per symbol per day for the historical DB
                        # It also handles cases where real-time data might have multiple bars for the same date,
                        # converting them into a single daily bar for historical storage.
                        # For simplicity, we'll take the last close, first open, etc.
                        # A more robust aggregation might involve weighted averages or specific rules.
                        aggregated_daily_data = df_to_archive.groupby(['Symbol', 'Date']).agg(
                            Open=('Open', 'first'),
                            High=('High', 'max'),
                            Low=('Low', 'min'),
                            Close=('Close', 'last'),
                            Adj_Close=('Adj_Close', 'last'), # Assuming last Adj_Close is representative
                            Volume=('Volume', 'sum')
                        ).reset_index()


                        self.historical_db_manager.upsert_historical_data(aggregated_daily_data)
                        logger.info(f"Archived {len(aggregated_daily_data)} daily records for {ticker} (data from {archive_candidate_date}) to historical DB.")
                        archived_any_data = True

                        # Optional: Remove archived data from real-time database to keep it lean
                        # This can be done by fetching IDs of the archived data and then deleting them.
                        # For simplicity, we'll just log this as a potential step.
                        # CAUTION: Deleting from a DB can be slow and risky. Only do if necessary.
                        # E.g., self.realtime_db_manager.delete_data_before_datetime(end_dt_to_archive)
                        # For now, we won't implement actual deletion, relying on the real-time DB growing.

                    else:
                        logger.debug(f"No real-time data found for {ticker} on {archive_candidate_date} to archive.")

                except Exception as e:
                    logger.error(f"Error archiving data for {ticker} for date {archive_candidate_date}: {e}")
                    self._error_queue.put(f"Archive error for {ticker} on {archive_candidate_date}: {str(e)}")

            if archived_any_data:
                self.last_archive_date = archive_candidate_date
                logger.info(f"Successfully completed archiving for {archive_candidate_date}.")
            else:
                logger.info(f"No new data was archived for {archive_candidate_date}.")


    def update_live_data(self) -> None:
        """Update live data for each ticker, storing to the real-time database,
        and perform daily archiving if needed."""
        logger.info("Starting live data update loop")

        while not self._stop_event.is_set():
            try:
                for ticker in self.tickers:
                    data = self.fetch_live_data(ticker)
                    if data is not None and not data.empty:
                        with self._data_lock:
                            self.realtime_db_manager.upsert_realtime_data(data)
                            logger.info(f"Upserted {len(data)} rows for {ticker} into real-time DB.")
                    elif data is not None and data.empty:
                        logger.warning(f"Fetch for {ticker} returned empty data, nothing to upsert.")

                # Check for daily archiving trigger
                now_utc = datetime.now(pytz.UTC)
                if now_utc.time() >= self.archive_time_utc and \
                   (self.last_archive_date is None or self.last_archive_date < now_utc.date()):
                    self._perform_daily_archive()

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in live data update loop: {str(e)}")
                self._error_queue.put(str(e))
                time.sleep(self.update_interval)

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
        Get latest data for specified tickers from the real-time database.

        Args:
            tickers: List of tickers (None for all initialized tickers)

        Returns:
            DataFrame of latest data
        """
        tickers = tickers or self.tickers
        all_latest_data = pd.DataFrame()
        for ticker in tickers:
            # Fetch the latest bar for each ticker
            latest_bar_df = self.realtime_db_manager.fetch_realtime_data(ticker=ticker, limit=1)
            if not latest_bar_df.empty:
                all_latest_data = pd.concat([all_latest_data, latest_bar_df], ignore_index=True)

        if all_latest_data.empty:
            logger.warning("No live data available in the real-time database.")
        return all_latest_data

    def get_historical_live_data(self, ticker: str, start_datetime: str = None, end_datetime: str = None) -> pd.DataFrame:
        """
        Get historical range of live data for a specific ticker from the real-time database.

        Args:
            ticker (str): The ticker symbol.
            start_datetime (str, optional): Start datetime in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str, optional): End datetime in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            pd.DataFrame: DataFrame of live data history.
        """
        return self.realtime_db_manager.fetch_realtime_data(ticker=ticker, start_datetime=start_datetime, end_datetime=end_datetime)


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
        update_interval=60, # Increased update_interval for real-world testing. yfinance has rate limits.
        data_interval='1m',
        archive_hour_utc=1 # Archive shortly after midnight UTC, adjust as needed for specific market close times
    )

    try:
        handler.start_live_updates()
        print("Live updates started...", flush=True)

        # Run for a longer period to potentially trigger archiving if crossing midnight UTC
        # For testing, you might need to adjust your system clock or the archive_hour_utc
        # to ensure the archiving logic is hit within your test run.
        for i in range(120): # Run for 2 hours (120 * 60 seconds)
            time.sleep(60) # Wait for one update cycle
            print(f"\nCheck {i+1}:", flush=True)

            latest_data = handler.get_latest_data()
            if not latest_data.empty:
                print("Latest fetched data (from real-time DB):", flush=True)
                print(latest_data[['symbol', 'datetime', 'close', 'volume']], flush=True)
            else:
                print("No latest data available yet in real-time DB.", flush=True)

            # Example of fetching historical real-time data
            if i % 10 == 0 and i > 0: # Check history every 10 minutes
                print("\nFetching historical 1-minute data for AAPL from real-time DB (last 10 min):", flush=True)
                end_dt_str = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
                start_dt_str = (datetime.now(pytz.UTC) - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                aapl_live_history = handler.get_historical_live_data(
                    ticker='AAPL',
                    start_datetime=start_dt_str,
                    end_datetime=end_dt_str
                )
                if not aapl_live_history.empty:
                    print(aapl_live_history[['symbol', 'datetime', 'close', 'volume', 'fetch_time']], flush=True)
                else:
                    print("No recent historical live data for AAPL yet in real-time DB.", flush=True)


    except KeyboardInterrupt:
        print("\nReceived interrupt signal", flush=True)
    finally:
        handler.stop_live_updates()
        print("Live updates stopped", flush=True)

if __name__=="__main__":
    main()