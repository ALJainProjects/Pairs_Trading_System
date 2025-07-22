import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Union, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed  # Import for parallelization

# External data provider imports
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from polygon.rest import RESTClient

try:
    from config.settings import RAW_DATA_DIR, DATA_DIR, ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY
    from config.logging_config import logger
    from src.data.database import HistoricalDatabaseManager
except ImportError:
    print("Warning: Could not import local config and database modules. Using mock values.")
    RAW_DATA_DIR = os.path.join(os.getcwd(), 'data', 'raw')
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', 'YOUR_ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', 'YOUR_ALPACA_SECRET_KEY')
    POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_POLYGON_API_KEY')

    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    class MockHistoricalDatabaseManager:
        def __init__(self):
            logger.info("Using MockHistoricalDatabaseManager. Data will not be persistently stored.")
            self.data_store = pd.DataFrame()

        def upsert_historical_data(self, df: pd.DataFrame):
            if df.empty:
                return
            # For a mock, we'll just append and deduplicate/sort
            logger.info(f"Mock DB: Upserting {len(df)} rows for {df['Symbol'].nunique()} symbols.")
            self.data_store = pd.concat([self.data_store, df], ignore_index=True)
            self.data_store.drop_duplicates(subset=['Date', 'Symbol'], inplace=True)
            self.data_store.sort_values(by=['Symbol', 'Date'], inplace=True)
            logger.info(f"Mock DB now has {len(self.data_store)} total rows.")


    HistoricalDatabaseManager = MockHistoricalDatabaseManager


# --- Abstract Base Class for Data Providers ---
class DataProvider(ABC):
    """Abstract interface for all data source providers."""

    @abstractmethod
    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Downloads data for a list of symbols and returns it in a standardized long-format DataFrame."""
        pass

    def _standardize_long_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ensure a common format for the final DataFrame."""
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'adj_close': 'Adj_Close', 'timestamp': 'Date', 'symbol': 'Symbol'
        }, inplace=True)

        if 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']

        # Convert to date object for consistency. If timestamp includes time, it truncates to date.
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        standard_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        return df.reindex(columns=standard_cols)


# --- Concrete Provider Implementations ---
class YFinanceProvider(DataProvider):
    """Data provider for Yahoo Finance."""

    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logger.info(
            f"YFinance: Downloading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

        df = yf.download(symbols, start=start_date, end=end_date, progress=False, auto_adjust=True, ignore_tz=True)
        if df.empty:
            logger.warning(f"YFinance: No data returned for symbols: {symbols}")
            return pd.DataFrame()

        if len(symbols) == 1:
            df = df.reset_index()
            df['Symbol'] = symbols[0]
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.stack(level=1).rename_axis(['Date', 'Symbol']).reset_index()
            else:
                logger.warning(
                    f"YFinance: Unexpected DataFrame structure from yfinance for multiple symbols. Attempting to melt.")
                if 'Symbol' not in df.columns:
                    logger.error(
                        "YFinance: Could not automatically infer symbols from columns. Skipping standardization.")
                    return pd.DataFrame()

        df.rename(columns={'Close': 'Adj_Close'}, inplace=True)
        return self._standardize_long_df(df)


class AlpacaProvider(DataProvider):
    """Data provider for Alpaca."""

    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or ALPACA_API_KEY == 'YOUR_ALPACA_API_KEY':
            raise ValueError("Alpaca API keys are not configured or are placeholder values.")
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        logger.info("AlpacaProvider initialized.")

    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logger.info(
            f"Alpaca: Downloading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            bars = self.client.get_stock_bars(request_params)
            if bars.df.empty:
                logger.warning(f"Alpaca: No data returned for symbols: {symbols}")
                return pd.DataFrame()
            return self._standardize_long_df(bars.df.reset_index())
        except Exception as e:
            logger.error(f"Alpaca: Error downloading data for {symbols}: {e}")
            return pd.DataFrame()


class PolygonProvider(DataProvider):
    """Data provider for Polygon.io, handles rate limiting."""

    def __init__(self):
        if not POLYGON_API_KEY or POLYGON_API_KEY == 'YOUR_POLYGON_API_KEY':
            raise ValueError("Polygon API key is not configured or is a placeholder value.")
        self.client = RESTClient(POLYGON_API_KEY)
        logger.info("PolygonProvider initialized.")

    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        all_bars = []
        logger.info(
            f"Polygon: Downloading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

        for symbol in symbols:
            try:
                from_ = start_date.strftime('%Y-%m-%d')
                to_ = end_date.strftime('%Y-%m-%d')

                resp = self.client.get_aggs(symbol, 1, "day", from_, to_, limit=50000)
                if not resp or not resp.results:
                    logger.warning(f"Polygon: No data returned for {symbol}.")
                    continue

                df = pd.DataFrame(resp.results)
                df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'},
                          inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                standardized_df = self._standardize_long_df(df)
                standardized_df['Symbol'] = symbol
                all_bars.append(standardized_df)

                logger.debug(f"Polygon: Waiting 12.1 seconds after downloading {symbol} to respect rate limits.")
                time.sleep(12.1)  # Polygon's free tier limit
            except Exception as e:
                logger.error(f"Polygon: Failed to download {symbol}: {e}")

        return pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()


# --- Main DataDownloader Facade ---
class DataDownloader:
    """
    Class for managing financial data downloads and processing.
    Supports various data providers and parallel downloading.
    """

    def __init__(self, source: str = 'yfinance', use_parallel: bool = True, num_parallel_workers: int = 5):
        self.provider = self._get_provider(source)
        self.db_manager = HistoricalDatabaseManager()
        self.use_parallel = use_parallel
        self.num_parallel_workers = num_parallel_workers
        self._setup_directories()
        logger.info(
            f"DataDownloader initialized. Provider: {self.provider.__class__.__name__}, Parallel: {self.use_parallel}, Workers: {self.num_parallel_workers}.")

    def _get_provider(self, source: str) -> DataProvider:
        source = source.lower()
        if source == 'alpaca':
            return AlpacaProvider()
        elif source == 'polygon':
            return PolygonProvider()
        elif source == 'yfinance':
            return YFinanceProvider()
        else:
            raise ValueError(f"Unknown data source: {source}")

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

    def _load_components_from_csv(self, filename: str) -> Optional[List[str]]:
        """Helper to load components from a specific CSV file in DATA_DIR."""
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            try:
                components_df = pd.read_csv(filepath)
                if not components_df.empty and 'Symbol' in components_df.columns:
                    logger.info(f"Loaded components from local CSV: {filename}")
                    return components_df['Symbol'].tolist()
            except Exception as e:
                logger.warning(f"Error loading components from {filename}: {e}. Will attempt fallback.")
        return None

    def get_nasdaq100_components(self) -> List[str]:
        """Get NASDAQ 100 components. Prioritizes local CSV provided by user."""
        current_date_str = datetime.now().strftime('%m-%d-%Y')
        dated_filename = f'nasdaq-100-index-{current_date_str}.csv'
        local_components = self._load_components_from_csv(dated_filename)
        if local_components: return local_components

        local_components = self._load_components_from_csv('nasdaq100_components.csv')  # Fallback to generic name
        if local_components: return local_components

        logger.info("Attempting to download NASDAQ 100 components from API...")
        try:
            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
            res = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
            res.raise_for_status()
            data = res.json()

            if not data or 'data' not in data or 'data' not in data['data'] or 'rows' not in data['data']['data']:
                raise ValueError("Unexpected JSON structure from NASDAQ API.")

            holdings = pd.DataFrame(data['data']['data']['rows'])
            holdings = holdings.rename(columns={'symbol': 'Symbol'})

            if not holdings.empty and 'Symbol' in holdings.columns:
                holdings['Symbol'].to_csv(os.path.join(DATA_DIR, dated_filename), index=False, header=True)
                logger.info(
                    f"Downloaded and saved NASDAQ 100 components to {os.path.join(DATA_DIR, dated_filename)}")
                return holdings['Symbol'].tolist()
            else:
                raise ValueError("NASDAQ API returned empty or malformed component data.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching NASDAQ 100 components: {e}")
            raise ValueError(f"Could not retrieve NASDAQ 100 components due to network error: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing NASDAQ 100 components data: {e}")
            raise ValueError(f"Could not retrieve valid NASDAQ 100 components: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred getting NASDAQ 100 components: {e}")
            raise ValueError(f"Could not retrieve valid NASDAQ 100 components: {e}")

    def get_sp500_components(self) -> List[str]:
        """Fetches S&P 500 components. Prioritizes local CSV provided by user."""
        current_date_str = datetime.now().strftime('%m-%d-%Y')
        dated_filename = f'sp-500-index-{current_date_str}.csv'
        local_components = self._load_components_from_csv(dated_filename)
        if local_components: return local_components

        local_components = self._load_components_from_csv('sp500_components.csv')  # Fallback
        if local_components: return local_components

        logger.info("Attempting to download S&P 500 components from Wikipedia...")
        try:
            html_data = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text
            soup = BeautifulSoup(html_data, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})

            if not table:
                raise ValueError("Could not find the S&P 500 table on Wikipedia page.")

            df = pd.read_html(str(table), header=0)[0]

            if 'Symbol' not in df.columns:
                df.rename(columns={'Ticker symbol': 'Symbol'}, inplace=True)

            if 'Symbol' not in df.columns or df.empty:
                raise ValueError("S&P 500 data from Wikipedia is empty or 'Symbol' column not found.")

            symbols = df['Symbol'].tolist()
            symbols = [s.replace('.', '-') for s in symbols]  # Handle common yahoo finance symbol differences

            pd.DataFrame({'Symbol': symbols}).to_csv(os.path.join(DATA_DIR, dated_filename), index=False,
                                                     header=True)
            logger.info(f"Downloaded and saved S&P 500 components to {os.path.join(DATA_DIR, dated_filename)}")
            return symbols

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching S&P 500 components from Wikipedia: {e}")
            raise ValueError(f"Could not retrieve S&P 500 components due to network error: {e}")
        except Exception as e:
            logger.error(f"Error parsing S&P 500 components from Wikipedia: {e}")
            raise ValueError(f"Could not retrieve valid S&P 500 components: {e}")

    def get_russell2000_components(self) -> List[str]:
        """Fetches Russell 2000 components. Prioritizes local CSV provided by user."""
        current_date_str = datetime.now().strftime('%m-%d-%Y')
        dated_filename = f'russell-2000-index-{current_date_str}.csv'
        local_components = self._load_components_from_csv(dated_filename)
        if local_components: return local_components

        local_components = self._load_components_from_csv('russell2000_components.csv')  # Fallback
        if local_components: return local_components

        logger.info("Attempting to download Russell 2000 components from Slickcharts (web scraping).")
        try:
            url = 'https://www.slickcharts.com/russell2000'
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            df_list = pd.read_html(response.text, match='Symbol')
            if not df_list:
                raise ValueError("Could not find table matching 'Symbol' on Slickcharts Russell 2000 page.")

            df = df_list[0]

            if 'Symbol' not in df.columns or df.empty:
                raise ValueError("Russell 2000 data from Slickcharts is empty or 'Symbol' column not found.")

            symbols = df['Symbol'].tolist()

            pd.DataFrame({'Symbol': symbols}).to_csv(os.path.join(DATA_DIR, dated_filename), index=False,
                                                     header=True)
            logger.info(
                f"Downloaded and saved Russell 2000 components from Slickcharts to {os.path.join(DATA_DIR, dated_filename)}")
            return symbols

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching Russell 2000 components from Slickcharts: {e}")
            raise ValueError(f"Could not retrieve Russell 2000 components due to network error: {e}")
        except Exception as e:
            logger.error(f"Error parsing Russell 2000 components from Slickcharts: {e}")
            raise ValueError(f"Could not retrieve valid Russell 2000 components: {e}")

    def _download_single_symbol_and_store(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Helper function to download data for a single symbol using the instance's provider
        and store it. Intended for parallel execution.
        """
        try:
            # Call the provider's download_data method with a single symbol list
            df = self.provider.download_data([symbol], start_date, end_date)

            # --- IMPORTANT: ADDED RATE LIMITING HERE FOR YFINANCE ---
            # Apply a small delay specifically if the current provider is YFinanceProvider
            # and we are running in a parallel context.
            if isinstance(self.provider, YFinanceProvider) and self.use_parallel:
                # Adjust this sleep time based on your observation and Yahoo Finance's tolerance.
                # 0.5 to 1.0 seconds is a good starting point for parallel yfinance calls.
                sleep_time = 0.75
                logger.debug(f"Sleeping for {sleep_time} seconds after downloading {symbol} (YFinance parallel).")
                time.sleep(sleep_time)
            # --- END OF RATE LIMITING ADDITION ---

            if not df.empty:
                self.db_manager.upsert_historical_data(df)
                return df
            else:
                logger.warning(f"No data downloaded for symbol {symbol} from {self.provider.__class__.__name__}.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error downloading or storing data for symbol {symbol}: {e}")
            return pd.DataFrame()

    def download_and_store_data(self,
                                symbols: List[str],
                                start_date: Union[str, datetime],
                                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Downloads historical data from the selected provider and stores/updates it in the historical database.
        Supports parallel downloading if configured.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if not symbols:
            logger.warning("No symbols provided for download.")
            return pd.DataFrame()

        logger.info(f"Starting data download using {self.provider.__class__.__name__} for {len(symbols)} symbols.")

        all_downloaded_dfs = []

        # Only use ThreadPoolExecutor if parallel is enabled AND the provider isn't Polygon (which has its own internal per-symbol sleep)
        if self.use_parallel and len(symbols) > 1 and not isinstance(self.provider, PolygonProvider):
            logger.info(f"Initiating parallel download with {self.num_parallel_workers} workers.")
            with ThreadPoolExecutor(max_workers=self.num_parallel_workers) as executor:
                futures = {executor.submit(self._download_single_symbol_and_store, symbol, start_dt, end_dt): symbol for
                           symbol in symbols}

                for i, future in enumerate(tqdm(as_completed(futures), total=len(symbols), desc="Downloading Symbols")):
                    symbol_downloaded = futures[future]
                    try:
                        df_single_symbol = future.result()
                        if not df_single_symbol.empty:
                            all_downloaded_dfs.append(df_single_symbol)
                        else:
                            logger.warning(f"No data received for {symbol_downloaded} in parallel task.")
                    except Exception as dl_e:
                        logger.error(f"Error processing parallel download result for {symbol_downloaded}: {dl_e}")

        else:
            if isinstance(self.provider, PolygonProvider) and self.use_parallel:
                logger.warning(
                    "PolygonProvider handles its own rate limiting internally. Disabling external parallelization for Polygon.")
            logger.info("Performing sequential download (or provider handles internal parallelization/batching).")

            # Use tqdm for sequential downloads to show progress for larger lists
            for symbol in tqdm(symbols, desc="Downloading Symbols Sequentially"):
                df_single_symbol = self._download_single_symbol_and_store(symbol, start_dt, end_dt)
                if not df_single_symbol.empty:
                    all_downloaded_dfs.append(df_single_symbol)

        if all_downloaded_dfs:
            combined_df = pd.concat(all_downloaded_dfs, ignore_index=True)
            logger.info(
                f"Total downloaded and stored {len(combined_df)} rows for {combined_df['Symbol'].nunique()} unique symbols.")
            return combined_df
        else:
            logger.warning("No data downloaded or stored after process completion.")
            return pd.DataFrame()


def main():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Define a smaller list of symbols for testing, as full indices can be very large
    # and hit API limits/take a long time for free tiers.
    test_symbols_sp500 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'PG', 'XOM']
    test_symbols_nasdaq = ['TSLA', 'ADBE', 'PYPL', 'INTC', 'CMCSA', 'QCOM', 'AMD', 'NFLX', 'SBUX', 'BKNG']
    test_symbols_russell = ['AMC', 'GME', 'RBLX', 'UPST', 'MRNA', 'PLTR', 'SNOW', 'CHPT', 'NIO', 'COIN']

    # You can get full index components if you uncomment these and are prepared for long downloads/API usage
    # Ensure your API keys are correctly set in config/settings.py or as environment variables.

    # --- Run with YFinance ---
    print("\n--- Running DataDownloader with YFinance ---")
    try:
        # Use more workers for YFinance as it's typically more resilient to parallel requests than Polygon's free tier
        yfinance_downloader = DataDownloader(source='yfinance', use_parallel=True, num_parallel_workers=10)

        # S&P 500 components
        # sp500_symbols = yfinance_downloader.get_sp500_components()
        # logger.info(f"YFinance: Downloading S&P 500 data for {len(sp500_symbols)} symbols.")
        # yfinance_sp500_df = yfinance_downloader.download_and_store_data(sp500_symbols, start_date, end_date)
        # logger.info(f"YFinance S&P 500 download complete. Total rows: {len(yfinance_sp500_df)}")

        logger.info(f"YFinance: Downloading sample S&P 500 data for {len(test_symbols_sp500)} symbols.")
        yfinance_sample_sp500_df = yfinance_downloader.download_and_store_data(test_symbols_sp500, start_date, end_date)
        logger.info(f"YFinance sample S&P 500 download complete. Total rows: {len(yfinance_sample_sp500_df)}")
        # print(yfinance_sample_sp500_df.head())

    except Exception as e:
        logger.error(f"YFinance Data Download Failed: {e}")

    # --- Run with Alpaca ---
    print("\n--- Running DataDownloader with Alpaca ---")
    try:
        # Alpaca's client handles batching internally, so parallelization can still be effective.
        alpaca_downloader = DataDownloader(source='alpaca', use_parallel=True, num_parallel_workers=5)

        # NASDAQ 100 components
        # nasdaq100_symbols = alpaca_downloader.get_nasdaq100_components()
        # logger.info(f"Alpaca: Downloading NASDAQ 100 data for {len(nasdaq100_symbols)} symbols.")
        # alpaca_nasdaq100_df = alpaca_downloader.download_and_store_data(nasdaq100_symbols, start_date, end_date)
        # logger.info(f"Alpaca NASDAQ 100 download complete. Total rows: {len(alpaca_nasdaq100_df)}")

        logger.info(f"Alpaca: Downloading sample NASDAQ 100 data for {len(test_symbols_nasdaq)} symbols.")
        alpaca_sample_nasdaq_df = alpaca_downloader.download_and_store_data(test_symbols_nasdaq, start_date, end_date)
        logger.info(f"Alpaca sample NASDAQ 100 download complete. Total rows: {len(alpaca_sample_nasdaq_df)}")
        # print(alpaca_sample_nasdaq_df.head())

    except ValueError as ve:
        logger.error(f"Alpaca Data Download Skipped (API Key Error): {ve}")
    except Exception as e:
        logger.error(f"Alpaca Data Download Failed: {e}")

    # --- Run with Polygon.io ---
    print("\n--- Running DataDownloader with Polygon.io ---")
    try:
        # PolygonProvider has internal rate limiting (time.sleep(12.1)) per symbol.
        # So, while use_parallel is set to True, the DataDownloader will detect PolygonProvider
        # and default to sequential processing in its `download_and_store_data` method for symbol iteration.
        polygon_downloader = DataDownloader(source='polygon', use_parallel=True, num_parallel_workers=1)

        # Russell 2000 components
        # russell2000_symbols = polygon_downloader.get_russell2000_components()
        # logger.info(f"Polygon: Downloading Russell 2000 data for {len(russell2000_symbols)} symbols.")
        # polygon_russell2000_df = polygon_downloader.download_and_store_data(russell2000_symbols, start_date, end_date)
        # logger.info(f"Polygon Russell 2000 download complete. Total rows: {len(polygon_russell2000_df)}")

        logger.info(f"Polygon: Downloading sample Russell 2000 data for {len(test_symbols_russell)} symbols.")
        polygon_sample_russell_df = polygon_downloader.download_and_store_data(test_symbols_russell, start_date,
                                                                               end_date)
        logger.info(f"Polygon sample Russell 2000 download complete. Total rows: {len(polygon_sample_russell_df)}")
        # print(polygon_sample_russell_df.head())

    except ValueError as ve:
        logger.error(f"Polygon Data Download Skipped (API Key Error): {ve}")
    except Exception as e:
        logger.error(f"Polygon Data Download Failed: {e}")

    print("\n--- All Data Download Processes Completed ---")


if __name__ == "__main__":
    main()