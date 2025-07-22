import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Union
from abc import ABC, abstractmethod
from tqdm import tqdm
import yfinance as yf
import requests
from bs4 import BeautifulSoup  # Import BeautifulSoup for web scraping

# Alpaca and Polygon imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from polygon import RESTClient

# Local project imports
from config.settings import RAW_DATA_DIR, DATA_DIR, ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY
from config.logging_config import logger
from src.data.database import HistoricalDatabaseManager


# --- Abstract Base Class for Data Providers ---
class DataProvider(ABC):
    """Abstract interface for all data source providers."""

    @abstractmethod
    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Downloads data and returns it in a standardized long-format DataFrame."""
        pass

    def _standardize_long_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ensure a common format for the final DataFrame."""
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'adj_close': 'Adj_Close', 'timestamp': 'Date', 'symbol': 'Symbol'
        }, inplace=True)

        if 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']  # Fallback if Adj_Close is not provided

        df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert to date object for consistency

        standard_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        return df.reindex(columns=standard_cols)


# --- Concrete Provider Implementations ---
class YFinanceProvider(DataProvider):
    """Data provider for Yahoo Finance."""

    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        df = yf.download(symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if df.empty: return pd.DataFrame()

        # Handle single symbol vs multiple symbols output from yfinance
        if len(symbols) == 1:
            df = df.reset_index()
            df['Symbol'] = symbols[0]  # Add symbol column for single ticker
        else:
            # If MultiIndex from yfinance, ensure correct column names and stacking
            # Filter out columns that are not directly price/volume data (e.g., 'Unnamed: 0' sometimes appears)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten the MultiIndex for easier processing. Keep only the last level for price/volume
                # If first level is redundant, map to second. If not, pick main columns.
                # Example: ('Adj Close', 'AAPL') -> 'Adj Close'
                # This assumes major columns like 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
                df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
                # Then rename, e.g., 'Adj Close_AAPL' to 'Adj Close' and stack
                df.columns = df.columns.map(lambda x: x.split('_')[0] if len(x.split('_')) > 1 else x)

                df = df.stack(level=1).rename_axis(['Date', 'Symbol']).reset_index()
            else:
                # If it's not a MultiIndex for multiple symbols, it means YF provided a flat DF,
                # but with columns like 'AAPL Close', 'MSFT Close'. Need to unpivot.
                # This is a less common case for multiple symbols, YF usually gives MultiIndex.
                # If it happens, we'd need to melt or use regex to extract symbol.
                # For now, assume consistent YF multi-symbol output.
                pass  # The original logic for `len(symbols) == 1` or MultiIndex handles most cases.

        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
        return self._standardize_long_df(df)


class AlpacaProvider(DataProvider):
    """Data provider for Alpaca."""

    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("Alpaca API keys are not configured.")
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        bars = self.client.get_stock_bars(request_params)
        return self._standardize_long_df(bars.df.reset_index())


class PolygonProvider(DataProvider):
    """Data provider for Polygon.io, handles rate limiting."""

    def __init__(self):
        if not POLYGON_API_KEY:
            raise ValueError("Polygon API key is not configured.")
        self.client = RESTClient(POLYGON_API_KEY)

    def download_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        all_bars = []
        for symbol in tqdm(symbols, desc="Downloading from Polygon.io"):
            try:
                from_ = start_date.strftime('%Y-%m-%d')
                to_ = end_date.strftime('%Y-%m-%d')

                resp = self.client.get_aggs(symbol, 1, "day", from_, to_, limit=50000)
                if not resp or not resp.results:  # Check resp.results for actual data
                    logger.warning(f"No data returned for {symbol} from Polygon.")
                    continue

                df = pd.DataFrame(resp.results)  # Access results attribute
                df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'},
                          inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                standardized_df = self._standardize_long_df(df.reset_index())
                standardized_df['Symbol'] = symbol  # Assign symbol here
                all_bars.append(standardized_df)

                time.sleep(12.1)  # Respect free tier limit of 5 requests/minute
            except Exception as e:
                logger.error(f"Failed to download {symbol} from Polygon: {e}")

        return pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()


# --- Main DataDownloader Facade ---
class DataDownloader:
    """Class for managing financial data downloads and processing."""

    def __init__(self, source: str = 'yfinance'):
        self.provider = self._get_provider(source)
        self.db_manager = HistoricalDatabaseManager()
        self._setup_directories()

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
        # DATA_DIR should already be created by settings.py, but ensure component dir exists
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
        # Try loading from the provided CSV file first
        local_components = self._load_components_from_csv('nasdaq-100-index-07-21-2025.csv')  #
        if local_components:
            return local_components

        components_path = os.path.join(DATA_DIR, 'nasdaq100_components.csv')
        # Fallback to general local CSV (if previously downloaded)
        local_components = self._load_components_from_csv('nasdaq100_components.csv')  #
        if local_components:
            return local_components

        # Fallback to NASDAQ API if no local CSV found
        logger.info("Attempting to download NASDAQ 100 components from API...")
        try:
            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
            res = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
            res.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            data = res.json()

            if not data or 'data' not in data or 'data' not in data['data'] or 'rows' not in data['data']['data']:
                raise ValueError("Unexpected JSON structure from NASDAQ API.")

            holdings = pd.DataFrame(data['data']['data']['rows'])
            holdings = holdings.rename(columns={'symbol': 'Symbol'})

            if not holdings.empty and 'Symbol' in holdings.columns:
                # Save to CSV for future use
                holdings['Symbol'].to_csv(components_path, index=False, header=True)
                logger.info(f"Downloaded and saved NASDAQ 100 components to {components_path}")
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
        """
        Fetches S&P 500 components. Prioritizes local CSV provided by user.
        """
        # Try loading from the provided CSV file first
        local_components = self._load_components_from_csv('sp-500-index-07-21-2025.csv')  #
        if local_components:
            return local_components

        components_path = os.path.join(DATA_DIR, 'sp500_components.csv')
        # Fallback to general local CSV (if previously downloaded)
        local_components = self._load_components_from_csv('sp500_components.csv')  #
        if local_components:
            return local_components

        # Fallback to Wikipedia if no local CSV found
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

            symbols = [s.replace('.', '-') for s in symbols]  # Handle specific cases like BRK.B -> BRK-B

            pd.DataFrame({'Symbol': symbols}).to_csv(components_path, index=False, header=True)
            logger.info(f"Downloaded and saved S&P 500 components to {components_path}")
            return symbols

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching S&P 500 components from Wikipedia: {e}")
            raise ValueError(f"Could not retrieve S&P 500 components due to network error: {e}")
        except Exception as e:
            logger.error(f"Error parsing S&P 500 components from Wikipedia: {e}")
            raise ValueError(f"Could not retrieve valid S&P 500 components: {e}")

    def get_russell2000_components(self) -> List[str]:
        """
        Fetches Russell 2000 components. Prioritizes local CSV provided by user.
        """
        # Try loading from the provided CSV file first
        local_components = self._load_components_from_csv('russell-2000-index-07-21-2025.csv')  #
        if local_components:
            return local_components

        components_path = os.path.join(DATA_DIR, 'russell2000_components.csv')
        # Fallback to general local CSV (if previously downloaded)
        local_components = self._load_components_from_csv('russell2000_components.csv')  #
        if local_components:
            return local_components

        # Fallback to scraping Slickcharts if no local CSV found
        logger.info("Attempting to download Russell 2000 components from Slickcharts (web scraping).")
        try:
            url = 'https://www.slickcharts.com/russell2000'
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            df_list = pd.read_html(response.text, match='Symbol')
            if not df_list:
                raise ValueError("Could not find table matching 'Symbol' on Slickcharts Russell 2000 page.")

            df = df_list[0]

            if 'Symbol' not in df.columns or df.empty:
                raise ValueError("Russell 2000 data from Slickcharts is empty or 'Symbol' column not found.")

            symbols = df['Symbol'].tolist()

            pd.DataFrame({'Symbol': symbols}).to_csv(components_path, index=False, header=True)
            logger.info(f"Downloaded and saved Russell 2000 components from Slickcharts to {components_path}")
            return symbols

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching Russell 2000 components from Slickcharts: {e}")
            raise ValueError(f"Could not retrieve Russell 2000 components due to network error: {e}")
        except Exception as e:
            logger.error(f"Error parsing Russell 2000 components from Slickcharts: {e}")
            raise ValueError(f"Could not retrieve valid Russell 2000 components: {e}")

    def download_and_store_data(self,
                                symbols: List[str],
                                start_date: Union[str, datetime],
                                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Downloads historical data from the selected provider and stores/updates it in the historical database.
        Uses the upsert_historical_data method for efficient and robust storage.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        logger.info(f"Starting data download using {self.provider.__class__.__name__}.")
        df = self.provider.download_data(symbols, start_dt, end_dt)

        if not df.empty:
            logger.info(f"Downloaded {len(df)} rows. Performing upsert to local historical database...")
            df_copy = df.copy()  # Avoid SettingWithCopyWarning
            # Ensure 'Adj_Close' is present before upserting if it was missed by provider
            if 'Adj_Close' not in df_copy.columns:
                df_copy['Adj_Close'] = df_copy['Close']
            self.db_manager.upsert_historical_data(df_copy)
            logger.info("Historical database upsert complete.")
        else:
            logger.warning("No data downloaded to store.")

        return df