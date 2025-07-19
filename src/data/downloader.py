import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Union
from abc import ABC, abstractmethod
from tqdm import tqdm
import yfinance as yf
import requests

# Alpaca and Polygon imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from polygon.rest import RESTClient

# Local project imports
from config.settings import RAW_DATA_DIR, DATA_DIR, ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY
from config.logging_config import logger
from src.data.database import HistoricalDatabaseManager # Changed import to HistoricalDatabaseManager


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
            df['Adj_Close'] = df['Close'] # Fallback if Adj_Close is not provided

        df['Date'] = pd.to_datetime(df['Date']).dt.date # Convert to date object for consistency

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
            df['Symbol'] = symbols[0] # Add symbol column for single ticker
        else:
            df = df.stack(level=1).rename_axis(['Date', 'Symbol']).reset_index()

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
                if not resp or not resp.results: # Check resp.results for actual data
                    logger.warning(f"No data returned for {symbol} from Polygon.")
                    continue

                df = pd.DataFrame(resp.results) # Access results attribute
                df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'},
                          inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                # Polygon data usually doesn't have an explicit 'symbol' column in the individual bars,
                # so we add it after standardization.
                standardized_df = self._standardize_long_df(df.reset_index())
                standardized_df['Symbol'] = symbol # Assign symbol here
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
        # Changed to HistoricalDatabaseManager
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

    def get_nasdaq100_components(self) -> List[str]:
        """Get NASDAQ 100 components."""
        try:
            # Adjust path for cross-platform compatibility and correct DATA_DIR usage
            components_path = os.path.join(DATA_DIR, 'nasdaq100_components.csv')

            if os.path.exists(components_path):
                components_df = pd.read_csv(components_path)
                if not components_df.empty and 'Symbol' in components_df.columns:
                    logger.info(f"Loaded NASDAQ 100 components from {components_path}")
                    return components_df['Symbol'].tolist()

            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
            res = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
            res.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
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
            # The df should already be in the standardized format ('Date', 'Symbol', etc.)
            # as returned by _standardize_long_df in the DataProvider.
            self.db_manager.upsert_historical_data(df)
            logger.info("Historical database upsert complete.")
        else:
            logger.warning("No data downloaded to store.")

        return df