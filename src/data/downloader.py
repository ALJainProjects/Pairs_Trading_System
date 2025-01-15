import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Union
from tqdm import tqdm
from config.settings import RAW_DATA_DIR
from config.settings import DATA_DIR
from config.logging_config import logger
import requests

class DataDownloader:
    """Class for managing financial data downloads and processing."""

    DEFAULT_YEARS = 10

    def __init__(self,
                 chunk_size: int = 50,
                 rate_limit_pause: float = 1.0):
        """
        Initialize the downloader.

        Args:
            chunk_size: Number of symbols per chunk
            rate_limit_pause: Pause between chunks (seconds)
        """
        self.chunk_size = chunk_size
        self.rate_limit_pause = rate_limit_pause
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(RAW_DATA_DIR.replace(r'\config', ''), exist_ok=True)

    @staticmethod
    def calculate_dates(years_back: int = DEFAULT_YEARS) -> tuple[datetime, datetime]:
        """
        Calculate start and end dates from today.

        Args:
            years_back: Number of years to look back

        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=years_back)

        if years_back > 10:
            logger.warning(
                f"Date range exceeds 10 years ({years_back} years) - "
                "consider shorter periods for better reliability"
            )

        return start_date, end_date

    def get_nasdaq100_components(self) -> List[str]:
        """Get NASDAQ 100 components."""
        try:
            components_path = os.path.join(DATA_DIR.replace(r'\config', ''), 'nasdaq100_components.csv')
            if os.path.exists(components_path):
                components_df = pd.read_csv(components_path)
                if not components_df.empty:
                    return components_df['Symbol'].tolist()

            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
            res = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
            holdings = pd.DataFrame(res.json()['data']['data']['rows'])
            holdings = holdings.rename(columns={'symbol': 'Symbol'})

            if not holdings.empty:
                return holdings['Symbol'].tolist()

        except Exception as e:
            logger.error(f"Error getting NASDAQ 100 components: {str(e)}")
            raise ValueError("Could not retrieve valid NASDAQ 100 components")

    def download_historical_data(self,
                                 symbols: List[str],
                                 start_date: Union[str, datetime, pd.Timestamp] = None,
                                 end_date: Union[str, datetime, pd.Timestamp] = None,
                                 years_back: int = None) -> None:
        """
        Download historical data for individual symbols and save to separate CSV files.

        Args:
            symbols: List of ticker symbols to download
            start_date: Optional start date (string or datetime)
            end_date: Optional end date (string or datetime)
            years_back: Optional number of years to look back from end_date
        """
        if end_date is None:
            end_date = pd.Timestamp.now().normalize()
        else:
            end_date = pd.Timestamp(end_date).normalize()

        if start_date is not None:
            start_date = pd.Timestamp(start_date).normalize()
        elif years_back is not None:
            start_date = end_date - pd.DateOffset(years=years_back)
        else:
            start_date = end_date - pd.DateOffset(years=self.DEFAULT_YEARS)

        logger.info(f"Downloading data from {start_date.date()} to {end_date.date()}")

        failed_symbols = []

        for symbol in tqdm(symbols, desc="Downloading data"):
            try:
                self._download_symbol(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {str(e)}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed to download data for {len(failed_symbols)} symbols: {', '.join(failed_symbols)}")

    def _download_symbol(self,
                         symbol: str,
                         start_date: datetime,
                         end_date: datetime) -> None:
        """
        Download data for a single symbol and save it as {symbol}.csv.

        Args:
            symbol: Stock symbol to download
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)

        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return

        df.reset_index(inplace=True)
        df['Symbol'] = symbol

        columns_to_save = ['Date', 'Symbol', 'Adj Close', 'High', 'Low', 'Open', 'Close', 'Volume']
        if all(col in df.columns for col in columns_to_save):
            df = df[columns_to_save]
        else:
            logger.warning(f"Missing expected columns for {symbol}, saving raw data.")

        df = df.pivot(index='Date', columns='Symbol', values=['Adj Close', 'High', 'Low', 'Open', 'Close', 'Volume'])
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df.reset_index(inplace=True)

        df.rename(columns={
            f'Close_{symbol}': 'Close',
            f'Adj Close_{symbol}': 'Adj_Close',
            f'High_{symbol}': 'High',
            f'Low_{symbol}': 'Low',
            f'Open_{symbol}': 'Open',
            f'Volume_{symbol}': 'Volume'
        }, inplace=True)

        df['Symbol'] = symbol
        df = df[['Date', 'Symbol', 'Adj_Close', 'High', 'Low', 'Open', 'Close', 'Volume']]

        filename = f"{symbol}.csv"
        df.to_csv(os.path.join(RAW_DATA_DIR.replace(r'\config', ''), filename), index=False)


def main():
    """Main execution function."""
    try:
        downloader = DataDownloader(
            chunk_size=50,
            rate_limit_pause=1.0
        )

        logger.info("Getting NASDAQ 100 components...")
        symbols = downloader.get_nasdaq100_components()
        logger.info(f"Retrieved {len(symbols)} valid symbols")

        logger.info("Downloading historical data...")
        downloader.download_historical_data(
            symbols,
            years_back=4
        )

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()