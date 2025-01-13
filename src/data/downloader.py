"""
Enhanced Data Download Module

This module handles downloading and processing of financial data with:
1. Automatic date calculation from today
2. Enhanced parallel processing
3. Robust error handling
4. Progress tracking
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
import warnings
from functools import partial
import multiprocessing
from config.settings import RAW_DATA_DIR
from config.logging_config import logger
import time

class DataDownloader:
    """Class for managing financial data downloads and processing."""

    DEFAULT_YEARS = 10

    def __init__(self,
                 max_download_workers: int = None,
                 max_process_workers: int = None,
                 chunk_size: int = 50,
                 rate_limit_pause: float = 1.0):
        """
        Initialize the downloader.

        Args:
            max_download_workers: Maximum concurrent downloads (None for CPU count)
            max_process_workers: Maximum processing workers (None for CPU count)
            chunk_size: Number of symbols per chunk
            rate_limit_pause: Pause between chunks (seconds)
        """
        self.max_download_workers = (max_download_workers or
                                   multiprocessing.cpu_count())
        self.max_process_workers = (max_process_workers or
                                  multiprocessing.cpu_count())
        self.chunk_size = chunk_size
        self.rate_limit_pause = rate_limit_pause
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

    @staticmethod
    def calculate_dates(years_back: int = DEFAULT_YEARS) -> Tuple[datetime, datetime]:
        """
        Calculate start and end dates from today.

        Args:
            years_back: Number of years to look back

        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=years_back)

        # Warn if range is very long
        if years_back > 10:
            logger.warning(
                f"Date range exceeds 10 years ({years_back} years) - "
                "consider shorter periods for better reliability"
            )

        return start_date, end_date

    def get_nasdaq100_components(self) -> List[str]:
        """Get NASDAQ 100 components with parallelized validation."""
        try:
            # Try loading from CSV first
            components_path = os.path.join(RAW_DATA_DIR, 'nasdaq100_components.csv')
            if os.path.exists(components_path):
                components_df = pd.read_csv(components_path)
                if not components_df.empty:
                    # Validate symbols in parallel
                    valid_symbols = self._validate_symbols_parallel(
                        components_df['Symbol'].tolist()
                    )
                    if valid_symbols:
                        return valid_symbols

            # Fallback to yfinance
            nasdaq = yf.Ticker('^NDX')
            holdings = nasdaq.get_top_holdings()

            if holdings is not None and not holdings.empty:
                # Validate and save components
                valid_symbols = self._validate_symbols_parallel(
                    holdings['Symbol'].tolist()
                )

                if valid_symbols:
                    pd.DataFrame({'Symbol': valid_symbols}).to_csv(
                        components_path, index=False
                    )
                    return valid_symbols

            raise ValueError("Could not retrieve valid NASDAQ 100 components")

        except Exception as e:
            logger.error(f"Error getting NASDAQ 100 components: {str(e)}")
            raise

    def _validate_symbols_parallel(self,
                                 symbols: List[str]) -> List[str]:
        """Validate multiple symbols in parallel."""
        valid_symbols = []

        with ThreadPoolExecutor(max_workers=self.max_download_workers) as executor:
            futures = {
                executor.submit(self._validate_symbol, symbol): symbol
                for symbol in symbols
            }

            for future in tqdm(as_completed(futures),
                             total=len(symbols),
                             desc="Validating symbols"):
                symbol = futures[future]
                try:
                    if future.result():
                        valid_symbols.append(symbol)
                except Exception:
                    logger.warning(f"Failed to validate {symbol}")

        return valid_symbols

    @staticmethod
    def _validate_symbol(symbol: str) -> bool:
        """Validate single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return bool(info and info.get('regularMarketPrice'))
        except:
            return False

    def download_historical_data(self,
                               symbols: List[str],
                               years_back: int = DEFAULT_YEARS) -> pd.DataFrame:
        """
        Download historical data with automatic date calculation.

        Args:
            symbols: List of ticker symbols
            years_back: Number of years to look back

        Returns:
            DataFrame of price data
        """
        # Calculate dates
        start_dt, end_dt = self.calculate_dates(years_back)
        logger.info(f"Downloading data from {start_dt.date()} to {end_dt.date()}")

        # Process in chunks
        chunks = [symbols[i:i + self.chunk_size]
                 for i in range(0, len(symbols), self.chunk_size)]

        all_data = {}
        failed_symbols = []

        for chunk_idx, chunk in enumerate(chunks, 1):
            logger.info(
                f"Processing chunk {chunk_idx}/{len(chunks)} "
                f"({len(chunk)} symbols)"
            )

            with ThreadPoolExecutor(max_workers=self.max_download_workers) as executor:
                futures = {}

                # Submit downloads with progress bar
                with tqdm(total=len(chunk),
                         desc=f"Downloading chunk {chunk_idx}") as pbar:
                    for symbol in chunk:
                        futures[executor.submit(
                            self._download_symbol,
                            symbol, start_dt, end_dt
                        )] = symbol

                    # Process as they complete
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                all_data[symbol] = result
                            else:
                                failed_symbols.append(symbol)
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {str(e)}")
                            failed_symbols.append(symbol)
                        pbar.update(1)

            # Rate limiting pause between chunks
            if chunk_idx < len(chunks):
                time.sleep(self.rate_limit_pause)

        # Handle download results
        if not all_data:
            raise ValueError("No data was successfully downloaded")

        prices = pd.DataFrame(all_data)

        # Save combined file with date range in filename
        filename = (f"combined_prices_{start_dt.strftime('%Y%m%d')}_"
                   f"{end_dt.strftime('%Y%m%d')}.csv")
        prices.to_csv(os.path.join(RAW_DATA_DIR, filename))

        if failed_symbols:
            logger.warning(f"Failed to download {len(failed_symbols)} symbols: "
                         f"{', '.join(failed_symbols)}")

        return prices

    def _download_symbol(self,
                        symbol: str,
                        start_date: datetime,
                        end_date: datetime,
                        max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Download data for a single symbol with retries."""
        retries = 0

        while retries < max_retries:
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=False
                )

                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    return None

                # Save individual file with date range
                filename = (f"{symbol}_{start_date.strftime('%Y%m%d')}_"
                          f"{end_date.strftime('%Y%m%d')}.csv")
                df.to_csv(os.path.join(RAW_DATA_DIR, filename))
                return df['Close']

            except Exception as e:
                retries += 1
                logger.warning(f"Attempt {retries} failed for {symbol}: {str(e)}")
                if retries < max_retries:
                    time.sleep(retries)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {symbol} after {max_retries} attempts")
                    return None

    def process_data(self,
                    prices: pd.DataFrame,
                    min_history: float = 0.95,
                    parallel: bool = True) -> pd.DataFrame:
        """
        Process price data with parallel computation option.

        Args:
            prices: DataFrame of price data
            min_history: Minimum fraction of history required
            parallel: Whether to use parallel processing

        Returns:
            DataFrame of cleaned returns
        """
        try:
            # Check data quality
            total_days = len(prices)
            min_days = int(total_days * min_history)

            # Remove columns with insufficient history
            valid_counts = prices.count()
            prices = prices.loc[:, valid_counts >= min_days]

            if prices.empty:
                raise ValueError("No symbols with sufficient history")

            # Process in parallel if requested
            if parallel and len(prices.columns) > 100:
                returns = self._parallel_process_returns(prices)
            else:
                returns = self._process_returns(prices)

            # Validate final output
            if returns.empty:
                raise ValueError("No valid returns after processing")

            removed_symbols = set(prices.columns) - set(returns.columns)
            if removed_symbols:
                logger.warning(f"Removed {len(removed_symbols)} symbols during processing")

            return returns

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

    def _parallel_process_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Process returns calculation in parallel."""
        # Split columns into chunks
        chunks = np.array_split(prices.columns, self.max_process_workers)

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(
                    executor.submit(self._process_returns, prices[chunk])
                )

            # Combine results
            results = [future.result() for future in futures]

        return pd.concat(results, axis=1)

    @staticmethod
    def _process_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Process single chunk of returns."""
        # Forward fill limited gaps
        prices = prices.fillna(method='ffill', limit=5)

        # Remove problematic data
        prices = prices.replace([np.inf, -np.inf], np.nan)
        prices = prices.dropna(axis=1)

        # Calculate returns
        returns = prices.pct_change()

        # Clean returns
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(method='ffill', limit=5)
        returns = returns.dropna()

        return returns

def main():
    """Main execution function."""
    try:
        # Initialize downloader with parallel processing
        downloader = DataDownloader(
            max_download_workers=multiprocessing.cpu_count(),
            max_process_workers=multiprocessing.cpu_count(),
            chunk_size=50,
            rate_limit_pause=1.0
        )

        # Get and validate components
        logger.info("Getting NASDAQ 100 components...")
        symbols = downloader.get_nasdaq100_components()
        logger.info(f"Retrieved {len(symbols)} valid symbols")

        # Download 10 years of data from today
        logger.info("Downloading historical data...")
        prices = downloader.download_historical_data(
            symbols,
            years_back=10
        )
        logger.info(f"Downloaded data for {len(prices.columns)} symbols")

        # Process data with parallel computation
        logger.info("Processing returns...")
        returns = downloader.process_data(
            prices,
            min_history=0.95,
            parallel=True
        )
        logger.info(f"Processed returns for {len(returns.columns)} symbols")

        return returns

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()