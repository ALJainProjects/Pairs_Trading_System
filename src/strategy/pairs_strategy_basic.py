"""
Basic Pairs Trading Strategy Module

This module contains a basic implementation of pairs trading strategy.
"""

from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS

# Import analysis modules
from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.analysis.cointegration import determine_cointegration, calculate_half_life
from src.analysis.clustering_analysis import AssetClusteringAnalyzer
from src.analysis.denoiser_usage import AssetAnalyzer

# Configuration and logging
from config.logging_config import logger

class PairsTrader:
    """Basic pairs trading strategy implementation."""

    def __init__(self,
                correlation_threshold: float = 0.8,
                lookback_period: int = 20,
                entry_threshold: float = 1.5,
                exit_threshold: float = 0.5):
        """
        Initialize the pairs trading strategy.

        Args:
            correlation_threshold: Minimum correlation coefficient to consider a pair
            lookback_period: Period for calculating statistics
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
        """
        self.correlation_threshold = correlation_threshold
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        # Initialize analyzers
        self.correlation_analyzer = None
        self.asset_analyzer = AssetAnalyzer()

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Find pairs suitable for trading based on correlation and cointegration.

        Args:
            prices: DataFrame of asset prices

        Returns:
            List of pairs (tuples of asset names)
        """
        # Calculate returns
        returns = prices.pct_change().dropna()

        # Initialize correlation analyzer
        self.correlation_analyzer = CorrelationAnalyzer(returns)

        # Get highly correlated pairs
        corr_pairs = self.correlation_analyzer.get_highly_correlated_pairs(
            correlation_type='pearson',
            threshold=self.correlation_threshold
        )

        # Filter for cointegrated pairs
        cointegrated_pairs = []
        for _, row in corr_pairs.iterrows():
            asset1, asset2 = row['asset1'], row['asset2']
            is_coint, _, _, _ = determine_cointegration(
                prices[asset1],
                prices[asset2]
            )
            if is_coint:
                cointegrated_pairs.append((asset1, asset2))

        return cointegrated_pairs

    def calculate_spread(self, prices: pd.DataFrame, pair: Tuple[str, str]) -> pd.Series:
        """
        Calculate the spread between two assets.

        Args:
            prices: DataFrame of asset prices
            pair: Tuple of asset names

        Returns:
            Series containing the spread
        """
        asset1, asset2 = pair
        # Calculate hedge ratio using OLS
        model = OLS(prices[asset1], prices[asset2]).fit()
        hedge_ratio = model.params[0]

        # Calculate spread
        spread = prices[asset1] - hedge_ratio * prices[asset2]
        return spread

    def generate_signals(self, prices: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Generate trading signals for pairs.

        Args:
            prices: DataFrame of asset prices
            pairs: List of pairs to generate signals for

        Returns:
            DataFrame containing signals for each pair
        """
        signals = pd.DataFrame(index=prices.index)

        for pair in pairs:
            asset1, asset2 = pair
            spread = self.calculate_spread(prices, pair)

            # Calculate z-score
            rolling_mean = spread.rolling(window=self.lookback_period).mean()
            rolling_std = spread.rolling(window=self.lookback_period).std()
            zscore = (spread - rolling_mean) / rolling_std

            # Generate signals
            signals[f"{asset1}_{asset2}_signal"] = 0

            # Long signals
            signals.loc[zscore < -self.entry_threshold, f"{asset1}_{asset2}_signal"] = 1
            signals.loc[zscore > -self.exit_threshold, f"{asset1}_{asset2}_signal"] = 0

            # Short signals
            signals.loc[zscore > self.entry_threshold, f"{asset1}_{asset2}_signal"] = -1
            signals.loc[zscore < self.exit_threshold, f"{asset1}_{asset2}_signal"] = 0

        return signals