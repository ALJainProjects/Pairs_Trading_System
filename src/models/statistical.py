import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.add_constant import add_constant
from typing import Tuple, Optional
from config.logging_config import logger

class StatisticalModel:
    """
    Provides a suite of statistical methods for analyzing asset pairs, including
    cointegration, hedge ratio calculation, spread analysis, and half-life of mean reversion.
    """

    def __init__(self):
        """Initializes the StatisticalModel."""
        logger.debug("StatisticalModel initialized.")

    def cointegration_test(self, asset1: pd.Series, asset2: pd.Series) -> Tuple[bool, float]:
        """
        Performs the Engle-Granger cointegration test.

        Args:
            asset1 (pd.Series): Price series of the first asset.
            asset2 (pd.Series): Price series of the second asset.

        Returns:
            Tuple[bool, float]: A tuple containing a boolean indicating if the pair is
                                cointegrated at the 5% level, and the p-value of the test.
        """
        if asset1.isnull().any() or asset2.isnull().any():
            asset1, asset2 = asset1.dropna(), asset2.dropna()
            common_index = asset1.index.intersection(asset2.index)
            asset1, asset2 = asset1[common_index], asset2[common_index]

        if len(asset1) < 50 or len(asset2) < 50:
             return False, 1.0

        _, pvalue, _ = coint(asset1, asset2)
        return pvalue < 0.05, pvalue

    def calculate_rolling_hedge_ratio(self, asset1: pd.Series, asset2: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculates the hedge ratio using a rolling OLS regression.

        Args:
            asset1 (pd.Series): Price series of the dependent asset.
            asset2 (pd.Series): Price series of the independent asset.
            window (int): The rolling window size for the regression.

        Returns:
            pd.Series: A time series of the rolling hedge ratio (beta).
        """
        hedge_ratios = pd.Series(np.nan, index=asset1.index)
        for i in range(window, len(asset1)):
            window_asset1 = asset1.iloc[i-window:i]
            window_asset2 = asset2.iloc[i-window:i]
            model = OLS(window_asset1, add_constant(window_asset2)).fit()
            hedge_ratios.iloc[i] = model.params.iloc[1]
        
        return hedge_ratios.ffill()

    def calculate_spread(self, asset1: pd.Series, asset2: pd.Series, hedge_ratio: pd.Series) -> pd.Series:
        """
        Calculates the spread between two assets using a dynamic hedge ratio.
        """
        return asset1 - (hedge_ratio * asset2)

    def calculate_zscore(self, spread: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculates the z-score of the spread to identify trading opportunities.
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_score = (spread - rolling_mean) / (rolling_std + 1e-8)
        return z_score

    def calculate_half_life(self, spread: pd.Series) -> Optional[float]:
        """
        Calculates the half-life of mean reversion for a spread series.
        """
        if adfuller(spread.dropna())[1] > 0.05:
            return None

        lag_spread = spread.shift(1).dropna()
        delta_spread = spread.diff().dropna()
        
        common_index = lag_spread.index.intersection(delta_spread.index)
        lag_spread = lag_spread[common_index]
        delta_spread = delta_spread[common_index]

        model = OLS(delta_spread, add_constant(lag_spread)).fit()
        lambda_ = model.params.iloc[1]

        if lambda_ < 0:
            return -np.log(2) / lambda_
        return None