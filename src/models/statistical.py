"""
Statistical Models Module

Enhancements:
 1. More robust cointegration testing using Engle-Granger and optional Johansen.
 2. Spread calculation with optional ratio-based spread.
 3. Extended docstrings for usage in a pair trading workflow.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from config.logging_config import logger

try:
    # Optional: If you want Johansen test
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    HAS_JOHANSEN = True
except ImportError:
    HAS_JOHANSEN = False

class StatisticalModel:
    """
    The StatisticalModel provides core statistical methods relevant to pair selection and
    signal generation in pair trading. Methods included:
      - Engle-Granger cointegration test
      - Optional Johansen cointegration test (if installed)
      - Calculating a spread between two assets via regression
      - Mean reversion signal generation
    """

    def __init__(self):
        logger.info("Initializing StatisticalModel.")

    def cointegration_test(self, asset1: pd.Series, asset2: pd.Series, significance: float = 0.05) -> bool:
        """
        Perform the Engle-Granger cointegration test between two price series.

        Args:
            asset1 (pd.Series): Price series of the first asset.
            asset2 (pd.Series): Price series of the second asset.
            significance (float): Significance threshold for cointegration (default=0.05).

        Returns:
            bool: True if cointegrated, False otherwise.
        """
        logger.info("Performing Engle-Granger cointegration test.")
        # Basic checks
        if len(asset1) != len(asset2):
            raise ValueError("Series must have the same length for Engle-Granger test.")
        if asset1.isnull().any() or asset2.isnull().any():
            raise ValueError("Series contain NaN values. Clean your data first.")

        score, pvalue, _ = coint(asset1, asset2)
        logger.debug(f"EG cointegration p-value: {pvalue:.4f}")
        return pvalue < significance

    def johansen_test(self, df_prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> bool:
        """
        (Optional) Perform Johansen cointegration test on multiple price series.

        Args:
            df_prices (pd.DataFrame): Price DataFrame with columns as different assets.
            det_order (int): Deterministic trend order (0=none, 1=constant, etc.).
            k_ar_diff (int): Lagged differences used in test.

        Returns:
            bool: True if test indicates at least one cointegration relationship, else False.
        """
        if not HAS_JOHANSEN:
            logger.warning("Johansen test not available; install statsmodels>=0.10+ with vector_ar.vecm.")
            return False

        logger.info("Performing Johansen cointegration test on multiple series.")
        result = coint_johansen(df_prices, det_order, k_ar_diff)

        # Basic check: if the trace statistic suggests cointegration
        # Compare trace stats to critical values at 5% level
        trace_stat = result.lr1
        crit_vals = result.cvt[:, 1]  # 1 => 5% level
        coint_count = sum(trace_stat > crit_vals)
        logger.debug(f"Johansen coint relationships found: {coint_count}")
        return coint_count > 0

    def calculate_spread(self, asset1: pd.Series, asset2: pd.Series, use_ratio: bool = False) -> pd.Series:
        """
        Calculate the spread between two assets using OLS or a ratio approach.

        Args:
            asset1 (pd.Series): First asset's price series.
            asset2 (pd.Series): Second asset's price series.
            use_ratio (bool): If True, use ratio-based spread (asset1 / asset2). Else OLS regression.

        Returns:
            pd.Series: Spread series over time.
        """
        logger.info(f"Calculating spread (use_ratio={use_ratio}).")
        if use_ratio:
            # Ratio approach
            spread = asset1 / asset2
        else:
            # OLS approach
            X = add_constant(asset2)
            model = OLS(asset1, X).fit()
            alpha = model.params["const"]
            beta = model.params[asset2.name]
            spread = asset1 - (beta * asset2 + alpha)
        logger.debug("Spread calculation completed.")
        return spread

    def mean_reversion_signal(self, spread: pd.Series, window: int = 20, z_threshold: float = 2.0) -> pd.Series:
        """
        Generate mean reversion trading signals based on z-score of the spread.

        Args:
            spread (pd.Series): Spread series (e.g., from calculate_spread).
            window (int): Rolling window for mean/std.
            z_threshold (float): +/- threshold to trigger signals.

        Returns:
            pd.Series: A series of signals [1, -1, 0].
        """
        logger.info("Generating mean reversion signals from spread.")
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()
        zscore = (spread - roll_mean) / (roll_std + 1e-12)

        signals = pd.Series(index=spread.index, data=0, dtype=float)
        signals[zscore > z_threshold] = -1.0   # short spread
        signals[zscore < -z_threshold] = 1.0   # long spread
        return signals
