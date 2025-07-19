import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.add_constant import add_constant
from typing import List, Tuple, Dict, Optional
from config.logging_config import logger
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def check_integration_order(series: pd.Series, p_threshold: float = 0.05) -> Optional[int]:
    """
    Checks the order of integration of a time series, up to I(2).
    A series is I(1) if it's not stationary but its first difference is.

    Args:
        series (pd.Series): The time series to check.
        p_threshold (float): The p-value threshold for the ADF test.

    Returns:
        Optional[int]: The order of integration (0, 1, or 2), or None if inconclusive.
    """
    # Test for I(0)
    p_value_level = adfuller(series.dropna())[1]
    if p_value_level < p_threshold:
        return 0
    
    # Test for I(1)
    p_value_diff1 = adfuller(series.diff().dropna())[1]
    if p_value_diff1 < p_threshold:
        return 1

    # Test for I(2)
    p_value_diff2 = adfuller(series.diff().diff().dropna())[1]
    if p_value_diff2 < p_threshold:
        return 2

    return None

def calculate_half_life(spread: pd.Series) -> float:
    """Calculates the half-life of mean reversion for a spread series."""
    lag_spread = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()
    
    common_index = lag_spread.index.intersection(delta_spread.index)
    lag_spread = lag_spread[common_index]
    delta_spread = delta_spread[common_index]
    
    model = OLS(delta_spread, add_constant(lag_spread)).fit()
    lambda_ = model.params.iloc[1]
    
    half_life = -np.log(2) / lambda_ if lambda_ < 0 else np.inf
    return half_life

def find_cointegrated_pairs(
    prices: pd.DataFrame,
    p_threshold: float = 0.05,
    min_half_life: int = 5,
    max_half_life: int = 252
) -> List[Dict]:
    """
    Finds cointegrated pairs within a DataFrame of prices.
    """
    n = prices.shape[1]
    symbols = prices.columns
    cointegrated_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            s1_name, s2_name = symbols[i], symbols[j]
            s1 = prices[s1_name].dropna()
            s2 = prices[s2_name].dropna()

            if check_integration_order(s1) != 1 or check_integration_order(s2) != 1:
                continue

            score, p_value, _ = coint(s1, s2)
            if p_value < p_threshold:
                model = OLS(s1, add_constant(s2)).fit()
                hedge_ratio = model.params[1]
                spread = s1 - hedge_ratio * s2
                
                half_life = calculate_half_life(spread)
                
                if min_half_life <= half_life <= max_half_life:
                    cointegrated_pairs.append({
                        'asset1': s1_name,
                        'asset2': s2_name,
                        'p_value': p_value,
                        'hedge_ratio': hedge_ratio,
                        'half_life': half_life
                    })
    
    return sorted(cointegrated_pairs, key=lambda x: x['p_value'])

def rolling_cointegration_test(s1: pd.Series, s2: pd.Series, window: int) -> pd.DataFrame:
    """
    Performs a rolling cointegration test to assess the stability of the relationship.
    """
    results = []
    for i in range(window, len(s1)):
        window_s1 = s1.iloc[i-window:i]
        window_s2 = s2.iloc[i-window:i]
        
        _, p_value, _ = coint(window_s1, window_s2)
        model = OLS(window_s1, add_constant(window_s2)).fit()
        hedge_ratio = model.params[1]
        
        results.append({
            'date': s1.index[i],
            'p_value': p_value,
            'hedge_ratio': hedge_ratio
        })
        
    return pd.DataFrame(results).set_index('date')

def find_cointegrating_vectors(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    """
    Uses the Johansen test to find cointegrating vectors in a multivariate system.
    """
    if prices.shape[1] < 3:
        raise ValueError("Johansen test requires at least 3 time series.")

    result = coint_johansen(prices, det_order, k_ar_diff)
    
    num_relationships = np.sum(result.lr1 > result.cvt[:, 1])
    
    if num_relationships == 0:
        logger.info("No cointegrating relationships found by Johansen test.")
        return pd.DataFrame()
    
    logger.info(f"Found {num_relationships} cointegrating relationship(s).")
    return pd.DataFrame(result.evec[:, :num_relationships], index=prices.columns)