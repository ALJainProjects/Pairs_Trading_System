import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
# Corrected import for add_constant
from statsmodels.tools.tools import add_constant
from typing import List, Tuple, Dict, Optional, Union
from config.logging_config import logger  # Assuming this is configured
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # For Johansen test


def _align_series_and_drop_na(*series: pd.Series) -> List[pd.Series]:
    """
    Helper function to align multiple pandas Series based on their common index,
    sort them, and drop rows where any of the series has a NaN value within the common index.

    This function is extracted from the `StatisticalModel` class for reusability
    in these standalone functions.

    Args:
        *series: Variable number of pandas Series to align.

    Returns:
        List[pd.Series]: A list of aligned and cleaned Series. Returns empty Series
                         of original types if no common non-NaN data.
    """
    if not series:
        logger.debug("No series provided for alignment.")
        return []

    # Concatenate all series into a single DataFrame for efficient alignment and NaN dropping
    # Use outer join to keep all possible indices initially
    df_concat = pd.concat(series, axis=1, join='outer')

    # Drop rows where any column (original series) has a NaN
    df_aligned = df_concat.dropna()

    if df_aligned.empty:
        logger.warning("All data points dropped after aligning and removing NaNs. No common non-NaN data found.")
        return [pd.Series(dtype=s.dtype, name=s.name) for s in series]  # Return empty series of original types

    # Ensure index is sorted for time series operations
    df_aligned = df_aligned.sort_index()

    result_series = [df_aligned.iloc[:, i].rename(series[i].name) for i in range(len(series))]

    # This check should ideally not trigger if df_aligned is not empty, but as a safeguard.
    if any(s.empty for s in result_series):
        logger.warning("After alignment and final NaN cleaning, at least one resulting series is empty. "
                       "This indicates an issue with the common index or data sparsity.")

    logger.debug(f"Aligned {len(series)} series. Original lengths: {[len(s) for s in series]}. "
                 f"Aligned length: {len(result_series[0]) if result_series else 0}.")
    return result_series


def check_integration_order(
        series: pd.Series,
        p_threshold: float = 0.05,
        min_observations: int = 20,  # Minimum observations required for ADF test
        regression: str = 'c'  # 'c' for constant, 'ct' for constant and trend, 'nc' for no constant/no trend
) -> Optional[int]:
    """
    Checks the order of integration of a time series, up to I(2).
    A series is I(d) if its d-th difference is stationary, but its (d-1)-th difference is not.

    Args:
        series (pd.Series): The time series to check.
        p_threshold (float): The p-value threshold for the ADF test.
        min_observations (int): Minimum number of non-null observations required for ADF test.
        regression (str): The regression type for the ADF test ('c', 'ct', 'nc').
                          'c': Constant (default for most economic series).
                          'ct': Constant and trend.
                          'nc': No constant and no trend.

    Returns:
        Optional[int]: The order of integration (0, 1, or 2), or None if inconclusive or not enough data.
    """
    # Drop NaNs once at the beginning to handle gaps
    series_clean = series.dropna()

    if len(series_clean) < min_observations:
        logger.warning(
            f"Series '{series.name}' too short ({len(series_clean)} obs) for ADF test at level. Min required: {min_observations}. Returning None.")
        return None
    if series_clean.nunique() <= 1:
        logger.warning(
            f"Series '{series.name}' is constant or has too few unique values for ADF test. Returning None.")
        return None

    # Test for I(0) (stationarity at level)
    try:
        p_value_level = adfuller(series_clean, regression=regression)[1]
        logger.debug(f"ADF test for '{series.name}' at level: p-value={p_value_level:.4f} (threshold={p_threshold})")
        if p_value_level < p_threshold:
            return 0
    except Exception as e:
        logger.error(f"ADF test failed for series '{series.name}' at level: {e}. Returning None.")
        return None

    # Test for I(1) (first difference is stationary)
    diff1_series = series_clean.diff().dropna()
    if len(diff1_series) < min_observations:
        logger.warning(
            f"Series '{series.name}' too short ({len(diff1_series)} obs) for ADF test after 1st difference. Min required: {min_observations}. Returning None.")
        return None
    if diff1_series.nunique() <= 1:
        logger.warning(
            f"First difference of series '{series.name}' is constant or has too few unique values for ADF test. Returning 1.")
        # If 1st diff is constant, its diff2 would be 0, adfuller might fail/give misleading result.
        # If it's constant, it's stationary, so it's I(1).
        return 1

    try:
        p_value_diff1 = adfuller(diff1_series, regression=regression)[1]
        logger.debug(
            f"ADF test for '{series.name}' after 1st diff: p-value={p_value_diff1:.4f} (threshold={p_threshold})")
        if p_value_diff1 < p_threshold:
            return 1
    except Exception as e:
        logger.error(f"ADF test failed for 1st differenced series '{series.name}': {e}. Returning None.")
        return None

    # Test for I(2) (second difference is stationary)
    diff2_series = diff1_series.diff().dropna()
    if len(diff2_series) < min_observations:
        logger.warning(
            f"Series '{series.name}' too short ({len(diff2_series)} obs) for ADF test after 2nd difference. Min required: {min_observations}. Returning None.")
        return None
    if diff2_series.nunique() <= 1:
        logger.warning(
            f"Second difference of series '{series.name}' is constant or has too few unique values for ADF test. Returning 2.")
        return 2

    try:
        p_value_diff2 = adfuller(diff2_series, regression=regression)[1]
        logger.debug(
            f"ADF test for '{series.name}' after 2nd diff: p-value={p_value_diff2:.4f} (threshold={p_threshold})")
        if p_value_diff2 < p_threshold:
            return 2
    except Exception as e:
        logger.error(f"ADF test failed for 2nd differenced series '{series.name}': {e}. Returning None.")
        return None

    logger.info(
        f"Integration order inconclusive for series '{series.name}'. P-values: Level={p_value_level:.4f}, Diff1={p_value_diff1:.4f}, Diff2={p_value_diff2:.4f}. Returning None.")
    return None


def calculate_half_life(spread: pd.Series) -> Optional[float]:
    """
    Calculates the half-life of mean reversion for a spread series using OLS regression.
    Half-life is meaningful only if the spread exhibits mean-reverting behavior (lambda < 0).

    Args:
        spread (pd.Series): The spread series.

    Returns:
        Optional[float]: The calculated half-life in periods, or None if the spread
                         does not exhibit mean reversion in the expected direction,
                         or if calculation fails due to insufficient data/variance.
                         Returns `np.inf` if lambda is non-negative, representing infinite half-life.
    """
    # Ensure no NaNs, and that there are enough observations
    spread_clean = spread.dropna()
    if len(spread_clean) < 10:  # Minimum observations for OLS after differencing/lagging
        logger.warning(
            f"Spread series '{spread.name}' too short ({len(spread_clean)} obs) to calculate half-life. Need at least 10. Returning None.")
        return None

    lag_spread = spread_clean.shift(1).dropna()
    delta_spread = spread_clean.diff().dropna()

    # Align the differenced and lagged series to ensure they have the same index
    aligned_lag_spread, aligned_delta_spread = _align_series_and_drop_na(lag_spread, delta_spread)

    if len(aligned_lag_spread) < 2:  # Need at least two points for OLS
        logger.warning(
            f"Not enough overlapping data points ({len(aligned_lag_spread)} obs) to calculate half-life for '{spread.name}'. Returning None.")
        return None

    # Add a check for constant lagged spread, which can cause singular matrix in OLS
    if aligned_lag_spread.nunique() <= 1:
        logger.warning(
            f"Lagged spread '{spread.name}' is constant, cannot calculate half-life (OLS will fail). Returning None.")
        return None

    try:
        # Regress delta_spread on lag_spread: delta_spread = lambda_ * lag_spread + c + epsilon
        model = OLS(aligned_delta_spread, add_constant(aligned_lag_spread)).fit()

        # Extract lambda_ (coefficient of the lagged spread)
        # Check if the parameter for the lagged spread exists (e.g., named 'x1' or by series name)
        # statsmodels' add_constant typically names the added constant 'const' and the variable 'x1' or its name
        lambda_ = model.params.get(aligned_lag_spread.name, None)
        if lambda_ is None and len(model.params) > 1:  # Fallback if name is not found, assume it's the second param
            lambda_ = model.params.iloc[1]

        if lambda_ is None:
            logger.warning(
                f"Could not extract lambda_ coefficient for '{spread.name}'. OLS parameters: {model.params.index.tolist()}. Returning None.")
            return None

        # Half-life = -ln(2) / lambda
        # lambda_ must be negative for mean reversion (i.e., when spread is high, it tends to decrease)
        if lambda_ < 0:
            half_life = -np.log(2) / lambda_
            logger.debug(f"Half-life for '{spread.name}' calculated: {half_life:.2f} periods.")
            return half_life
        else:
            logger.info(
                f"Lambda ({lambda_:.4f}) for '{spread.name}' is non-negative (>= 0), indicating no mean reversion or divergence. Half-life is infinite.")
            return np.inf  # Use float('inf') for consistency
    except ValueError as ve:
        logger.error(
            f"ValueError during OLS regression for half-life of '{spread.name}': {ve}. This often means insufficient variance.")
        return None
    except Exception as e:
        logger.error(f"Error calculating half-life for '{spread.name}': {e}. Returning None.")
        return None


def get_spread_series(s1: pd.Series, s2: pd.Series, hedge_ratio: float) -> pd.Series:
    """
    Calculates the spread series for a pair: Spread = s1 - hedge_ratio * s2.

    Args:
        s1 (pd.Series): The first asset's price series.
        s2 (pd.Series): The second asset's price series.
        hedge_ratio (float): The hedge ratio to apply to s2.

    Returns:
        pd.Series: The calculated spread series. The index will be the common, non-null index of s1 and s2.
                   Returns an empty Series if no overlapping non-null data.
    """
    aligned_s1, aligned_s2 = _align_series_and_drop_na(s1, s2)

    if aligned_s1.empty or aligned_s2.empty:
        logger.warning(
            f"No overlapping non-null data for spread calculation between {s1.name} and {s2.name}. Returning empty Series.")
        return pd.Series(dtype=float)

    spread = aligned_s1 - hedge_ratio * aligned_s2
    spread.name = f"Spread_{s1.name}_{s2.name}"
    logger.debug(f"Calculated spread for {s1.name}-{s2.name} with hedge ratio {hedge_ratio:.4f}. Length: {len(spread)}")
    return spread


def find_cointegrated_pairs(
        prices: pd.DataFrame,
        p_threshold: float = 0.05,
        min_half_life: Optional[int] = None,  # Made optional
        max_half_life: Optional[int] = None,  # Made optional
        integration_test_min_obs: int = 20,
        adf_regression_type: str = 'c'
) -> List[Dict]:
    """
    Finds cointegrated pairs within a DataFrame of prices using the Engle-Granger two-step method.
    The process involves:
    1. Checking if both series are I(1).
    2. Performing the Engle-Granger cointegration test on the pair.
    3. Calculating the hedge ratio via OLS regression of one on the other.
    4. Calculating the half-life of mean reversion for the resulting spread.

    Args:
        prices (pd.DataFrame): DataFrame of asset prices, where columns are asset symbols and index is datetime.
        p_threshold (float): The p-value threshold for the cointegration test (Engle-Granger).
        min_half_life (Optional[int]): Minimum acceptable half-life of mean reversion for the spread in periods.
                                        If None, this criterion is skipped.
        max_half_life (Optional[int]): Maximum acceptable half-life of mean reversion for the spread in periods.
                                        If None, this criterion is skipped.
        integration_test_min_obs (int): Minimum observations for ADF test in check_integration_order
                                        and for cointegration test.
        adf_regression_type (str): Regression type for ADF test and `coint` ('c', 'ct', 'nc').

    Returns:
        List[Dict]: A list of dictionaries, each describing a cointegrated pair that meets
                    all criteria. Dictionaries contain 'asset1', 'asset2', 'p_value',
                    'hedge_ratio', and 'half_life'. Sorted by p-value in ascending order.
    """
    n = prices.shape[1]
    symbols = prices.columns.tolist()
    cointegrated_pairs = []

    # Drop NaNs from the whole price DataFrame once to ensure consistent indices for pairs
    prices_clean = prices.dropna()
    if prices_clean.empty:
        logger.warning("Prices DataFrame is empty after dropping NaNs. No pairs to find.")
        return []

    if len(prices_clean) < integration_test_min_obs:
        logger.warning(
            f"Not enough data points ({len(prices_clean)}) in cleaned prices for integration tests. Min required: {integration_test_min_obs}. Returning empty list.")
        return []

    logger.info(f"Starting to find cointegrated pairs among {n} assets (total {n * (n - 1) // 2} pairs).")

    # Pre-calculate integration order for all series to avoid redundant calculations
    integration_orders: Dict[str, Optional[int]] = {}
    for symbol in symbols:
        integration_orders[symbol] = check_integration_order(prices_clean[symbol], p_threshold=0.05,
                                                             min_observations=integration_test_min_obs,
                                                             regression=adf_regression_type)
        logger.debug(f"Integration order for {symbol}: I({integration_orders[symbol]}).")

    for i in range(n):
        for j in range(i + 1, n):
            s1_name, s2_name = symbols[i], symbols[j]
            s1 = prices_clean[s1_name]
            s2 = prices_clean[s2_name]

            # Use pre-calculated integration orders
            s1_order = integration_orders.get(s1_name)
            s2_order = integration_orders.get(s2_name)

            # Only proceed if both series are I(1)
            if s1_order != 1 or s2_order != 1:
                logger.debug(
                    f"Skipping {s1_name}-{s2_name}: Not both I(1) for Engle-Granger. S1: I({s1_order}), S2: I({s2_order}).")
                continue

            # Ensure sufficient data for coint test, even after potential internal alignment
            if len(s1) < integration_test_min_obs or len(s2) < integration_test_min_obs:
                logger.debug(
                    f"Skipping {s1_name}-{s2_name}: Insufficient data ({len(s1)}/{len(s2)}) for coint test within window. Min required: {integration_test_min_obs}.")
                continue

            # Perform Engle-Granger cointegration test
            try:
                # `coint` internally aligns and drops NaNs. Uses OLS residuals.
                # Use the same trend parameter for the cointegration test as for ADF tests.
                _, p_value, _ = coint(s1, s2, trend=adf_regression_type)
            except ValueError as ve:
                logger.warning(
                    f"Cointegration test failed for {s1_name}-{s2_name} due to ValueError (e.g., all identical values, singular matrix): {ve}. Skipping pair.")
                continue
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during cointegration test for {s1_name}-{s2_name}: {e}. Skipping pair.")
                continue

            if p_value < p_threshold:
                # Calculate hedge ratio and spread using OLS
                # OLS needs data to be aligned and non-null
                aligned_s1_ols, aligned_s2_ols = _align_series_and_drop_na(s1, s2)
                if aligned_s1_ols.empty or aligned_s2_ols.empty:
                    logger.warning(
                        f"No aligned data for OLS regression between {s1_name} and {s2_name}. Skipping pair.")
                    continue
                if aligned_s2_ols.nunique() <= 1:  # Check for constant independent variable
                    logger.warning(
                        f"Independent variable {s2_name} is constant for OLS. Skipping pair {s1_name}-{s2_name}.")
                    continue

                try:
                    # Regress s1 on s2 to get hedge ratio (s1 = beta * s2 + epsilon)
                    # Use `add_constant` from `statsmodels.tools.add_constant`
                    model = OLS(aligned_s1_ols, add_constant(aligned_s2_ols)).fit()

                    # model.params[1] is the coefficient for s2 (the hedge ratio)
                    # Accessing by name is safer if column names are propagated
                    hedge_ratio = model.params.get(aligned_s2_ols.name, None)
                    if hedge_ratio is None and len(model.params) > 1:  # Fallback to position
                        hedge_ratio = model.params.iloc[1]

                    if hedge_ratio is None:
                        logger.warning(
                            f"OLS model for {s1_name}-{s2_name} did not return a valid hedge ratio. Parameters: {model.params.index.tolist()}. Skipping.")
                        continue

                    # Calculate the spread using this hedge ratio
                    spread = get_spread_series(aligned_s1_ols, aligned_s2_ols, hedge_ratio)
                except Exception as e:
                    logger.error(f"OLS regression failed for {s1_name}-{s2_name}: {e}. Skipping pair.")
                    continue

                # Check half-life of the spread
                half_life = calculate_half_life(spread)

                # Filter by half-life if min/max thresholds are provided
                if half_life is None or not np.isfinite(half_life):  # Handle None, NaN or infinite half-life
                    logger.debug(f"Skipping {s1_name}-{s2_name}: Invalid or infinite half-life ({half_life}).")
                    continue

                half_life_meets_criteria = True
                if min_half_life is not None and half_life < min_half_life:
                    half_life_meets_criteria = False
                    logger.debug(
                        f"Skipping {s1_name}-{s2_name}: Half-life ({half_life:.2f}) < min_half_life ({min_half_life}).")
                if max_half_life is not None and half_life > max_half_life:
                    half_life_meets_criteria = False
                    logger.debug(
                        f"Skipping {s1_name}-{s2_name}: Half-life ({half_life:.2f}) > max_half_life ({max_half_life}).")

                if half_life_meets_criteria:
                    cointegrated_pairs.append({
                        'asset1': s1_name,
                        'asset2': s2_name,
                        'p_value': p_value,
                        'hedge_ratio': hedge_ratio,
                        'half_life': half_life
                    })
                    logger.info(
                        f"Found cointegrated pair: {s1_name}-{s2_name} (P-value: {p_value:.4f}, Hedge Ratio: {hedge_ratio:.4f}, Half-life: {half_life:.2f} periods)")
            else:
                logger.debug(f"Skipping {s1_name}-{s2_name}: Cointegration P-value ({p_value:.4f}) >= {p_threshold}.")

    # Sort by p-value for a ranked list of pairs
    return sorted(cointegrated_pairs, key=lambda x: x['p_value'])


def rolling_cointegration_test(
        s1: pd.Series,
        s2: pd.Series,
        window: int,
        min_observations_for_coint: int = 20,  # Minimum observations for coint/OLS within window
        adf_regression_type: str = 'c'
) -> pd.DataFrame:
    """
    Performs a rolling Engle-Granger cointegration test and calculates the rolling hedge ratio
    to assess the stability of the relationship between two series over time.

    Args:
        s1 (pd.Series): The first time series (dependent variable in OLS).
        s2 (pd.Series): The second time series (independent variable in OLS).
        window (int): The rolling window size.
        min_observations_for_coint (int): Minimum number of non-null observations required
                                          within a window for the cointegration test and OLS.
        adf_regression_type (str): Regression type for the `coint` test ('c', 'ct', 'nc').

    Returns:
        pd.DataFrame: DataFrame with 'p_value_coint' and 'hedge_ratio_ols' for each rolling window.
                      The index is the end date of each window. NaNs will be present for windows
                      that don't meet criteria or where calculations fail.
    """
    results = []

    # Align and drop NaNs once at the beginning to simplify rolling window logic
    aligned_s1, aligned_s2 = _align_series_and_drop_na(s1, s2)
    if aligned_s1.empty:
        logger.warning(
            f"Input series '{s1.name}' and '{s2.name}' are empty after alignment and dropping NaNs. Cannot perform rolling cointegration.")
        return pd.DataFrame(columns=['p_value_coint', 'hedge_ratio_ols'])

    if len(aligned_s1) < window:
        logger.warning(
            f"Length of aligned series ({len(aligned_s1)}) is less than window size ({window}). Cannot perform rolling cointegration.")
        return pd.DataFrame(columns=['p_value_coint', 'hedge_ratio_ols'])

    for i in range(window, len(aligned_s1) + 1):
        window_s1 = aligned_s1.iloc[i - window:i]
        window_s2 = aligned_s2.iloc[i - window:i]
        current_date = aligned_s1.index[i - 1]  # End date of the current window

        # Check if current window has enough data points for tests
        if len(window_s1) < min_observations_for_coint or window_s1.nunique() <= 1 or \
                len(window_s2) < min_observations_for_coint or window_s2.nunique() <= 1:
            logger.debug(f"Window ending {current_date} has insufficient or constant data for tests. Skipping.")
            results.append({
                'date': current_date,
                'p_value_coint': np.nan,
                'hedge_ratio_ols': np.nan
            })
            continue

        p_value = np.nan
        hedge_ratio = np.nan

        try:
            # Cointegration test for the window
            _, p_value, _ = coint(window_s1, window_s2, trend=adf_regression_type)

            # OLS regression to get hedge ratio
            model = OLS(window_s1, add_constant(window_s2)).fit()
            if len(model.params) > 1:
                # Attempt to get by name, fallback to position
                hedge_ratio = model.params.get(window_s2.name, None)
                if hedge_ratio is None:
                    hedge_ratio = model.params.iloc[1]  # Assume second param is the coefficient
            else:
                logger.warning(
                    f"OLS model for window ending {current_date} did not return enough parameters for hedge ratio.")
                hedge_ratio = np.nan  # Keep as NaN if not enough params
        except ValueError as ve:
            logger.warning(f"ValueError in rolling coint/OLS for window ending {current_date}: {ve}. Skipping.")
        except Exception as e:
            logger.error(f"Unexpected error in rolling coint/OLS for window ending {current_date}: {e}. Skipping.")

        results.append({
            'date': current_date,
            'p_value_coint': p_value,
            'hedge_ratio_ols': hedge_ratio
        })

    logger.info(f"Completed rolling cointegration test for {s1.name}-{s2.name} with window {window}.")
    return pd.DataFrame(results).set_index('date')


def find_cointegrating_vectors(
        prices: pd.DataFrame,
        det_order: int = 0,  # Order of deterministic trend in the data: -1 (no trend), 0 (constant), 1 (linear trend)
        k_ar_diff: int = 1,  # Number of lagged differences in the VAR model
        normalize_to_asset: Optional[str] = None  # Symbol to normalize the cointegrating vector to 1
) -> pd.DataFrame:
    """
    Uses the Johansen test to find cointegrating vectors in a multivariate system.
    This test is typically used for 3 or more I(1) time series.

    Args:
        prices (pd.DataFrame): DataFrame of asset prices. Columns are asset symbols, index is datetime.
                               Must contain at least 2 columns, but ideally 3+.
        det_order (int): Order of deterministic trend in the data:
                         -1: No deterministic trend (series are I(1) without drift, CI is I(0) without trend)
                          0: Constant (series are I(1) with drift, CI is I(0) with zero mean)
                          1: Linear trend (series are I(1) with linear trend, CI is I(0) with non-zero mean)
        k_ar_diff (int): Number of lagged differences in the VAR model.
                         (k_ar_diff = p-1 where p is the order of the VAR model).
                         A common starting point is 1. Higher values increase degrees of freedom.
        normalize_to_asset (Optional[str]): If provided, the cointegrating vectors will be
                                             normalized such that the coefficient for this asset is 1.

    Returns:
        pd.DataFrame: A DataFrame where columns are cointegrating vectors and index are asset symbols.
                      Returns an empty DataFrame if no cointegrating relationships found or input is invalid.
    """
    prices_clean_df = prices.dropna()
    if prices_clean_df.empty:
        logger.warning("Prices DataFrame is empty after dropping NaNs. Cannot perform Johansen test.")
        return pd.DataFrame()

    num_series = prices_clean_df.shape[1]
    if num_series < 2:
        raise ValueError(f"Johansen test requires at least 2 time series. Got {num_series}.")
    if num_series < 3:
        logger.info(
            f"Johansen test with {num_series} series. Often more useful with 3 or more for multivariate insights.")

    # Check for constant series in the input, which can cause issues with VAR models
    for col in prices_clean_df.columns:
        if prices_clean_df[col].nunique() <= 1:
            logger.warning(
                f"Column '{col}' is constant. This can cause issues with the Johansen test. Consider removing or handling before calling this function.")
            # For robustness, we could filter here, but then the result DataFrame columns might change unexpectedly.
            # Better to inform the user.

    # Check for sufficient observations vs. k_ar_diff and num_series
    min_obs_for_johansen = num_series + k_ar_diff + 2  # Rule of thumb
    if len(prices_clean_df) < min_obs_for_johansen:
        logger.warning(
            f"Insufficient observations ({len(prices_clean_df)}) for Johansen test with {num_series} series and k_ar_diff={k_ar_diff}. "
            f"Need at least {min_obs_for_johansen}. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        # coint_johansen expects the full time series as input
        result = coint_johansen(prices_clean_df, det_order, k_ar_diff)
    except Exception as e:
        logger.error(
            f"Johansen test failed: {e}. Ensure series are I(1) and enough observations for k_ar_diff. Returning empty DataFrame.")
        return pd.DataFrame()

    # Determine the number of cointegrating relationships (rank) based on trace statistic
    # result.lr1 are trace statistics, result.cvt are critical values for trace statistic
    num_relationships_trace = np.sum(result.lr1 > result.cvt[:, det_order])
    # Can also use max-eigenvalue statistic: result.lr2 are max-eigenvalue statistics, result.cev are critical values
    # num_relationships_max_eig = np.sum(result.lr2 > result.cev[:, det_order])

    if num_relationships_trace == 0:
        logger.info(
            "No cointegrating relationships found by Johansen test (Trace statistic). Returning empty DataFrame.")
        return pd.DataFrame()

    logger.info(f"Found {num_relationships_trace} cointegrating relationship(s) based on Trace statistic.")

    # Extract the cointegrating vectors (eigenvectors corresponding to significant eigenvalues)
    # The first `num_relationships_trace` columns of result.evec are the cointegrating vectors
    cointegrating_vectors = pd.DataFrame(result.evec[:, :num_relationships_trace], index=prices_clean_df.columns)
    cointegrating_vectors.columns = [f'Vector_{i + 1}' for i in range(num_relationships_trace)]

    # Normalize vectors if requested
    if normalize_to_asset:
        if normalize_to_asset not in prices_clean_df.columns:
            logger.warning(f"Normalization asset '{normalize_to_asset}' not found in prices. Skipping normalization.")
        else:
            normalized_vectors = cointegrating_vectors.copy()
            for col in cointegrating_vectors.columns:
                norm_coeff = cointegrating_vectors.loc[normalize_to_asset, col]
                if pd.isna(norm_coeff) or abs(norm_coeff) < 1e-9:  # Check for NaN or near-zero
                    logger.warning(
                        f"Coefficient for '{normalize_to_asset}' in '{col}' is NaN or near zero ({norm_coeff:.2e}). Skipping normalization for this vector and setting to NaN.")
                    normalized_vectors[col] = np.nan  # Mark as NaN if normalization failed
                else:
                    normalized_vectors[col] = cointegrating_vectors[col] / norm_coeff

            # Drop vectors that became all NaN due to normalization failure
            normalized_vectors.dropna(axis=1, how='all', inplace=True)
            if normalized_vectors.empty:
                logger.warning(
                    "All cointegrating vectors became NaN after normalization due to zero/NaN coefficients. Returning empty DataFrame.")
                return pd.DataFrame()

            cointegrating_vectors = normalized_vectors
            logger.info(f"Cointegrating vectors normalized to '{normalize_to_asset}'.")

    return cointegrating_vectors


# --- Example Usage (main function and dummy data for testing) ---
def main():
    # 1. Generate Sample Data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500)

    # Stationary series (I(0))
    s_i0 = pd.Series(np.random.normal(0, 1, 500), index=dates, name='I0_Series')

    # Non-stationary I(1) series (Random Walk)
    s_i1_a = pd.Series(np.random.normal(0, 1, 500), index=dates, name='I1_A').cumsum() + 100
    s_i1_b = pd.Series(np.random.normal(0, 1, 500), index=dates, name='I1_B').cumsum() + 102

    # Cointegrated pair (s_i1_c and s_i1_d)
    # The spread between s_i1_d and s_i1_c * 0.8 should be stationary
    noise_stationary = pd.Series(np.random.normal(0, 0.5, 500), index=dates)  # Mean-reverting component
    s_i1_c = pd.Series(np.random.normal(0, 1, 500), index=dates, name='I1_C').cumsum() + 50
    s_i1_d = s_i1_c * 0.8 + noise_stationary + 10  # s_i1_d is cointegrated with s_i1_c, with a mean of 10

    # Another I(1) series for multivariate test
    s_i1_e = pd.Series(np.random.normal(0, 1, 500), index=dates, name='I1_E').cumsum() + 200

    # Create a DataFrame of prices
    prices_df = pd.DataFrame({
        'I0_Series': s_i0,
        'I1_A': s_i1_a,
        'I1_B': s_i1_b,
        'I1_C': s_i1_c,
        'I1_D': s_i1_d,
        'I1_E': s_i1_e
    })

    # Introduce some NaNs to test robustness
    prices_df_with_nans = prices_df.copy()
    prices_df_with_nans.loc[prices_df_with_nans.index[10:15], 'I0_Series'] = np.nan
    prices_df_with_nans.loc[prices_df_with_nans.index[200:205], 'I1_C'] = np.nan
    prices_df_with_nans.loc[prices_df_with_nans.index[201:206], 'I1_D'] = np.nan  # Overlapping NaNs for alignment test

    print("--- Testing check_integration_order ---")
    print(f"I0_Series integration order: {check_integration_order(prices_df_with_nans['I0_Series'])}")
    print(f"I1_A integration order: {check_integration_order(prices_df_with_nans['I1_A'])}")
    print(f"I1_C integration order: {check_integration_order(prices_df_with_nans['I1_C'])}")
    print(f"I1_D integration order: {check_integration_order(prices_df_with_nans['I1_D'])}")
    # Test with custom regression type
    print(
        f"I1_A integration order (regression='ct'): {check_integration_order(prices_df_with_nans['I1_A'], regression='ct')}")
    # Test with very short series
    print(f"Short series integration order: {check_integration_order(prices_df_with_nans['I0_Series'].iloc[:5])}")
    # Test with constant series
    print(f"Constant series integration order: {check_integration_order(pd.Series([100] * 50, index=dates[:50]))}")

    print("\n--- Testing calculate_half_life ---")
    # For a mean-reverting spread, half-life should be finite
    spread_c_d = get_spread_series(prices_df_with_nans['I1_C'], prices_df_with_nans['I1_D'],
                                   0.8)  # Approx hedge ratio for I1_C, I1_D
    half_life_c_d = calculate_half_life(spread_c_d)
    print(f"Half-life of I1_C/I1_D spread: {half_life_c_d:.2f} periods" if half_life_c_d is not None and np.isfinite(
        half_life_c_d) else f"Half-life of I1_C/I1_D spread: {half_life_c_d}")

    # For a random walk spread, half-life should be infinite
    # Spread of two independent random walks is also a random walk, thus I(1)
    spread_a_b = get_spread_series(prices_df_with_nans['I1_A'], prices_df_with_nans['I1_B'], 1.0)
    half_life_a_b = calculate_half_life(spread_a_b)
    print(f"Half-life of I1_A/I1_B spread: {half_life_a_b:.2f} periods" if half_life_a_b is not None and np.isfinite(
        half_life_a_b) else f"Half-life of I1_A/I1_B spread: {half_life_a_b}")

    # Test with a very short spread
    print(f"Half-life of short spread: {calculate_half_life(spread_c_d.iloc[:5])}")

    print("\n--- Testing find_cointegrated_pairs ---")
    cointegrated_pairs = find_cointegrated_pairs(prices_df_with_nans, p_threshold=0.05, min_half_life=10,
                                                 max_half_life=100)
    if cointegrated_pairs:
        for pair in cointegrated_pairs:
            print(
                f"Pair: {pair['asset1']}-{pair['asset2']}, P-value: {pair['p_value']:.4f}, Hedge Ratio: {pair['hedge_ratio']:.4f}, Half-life: {pair['half_life']:.2f}")
    else:
        print("No cointegrated pairs found based on current criteria.")

    print("\n--- Testing find_cointegrated_pairs without half-life filter ---")
    cointegrated_pairs_no_hl_filter = find_cointegrated_pairs(prices_df_with_nans, p_threshold=0.05)
    if cointegrated_pairs_no_hl_filter:
        for pair in cointegrated_pairs_no_hl_filter:
            print(
                f"Pair: {pair['asset1']}-{pair['asset2']}, P-value: {pair['p_value']:.4f}, Hedge Ratio: {pair['hedge_ratio']:.4f}, Half-life: {pair['half_life']:.2f}")
    else:
        print("No cointegrated pairs found without half-life filter.")

    print("\n--- Testing rolling_cointegration_test ---")
    rolling_results = rolling_cointegration_test(prices_df_with_nans['I1_C'], prices_df_with_nans['I1_D'], window=60)
    print("Rolling Cointegration Test Results (first 5 rows):")
    print(rolling_results.head())
    print("Rolling Cointegration Test Results (last 5 rows):")
    print(rolling_results.tail())
    print(f"Number of NaNs in rolling p_value: {rolling_results['p_value_coint'].isnull().sum()}")
    print(f"Number of NaNs in rolling hedge_ratio: {rolling_results['hedge_ratio_ols'].isnull().sum()}")

    # Test a non-cointegrated pair for rolling
    rolling_no_coint = rolling_cointegration_test(prices_df_with_nans['I1_A'], prices_df_with_nans['I1_B'], window=60)
    print("\nRolling Cointegration Test for non-cointegrated pair (first 5 rows):")
    print(rolling_no_coint.head())
    print(f"Number of NaNs in rolling p_value (non-coint): {rolling_no_coint['p_value_coint'].isnull().sum()}")

    print("\n--- Testing find_cointegrating_vectors (Johansen Test) ---")
    # Multivariate series (I1_A, I1_C, I1_D, I1_E)
    multi_series = prices_df_with_nans[['I1_A', 'I1_C', 'I1_D', 'I1_E']]

    # Example 1: No normalization
    c_vectors = find_cointegrating_vectors(multi_series, det_order=0, k_ar_diff=1)
    if not c_vectors.empty:
        print("\nCointegrating Vectors (no normalization):")
        print(c_vectors)
    else:
        print("\nNo cointegrating vectors found (no normalization).")

    # Example 2: Normalize to a specific asset
    c_vectors_norm = find_cointegrating_vectors(multi_series, det_order=0, k_ar_diff=1, normalize_to_asset='I1_C')
    if not c_vectors_norm.empty:
        print("\nCointegrating Vectors (normalized to I1_C):")
        print(c_vectors_norm)
    else:
        print("\nNo cointegrating vectors found (normalized to I1_C).")

    # Example 3: Test with only 2 series for Johansen (should not raise ValueError due to new logic, but log info)
    print("\n--- Testing Johansen with 2 series ---")
    c_vectors_2 = find_cointegrating_vectors(prices_df_with_nans[['I1_C', 'I1_D']], det_order=0, k_ar_diff=1)
    if not c_vectors_2.empty:
        print("Cointegrating Vectors for 2 series:")
        print(c_vectors_2)
    else:
        print("No cointegrating vectors found for 2 series.")

    # Example 4: Test with constant series in multivariate data
    multi_series_with_constant = prices_df_with_nans[['I1_A', 'I1_C']].copy()
    multi_series_with_constant['CONSTANT_SERIES'] = 100
    print("\n--- Testing Johansen with a constant series ---")
    c_vectors_const = find_cointegrating_vectors(multi_series_with_constant, det_order=0, k_ar_diff=1)
    if not c_vectors_const.empty:
        print("Cointegrating Vectors with constant series:")
        print(c_vectors_const)
    else:
        print("No cointegrating vectors found with constant series.")


if __name__ == "__main__":
    main()