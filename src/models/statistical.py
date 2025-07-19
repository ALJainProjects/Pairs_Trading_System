import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.add_constant import add_constant
from typing import Tuple, Optional, Dict, List, Union
from config.logging_config import logger


class StatisticalModel:
    """
    Provides a suite of statistical methods for analyzing asset pairs, including
    cointegration, hedge ratio calculation, spread analysis, and half-life of mean reversion.
    """

    def __init__(self):
        """Initializes the StatisticalModel."""
        logger.info("StatisticalModel initialized.")

    def _align_series(self, *series: pd.Series) -> List[pd.Series]:
        """
        Aligns multiple pandas Series based on their common index, sorts them,
        and drops rows where any of the series has a NaN value within the common index.

        Args:
            *series: Variable number of pandas Series to align.

        Returns:
            List[pd.Series]: A list of aligned and cleaned Series. Returns an empty list
                             if no series are provided or if all common data is NaN.
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
            return [pd.Series(dtype=s.dtype) for s in series]  # Return empty series of original types

        # Ensure index is sorted for time series operations
        df_aligned = df_aligned.sort_index()

        result_series = [df_aligned.iloc[:, i].rename(series[i].name) for i in range(len(series))]

        if any(s.empty for s in result_series):
            # This check should ideally not trigger if df_aligned is not empty, but as a safeguard.
            logger.warning("After alignment and final NaN cleaning, at least one resulting series is empty. "
                           "This indicates an issue with the common index or data sparsity.")

        logger.debug(f"Aligned {len(series)} series. Original lengths: {[len(s) for s in series]}. "
                     f"Aligned length: {len(result_series[0]) if result_series else 0}.")
        return result_series

    def cointegration_test(self, asset1: pd.Series, asset2: pd.Series, alpha: float = 0.05) -> \
            Tuple[bool, float, Optional[Dict[str, float]]]:
        """
        Performs the Engle-Granger cointegration test. The null hypothesis is that there is no
        cointegration. A low p-value (e.g., < alpha) leads to the rejection of the null hypothesis,
        suggesting the pair is cointegrated.

        Args:
            asset1 (pd.Series): Price series of the first asset (dependent variable in implied regression).
            asset2 (pd.Series): Price series of the second asset (independent variable in implied regression).
            alpha (float): Significance level for the test (e.g., 0.01, 0.05, 0.10).

        Returns:
            Tuple[bool, float, Optional[Dict]]: A tuple containing:
                - bool: True if the pair is cointegrated at the specified alpha level, False otherwise.
                - float: The p-value of the test.
                - Optional[Dict]: A dictionary of critical values for 1%, 5%, and 10% significance levels,
                                  or None if the test cannot be run (e.g., insufficient data, error).
        """
        aligned_asset1, aligned_asset2 = self._align_series(asset1, asset2)

        if len(aligned_asset1) < 50:  # Minimum recommended for reliable cointegration test
            logger.warning(
                f"Insufficient data points for cointegration test. Need at least 50. Got {len(aligned_asset1)}. "
                "Returning non-cointegrated result.")
            return False, 1.0, None  # Return p-value of 1.0 to indicate no significance

        try:
            # coint returns (t-statistic, p-value, critical_values_dict)
            t_stat, pvalue, critical_values = coint(aligned_asset1, aligned_asset2)
            is_cointegrated = pvalue < alpha
            logger.info(
                f"Cointegration test for pair ({asset1.name} vs {asset2.name}): p-value={pvalue:.4f}, alpha={alpha}. Cointegrated: {is_cointegrated}.")
            # Rename critical values for clarity
            critical_values_dict = {f'critical_{k}': v for k, v in critical_values.items()}
            return is_cointegrated, pvalue, critical_values_dict
        except ValueError as ve:
            logger.error(f"ValueError during cointegration test (e.g., singular matrix, all NaNs): {ve}")
            return False, 1.0, None
        except Exception as e:
            logger.error(f"An unexpected error occurred during cointegration test: {e}")
            return False, 1.0, None

    def calculate_rolling_hedge_ratio(self, asset1: pd.Series, asset2: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculates the hedge ratio (beta) using a rolling Ordinary Least Squares (OLS) regression
        where asset1 (Y) is regressed on asset2 (X).

        Args:
            asset1 (pd.Series): Price series of the dependent asset (Y).
            asset2 (pd.Series): Price series of the independent asset (X).
            window (int): The rolling window size (number of periods) for the regression.
                          Must be at least 2 for OLS.

        Returns:
            pd.Series: A time series of the rolling hedge ratio (beta). The series will have
                       NaNs at the beginning corresponding to the window size, and will be
                       reindexed to the original `asset1` index, with `ffill` for initial NaNs.
        """
        # Align inputs first to ensure consistent data for the rolling window
        aligned_asset1, aligned_asset2 = self._align_series(asset1, asset2)

        if len(aligned_asset1) < window:
            logger.warning(f"Not enough common data points ({len(aligned_asset1)}) for rolling hedge ratio "
                           f"with window {window}. Returning all NaNs.")
            return pd.Series(np.nan, index=asset1.index)  # Return a series of NaNs with original index

        # Create a DataFrame from the aligned series
        df_aligned = pd.DataFrame({'Y': aligned_asset1, 'X': aligned_asset2})

        def ols_beta_func(window_df: pd.DataFrame) -> float:
            """Helper function to compute OLS beta for a single window."""
            if len(window_df) < 2:  # OLS requires at least 2 observations
                return np.nan

            # Use 'add' to ensure constant is added if not present (robust)
            X_const = add_constant(window_df['X'], has_constant='add')

            try:
                # Fit OLS model: Y ~ C + X. The coefficient of X is the hedge ratio (beta)
                model = OLS(window_df['Y'], X_const, missing='drop').fit()
                return model.params.get('X', np.nan)  # Safely get 'X' coefficient
            except Exception as e:
                # Log a debug message for individual window failures, not an error.
                logger.debug(f"OLS failed for a window (len: {len(window_df)}): {e}. Returning NaN.")
                return np.nan

        # Apply the OLS function over a rolling window. raw=False passes DataFrame chunks.
        # The result's index will be aligned with the right (end) edge of each window.
        hedge_ratios_rolling = df_aligned.rolling(window=window).apply(ols_beta_func, raw=False)

        # Select the 'Y' column which contains the beta values (as we applied the function on df_aligned)
        hedge_ratios = hedge_ratios_rolling['Y'].copy()

        # Reindex to the original asset1 index. This is crucial for maintaining the expected length
        # and index for subsequent calculations like spread.
        # ffill propagates the last valid hedge ratio forward to fill initial NaNs from rolling window.
        full_hedge_ratios = pd.Series(np.nan, index=asset1.index, name='HedgeRatio')
        full_hedge_ratios.update(hedge_ratios)  # Update existing NaNs with calculated values
        full_hedge_ratios = full_hedge_ratios.ffill()  # Fill leading NaNs

        logger.info(f"Calculated rolling hedge ratio with window {window}. "
                    f"First valid value at {full_hedge_ratios.first_valid_index()}.")
        return full_hedge_ratios

    def calculate_spread(self, asset1: pd.Series, asset2: pd.Series, hedge_ratio: pd.Series) -> pd.Series:
        """
        Calculates the spread between two assets using a dynamic or static hedge ratio.
        Spread = Asset1 - (Hedge Ratio * Asset2).

        Args:
            asset1 (pd.Series): Price series of the first asset.
            asset2 (pd.Series): Price series of the second asset.
            hedge_ratio (pd.Series): Time series of the rolling hedge ratio (or a single float for static).

        Returns:
            pd.Series: The calculated spread series. Its index will match the common index
                       of the input series. NaNs will be present where any input was NaN.
        """
        # Ensure all inputs are aligned for element-wise operations
        # This will return series with a common, sorted, non-NaN index
        aligned_asset1, aligned_asset2, aligned_hedge_ratio = self._align_series(asset1, asset2, hedge_ratio)

        if aligned_asset1.empty:
            logger.warning(
                "No common, non-NaN data after aligning inputs for spread calculation. Returning empty Series.")
            return pd.Series(dtype=float, index=asset1.index)

        # Perform calculation only on the aligned part
        spread = aligned_asset1 - (aligned_hedge_ratio * aligned_asset2)
        spread.name = f"Spread_{asset1.name}_{asset2.name}"

        # Reindex to original asset1 index and fill missing values with NaN
        # This is important to maintain the original time index structure for consistent downstream use.
        full_spread = pd.Series(np.nan, index=asset1.index, name=spread.name)
        full_spread.loc[spread.index] = spread.values  # Assign values back to correct positions

        logger.info(f"Calculated spread. First valid value at {full_spread.first_valid_index()}.")
        return full_spread

    def calculate_zscore(self, spread: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculates the z-score of the spread to identify trading opportunities.
        Z-score = (Spread - Rolling Mean) / Rolling Standard Deviation.

        Args:
            spread (pd.Series): The spread series.
            window (int): The rolling window size for mean and standard deviation calculation.

        Returns:
            pd.Series: The calculated z-score series. NaNs will be present at the beginning
                       due to the rolling window and where standard deviation is zero.
        """
        spread_clean = spread.dropna()

        if spread_clean.empty:
            logger.warning("Spread series is empty or all NaNs after dropping. Cannot calculate z-score.")
            return pd.Series(np.nan, index=spread.index, name="Z_Score")

        if len(spread_clean) < window:
            logger.warning(f"Not enough valid data points ({len(spread_clean)}) for z-score calculation "
                           f"with window {window}. Returning NaNs.")
            return pd.Series(np.nan, index=spread.index, name="Z_Score")

        rolling_mean = spread_clean.rolling(window=window).mean()
        # Add a small epsilon to standard deviation to prevent division by zero for constant windows
        rolling_std = spread_clean.rolling(window=window).std()

        # Handle cases where rolling_std might be 0 (e.g., constant spread over window)
        # Replacing 0s with NaN will result in NaN z-scores for those periods, which is appropriate.
        rolling_std_safe = rolling_std.replace(0, np.nan)

        z_score = (spread_clean - rolling_mean) / rolling_std_safe
        z_score.name = "Z_Score"

        # Reindex to original spread index to maintain consistent length
        full_z_score = pd.Series(np.nan, index=spread.index, name=z_score.name)
        full_z_score.update(z_score)  # Update non-NaN values

        logger.info(f"Calculated z-score with window {window}. "
                    f"First valid value at {full_z_score.first_valid_index()}.")
        return full_z_score

    def calculate_half_life(self, spread: pd.Series) -> Optional[float]:
        """
        Calculates the half-life of mean reversion for a spread series.
        This is calculated as -ln(2) / lambda, where lambda is the coefficient
        from regressing delta(spread) on lagged spread.
        Half-life is calculated only if the spread is found to be stationary (ADF p-value <= 0.05).

        Args:
            spread (pd.Series): The spread series.

        Returns:
            Optional[float]: The calculated half-life in periods (time units of the spread index),
                             or None if the spread is not stationary, if the calculation fails,
                             or if data is insufficient.
        """
        spread_clean = spread.dropna()

        if len(spread_clean) < 10:  # Minimum observations for ADF test and regression
            logger.warning(
                f"Insufficient data points ({len(spread_clean)}) for half-life calculation after dropping NaNs. Need at least 10.")
            return None

        if spread_clean.nunique() < 2:  # Need at least two unique values for variance/regression
            logger.warning(
                "Spread series has insufficient unique values for half-life calculation (e.g., constant spread).")
            return None

        # Perform Augmented Dickey-Fuller test for stationarity
        # Null hypothesis: unit root (non-stationary)
        # Alternative hypothesis: no unit root (stationary)
        try:
            adf_test_result = adfuller(spread_clean, regression='c')  # 'c' for constant, common for financial series
            adf_pvalue = adf_test_result[1]
            logger.info(f"ADF test p-value for spread: {adf_pvalue:.4f}")
        except ValueError as ve:
            logger.error(f"ValueError during ADF test for half-life: {ve}. Check spread data for NaNs/invariance.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during ADF test for half-life: {e}")
            return None

        # Half-life is meaningful only if the series is stationary
        if adf_pvalue > 0.05:
            logger.info("Spread is not stationary (ADF p-value > 0.05). Half-life of mean reversion is not applicable.")
            return None

        # Calculate delta_spread = spread(t) - spread(t-1)
        # Calculate lag_spread = spread(t-1)
        lag_spread = spread_clean.shift(1).dropna()
        delta_spread = spread_clean.diff().dropna()

        # Align indices after shifting and differencing
        aligned_lag_spread, aligned_delta_spread = self._align_series(lag_spread, delta_spread)

        if len(aligned_lag_spread) < 2:  # Need at least two points for OLS
            logger.warning(
                f"Insufficient aligned data points ({len(aligned_lag_spread)}) for OLS in half-life calculation after differencing.")
            return None

        try:
            # Regress delta_spread on lag_spread: delta_spread = lambda_ * lag_spread + c + epsilon
            # The coefficient of lag_spread is lambda_ (rate of mean reversion)
            # Add constant for intercept
            model = OLS(aligned_delta_spread, add_constant(aligned_lag_spread), missing='drop').fit()
            # Safely get 'aligned_lag_spread' coefficient. Use .get() in case it's named differently.
            lambda_ = model.params.get(aligned_lag_spread.name, None)
            if lambda_ is None:  # Fallback if column name not found, try by position
                if len(model.params) > 1:
                    lambda_ = model.params.iloc[1]
                else:
                    logger.error("Could not extract lambda_ from OLS model parameters.")
                    return None

            # Half-life = -ln(2) / lambda
            # lambda_ must be negative for mean reversion (i.e., when spread is high, it tends to decrease)
            if lambda_ < 0:
                half_life = -np.log(2) / lambda_
                logger.info(f"Half-life of mean reversion calculated: {half_life:.2f} periods.")
                return half_life
            else:
                logger.info(
                    f"Lambda ({lambda_:.4f}) is not negative or zero. Spread does not exhibit mean reversion in the expected direction.")
                return None
        except ValueError as ve:
            logger.error(
                f"ValueError during OLS regression for half-life calculation: {ve}. Check spread data for variance.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during OLS regression for half-life calculation: {e}")
            return None


# Example Usage (unchanged for demonstrating functionality)
if __name__ == "__main__":
    from datetime import datetime, timedelta

    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    # Asset 1 (Y) - base price with some trend
    asset1_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 200))

    # Asset 2 (X) - correlated with asset 1, but with some noise
    # To make them cointegrated, make asset2 a linear combination of asset1 plus stationary noise
    noise = np.random.normal(0, 0.2, 200) + np.sin(np.linspace(0, 10, 200))  # Stationary noise
    asset2_prices = 0.8 * asset1_prices + 5 + noise  # Example: Y = beta*X + C + spread (where spread is stationary)

    # Convert to pandas Series with DateTimeIndex
    asset1_series = pd.Series(asset1_prices, index=dates, name='Asset1')
    asset2_series = pd.Series(asset2_prices, index=dates, name='Asset2')

    # Introduce some NaNs to test alignment
    asset1_series_with_nans = asset1_series.copy()
    asset2_series_with_nans = asset2_series.copy()
    asset1_series_with_nans.iloc[5:10] = np.nan
    asset2_series_with_nans.iloc[7:12] = np.nan
    asset1_series_with_nans.iloc[150] = np.nan

    stats_model = StatisticalModel()

    print("\n--- Cointegration Test ---")
    is_coint, p_value, critical_vals = stats_model.cointegration_test(asset1_series_with_nans, asset2_series_with_nans)
    print(f"Is cointegrated (alpha=0.05)? {is_coint}")
    print(f"P-value: {p_value:.4f}")
    if critical_vals:
        print(f"Critical Values: {critical_vals}")

    # Test with non-cointegrated data (random walks)
    rw1 = pd.Series(np.cumsum(np.random.normal(0, 1, 200)), index=dates, name='RW1')
    rw2 = pd.Series(np.cumsum(np.random.normal(0, 1, 200)), index=dates, name='RW2')
    is_coint_rw, p_value_rw, _ = stats_model.cointegration_test(rw1, rw2)
    print(f"\nIs random walks cointegrated? {is_coint_rw} (P-value: {p_value_rw:.4f})")

    print("\n--- Rolling Hedge Ratio Calculation ---")
    rolling_hedge_ratio = stats_model.calculate_rolling_hedge_ratio(asset1_series_with_nans, asset2_series_with_nans,
                                                                    window=60)
    print(
        f"Rolling Hedge Ratio (first 5 and last 5 values):\n{rolling_hedge_ratio.head(5)}\n{rolling_hedge_ratio.tail(5)}")
    print(f"Number of NaNs in rolling hedge ratio: {rolling_hedge_ratio.isnull().sum()}")

    print("\n--- Spread Calculation ---")
    spread = stats_model.calculate_spread(asset1_series_with_nans, asset2_series_with_nans, rolling_hedge_ratio)
    print(f"Spread (first 5 and last 5 values):\n{spread.head(5)}\n{spread.tail(5)}")
    print(f"Number of NaNs in spread: {spread.isnull().sum()}")

    print("\n--- Z-score Calculation ---")
    z_score = stats_model.calculate_zscore(spread, window=60)
    print(f"Z-score (first 5 and last 5 values):\n{z_score.head(5)}\n{z_score.tail(5)}")
    print(f"Number of NaNs in Z-score: {z_score.isnull().sum()}")

    print("\n--- Half-Life Calculation ---")
    # For half-life, a truly mean-reverting series is best. The synthetic spread should be.
    half_life = stats_model.calculate_half_life(spread)
    if half_life is not None:
        print(f"Half-life of spread: {half_life:.2f} periods")
    else:
        print("Could not calculate half-life (spread not stationary or insufficient data).")

    # Test half-life with a known non-stationary series (random walk)
    half_life_rw = stats_model.calculate_half_life(rw1)
    if half_life_rw is not None:
        print(f"Half-life of RW1: {half_life_rw:.2f} periods (Should be None)")
    else:
        print("Could not calculate half-life for RW1 (expected, as it's not stationary).")