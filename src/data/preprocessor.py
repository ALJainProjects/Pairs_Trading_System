import pandas as pd
from typing import List, Optional, Dict, Union
from scipy.stats import zscore
from config.logging_config import logger
import numpy as np
import joblib  # For saving/loading scalers


class Preprocessor:
    """Data preprocessing class for financial time series data."""

    NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]
    VALID_NORMALIZATION_METHODS = ["min-max", "z-score", "robust"]
    VALID_IMPUTATION_METHODS = ["ffill", "bfill", "mean", "median", "interpolate"]
    VALID_OUTLIER_METHODS = ["zscore", "iqr"]

    def __init__(self, numeric_columns: Optional[List[str]] = None):
        """
        Initialize preprocessor with configurable column names.

        Args:
            numeric_columns: List of numeric column names to process
        """
        self.numeric_columns = numeric_columns or self.NUMERIC_COLS
        self.scaling_factors: Dict[str, Dict[str, Dict[str, float]]] = {}  # Stores scaling factors per symbol/global
        logger.info(f"Preprocessor initialized with numeric columns: {self.numeric_columns}")

    def clean_data(self, df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
        """
        Clean data by handling dates, NaNs (for key columns), and duplicates.

        Args:
            df: Input DataFrame
            date_column: Name of date column (e.g., 'Date' for historical, 'datetime' for real-time)

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data: handling dates, key NaNs, and duplicates")
        df = df.copy()

        if date_column not in df.columns:
            raise ValueError(f"DataFrame must contain a '{date_column}' column.")

        # Convert to datetime, coercing errors
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            original_type = df[date_column].dtype
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            invalid_dates = df[date_column].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates in '{date_column}' column. Coerced to NaT.")
            if original_type == object and pd.api.types.is_datetime64_any_dtype(df[date_column]):
                logger.info(f"Converted '{date_column}' from object to datetime type.")

        # Drop rows where essential columns are missing
        missing_before_key_cols = df[date_column].isna().sum() + df['Symbol'].isna().sum()
        df.dropna(subset=[date_column, "Symbol"], inplace=True)
        missing_after_key_cols = df[date_column].isna().sum() + df['Symbol'].isna().sum()  # Should be 0

        if missing_before_key_cols > 0 and missing_after_key_cols == 0:
            logger.info(f"Removed {missing_before_key_cols} rows with missing 'Symbol' or '{date_column}'.")

        # Sort for proper time series operations (e.g., ffill, interpolation)
        df.sort_values(by=[date_column, "Symbol"], inplace=True)

        # Handle duplicates
        duplicates_count = df.duplicated(subset=["Symbol", date_column], keep='last').sum()
        if duplicates_count > 0:
            logger.warning(
                f"Removing {duplicates_count} duplicate entries based on ['Symbol', '{date_column}'] by keeping last.")
            df.drop_duplicates(subset=["Symbol", date_column], keep='last', inplace=True)

        df.reset_index(drop=True, inplace=True)
        logger.info("Data cleaning complete.")
        return df

    def handle_missing_values(self,
                              df: pd.DataFrame,
                              method: str = "ffill",
                              groupby: Optional[str] = "Symbol") -> pd.DataFrame:
        """
        Handle missing values in numeric columns.

        Args:
            df: Input DataFrame.
            method: Imputation method ('ffill', 'bfill', 'mean', 'median', 'interpolate').
            groupby: Column to group by for imputation (None for global).

        Returns:
            DataFrame with missing values handled.
        """
        if method not in self.VALID_IMPUTATION_METHODS:
            raise ValueError(f"Imputation method must be one of {self.VALID_IMPUTATION_METHODS}.")

        logger.info(f"Handling missing values using '{method}' method.")
        df_processed = df.copy()
        missing_counts_before = df_processed[self.numeric_columns].isna().sum()

        def impute_group(group):
            for col in self.numeric_columns:
                if col not in group.columns:
                    continue

                if group[col].isna().any():
                    initial_missing = group[col].isna().sum()
                    if method == "ffill":
                        group[col].fillna(method='ffill', inplace=True)
                    elif method == "bfill":
                        group[col].fillna(method='bfill', inplace=True)
                    elif method == "mean":
                        group[col].fillna(group[col].mean(), inplace=True)
                    elif method == "median":
                        group[col].fillna(group[col].median(), inplace=True)
                    elif method == "interpolate":
                        group[col].interpolate(method='linear', limit_direction='both', inplace=True)

                    # After ffill/bfill/interpolate, there might still be NaNs at the very start/end
                    # Fill any remaining with mean/median if numerical method chosen, or 0/appropriate default
                    if group[col].isna().any():
                        if method in ["mean", "median"]:
                            group[col].fillna(group[col].mean() if method == "mean" else group[col].median(),
                                              inplace=True)
                        else:  # Fallback for ffill/bfill/interpolate if leading/trailing NaNs remain
                            # For financial data, a 0 or the global mean/median might be a safer fallback
                            group[col].fillna(0, inplace=True)  # Or df_processed[col].mean() if global context needed

                    final_missing = group[col].isna().sum()
                    if initial_missing > final_missing:
                        logger.debug(f"Imputed {initial_missing - final_missing} values in '{col}' for group.")
            return group

        if groupby:
            df_processed = df_processed.groupby(groupby, group_keys=False).apply(impute_group)
        else:
            df_processed = impute_group(df_processed)

        missing_counts_after = df_processed[self.numeric_columns].isna().sum()
        for col in self.numeric_columns:
            if missing_counts_before[col] > missing_counts_after[col]:
                logger.info(
                    f"Handled {missing_counts_before[col] - missing_counts_after[col]} missing values in column '{col}'.")
        return df_processed

    def fit_transform_normalize_data(self,
                                     df: pd.DataFrame,
                                     method: str = "min-max",
                                     groupby: Optional[str] = "Symbol") -> pd.DataFrame:
        """
        Fits the scaler to the data and transforms it. Stores scaling factors.

        Args:
            df: Input DataFrame
            method: Normalization method ('min-max', 'z-score', or 'robust')
            groupby: Column to group by for normalization (None for global)

        Returns:
            DataFrame with normalized values
        """
        if method not in self.VALID_NORMALIZATION_METHODS:
            raise ValueError(f"Normalization method must be one of {self.VALID_NORMALIZATION_METHODS}.")

        logger.info(f"Fitting and transforming data using {method} normalization.")
        df_transformed = df.copy()
        self.scaling_factors.clear()  # Clear old factors before fitting new ones

        def _fit_transform_group(group):
            group_id = group['Symbol'].iloc[0] if 'Symbol' in group.columns else 'global'
            self.scaling_factors[group_id] = {}

            for col in self.numeric_columns:
                if col not in group.columns or group[col].isnull().all():
                    continue  # Skip if column not present or all NaNs

                if method == "min-max":
                    min_val = group[col].min()
                    max_val = group[col].max()
                    if max_val - min_val != 0:
                        group[col] = (group[col] - min_val) / (max_val - min_val)
                        self.scaling_factors[group_id][col] = {'min': min_val, 'max': max_val}
                    else:  # Handle constant values
                        group[col] = 0.0  # Normalized to 0 if constant
                        self.scaling_factors[group_id][col] = {'min': min_val,
                                                               'max': min_val + 1e-9}  # Prevent div by zero later
                        logger.warning(f"Column '{col}' for group '{group_id}' has constant values. Normalized to 0.")

                elif method == "z-score":
                    mean = group[col].mean()
                    std = group[col].std()
                    if std != 0:
                        group[col] = (group[col] - mean) / std
                        self.scaling_factors[group_id][col] = {'mean': mean, 'std': std}
                    else:  # Handle constant values
                        group[col] = 0.0  # Normalized to 0 if constant
                        self.scaling_factors[group_id][col] = {'mean': mean, 'std': 1.0}  # Prevent div by zero later
                        logger.warning(f"Column '{col}' for group '{group_id}' has constant values. Normalized to 0.")

                elif method == "robust":
                    q75, q25 = group[col].quantile([0.75, 0.25])
                    iqr = q75 - q25
                    if iqr != 0:
                        group[col] = (group[col] - q25) / iqr
                        self.scaling_factors[group_id][col] = {'q25': q25, 'iqr': iqr}
                    else:  # Handle constant values
                        group[col] = 0.0  # Normalized to 0 if constant
                        self.scaling_factors[group_id][col] = {'q25': q25, 'iqr': 1.0}  # Prevent div by zero later
                        logger.warning(f"Column '{col}' for group '{group_id}' has constant values. Normalized to 0.")
            return group

        if groupby and groupby in df_transformed.columns:
            df_transformed = df_transformed.groupby(groupby, group_keys=False).apply(_fit_transform_group)
        else:
            df_transformed = _fit_transform_group(df_transformed)

        logger.info("Normalization fit and transform complete.")
        return df_transformed

    def inverse_transform_normalize_data(self,
                                         df: pd.DataFrame,
                                         groupby: Optional[str] = "Symbol") -> pd.DataFrame:
        """
        Inverse transforms previously normalized data using stored scaling factors.

        Args:
            df: Input DataFrame with normalized values
            groupby: Column used for grouping during normalization (must match fit_transform)

        Returns:
            DataFrame with denormalized (original scale) values
        """
        if not self.scaling_factors:
            raise RuntimeError("No scaling factors found. Call fit_transform_normalize_data first.")

        logger.info("Inverse transforming normalized data.")
        df_denormalized = df.copy()

        # Determine the normalization method from the stored scaling factors (assuming consistent method)
        # This is a bit of a heuristic; a more robust way would be to store the method directly.
        first_group_id = next(iter(self.scaling_factors))
        first_col_factors = next(iter(self.scaling_factors[first_group_id].values()))
        if 'min' in first_col_factors:
            method = "min-max"
        elif 'mean' in first_col_factors:
            method = "z-score"
        elif 'q25' in first_col_factors:
            method = "robust"
        else:
            raise ValueError("Could not determine normalization method from stored scaling factors.")

        def _inverse_transform_group(group):
            group_id = group['Symbol'].iloc[0] if 'Symbol' in group.columns else 'global'
            group_scaling = self.scaling_factors.get(group_id)

            if not group_scaling:
                logger.warning(
                    f"No scaling factors found for group '{group_id}'. Skipping inverse transformation for this group.")
                return group

            for col in self.numeric_columns:
                if col not in group.columns or col not in group_scaling:
                    continue

                factors = group_scaling[col]
                if method == "min-max":
                    min_val = factors['min']
                    max_val = factors['max']
                    group[col] = group[col] * (max_val - min_val) + min_val

                elif method == "z-score":
                    mean = factors['mean']
                    std = factors['std']
                    group[col] = group[col] * std + mean

                elif method == "robust":
                    q25 = factors['q25']
                    iqr = factors['iqr']
                    group[col] = group[col] * iqr + q25
            return group

        if groupby and groupby in df_denormalized.columns:
            df_denormalized = df_denormalized.groupby(groupby, group_keys=False).apply(_inverse_transform_group)
        else:
            df_denormalized = _inverse_transform_group(df_denormalized)

        logger.info("Inverse transformation complete.")
        return df_denormalized

    def handle_outliers(self,
                        df: pd.DataFrame,
                        method: str = "zscore",
                        threshold: float = 3.0,
                        groupby: Optional[str] = "Symbol") -> pd.DataFrame:
        """
        Handle outliers using winsorization with specified method.

        Args:
            df: Input DataFrame
            method: Method to detect outliers ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            groupby: Column to group by for outlier detection

        Returns:
            DataFrame with outliers winsorized
        """
        if method not in self.VALID_OUTLIER_METHODS:
            raise ValueError(f"Outlier detection method must be one of {self.VALID_OUTLIER_METHODS}.")

        logger.info(f"Winsorizing outliers using {method} method with threshold {threshold}.")
        df_processed = df.copy()

        def winsorize_group(group):
            for col in self.numeric_columns:
                if col not in group.columns or group[col].isnull().all():
                    continue  # Skip if column not present or all NaNs

                # Convert to numpy array for zscore and percentile calculations
                # Drop NaNs for these calculations to avoid issues
                data_series = group[col].dropna()

                if data_series.empty:
                    continue

                n_outliers_group = 0
                if method == "zscore":
                    # Calculate z-scores for non-NaN values
                    col_mean = data_series.mean()
                    col_std = data_series.std()

                    if col_std == 0:  # Handle constant series
                        logger.debug(
                            f"Column '{col}' for group '{group.name}' has constant values. No z-score outliers.")
                        continue

                    z_scores = (data_series - col_mean) / col_std

                    # Identify outliers based on threshold
                    upper_mask_data = z_scores > threshold
                    lower_mask_data = z_scores < -threshold

                    # Get the actual values corresponding to the masks
                    upper_outliers = data_series[upper_mask_data]
                    lower_outliers = data_series[lower_mask_data]

                    # Determine bounds using non-outlier data
                    non_outlier_data = data_series[~(upper_mask_data | lower_mask_data)]

                    if not non_outlier_data.empty:
                        upper_bound = non_outlier_data.max()
                        lower_bound = non_outlier_data.min()
                    else:  # If all data are outliers or only a few points, default to original min/max
                        upper_bound = data_series.max()
                        lower_bound = data_series.min()
                        logger.warning(
                            f"No non-outlier data found for '{col}' in group '{group.name}'. Using series min/max as bounds.")

                    # Apply winsorization to the original group dataframe (including NaNs if present)
                    # We need to apply these back to the original `group[col]` series using its index.
                    # Create a temporary series with the same index as the original group[col]
                    temp_z_scores = pd.Series(np.nan, index=group.index)
                    temp_z_scores.loc[data_series.index] = z_scores  # Populate z-scores only where data exists

                    upper_mask = temp_z_scores > threshold
                    lower_mask = temp_z_scores < -threshold

                    n_outliers_group = (upper_mask | lower_mask).sum()

                    group.loc[upper_mask, col] = upper_bound
                    group.loc[lower_mask, col] = lower_bound

                elif method == "iqr":
                    q75, q25 = data_series.quantile([0.75, 0.25])
                    iqr = q75 - q25

                    if iqr == 0:  # Handle constant series
                        logger.debug(f"Column '{col}' for group '{group.name}' has constant values. No IQR outliers.")
                        continue

                    lower_bound = q25 - threshold * iqr
                    upper_bound = q75 + threshold * iqr

                    upper_mask = group[col] > upper_bound
                    lower_mask = group[col] < lower_bound

                    n_outliers_group = (upper_mask | lower_mask).sum()

                    group.loc[upper_mask, col] = upper_bound
                    group.loc[lower_mask, col] = lower_bound

                if n_outliers_group > 0:
                    logger.info(f"Winsorized {n_outliers_group} outliers in '{col}' for group '{group.name}'.")
            return group

        if groupby and groupby in df_processed.columns:
            # Pass group.name to the winsorize_group function for better logging context
            return df_processed.groupby(groupby, group_keys=False).apply(lambda x: winsorize_group(x.copy()))
        return winsorize_group(df_processed)

    def save_scalers(self, filepath: Union[str, Path]) -> None:
        """
        Saves the fitted scaling factors to a file using joblib.

        Args:
            filepath: The path to the file where scalers will be saved.
        """
        try:
            joblib.dump(self.scaling_factors, filepath)
            logger.info(f"Scaling factors saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scaling factors to {filepath}: {e}")
            raise

    def load_scalers(self, filepath: Union[str, Path]) -> None:
        """
        Loads scaling factors from a file.

        Args:
            filepath: The path to the file from which scalers will be loaded.
        """
        try:
            self.scaling_factors = joblib.load(filepath)
            logger.info(f"Scaling factors loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Scaler file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading scaling factors from {filepath}: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    from datetime import date, timedelta

    # Create sample data for demonstration
    dates = [date(2023, 1, i) for i in range(1, 11)] * 2
    symbols = ['AAPL'] * 10 + ['MSFT'] * 10

    data = {
        'Date': dates,
        'Symbol': symbols,
        'Open': [100 + i for i in range(10)] + [200 + i for i in range(10)],
        'High': [102 + i for i in range(10)] + [203 + i for i in range(10)],
        'Low': [98 + i for i in range(10)] + [197 + i for i in range(10)],
        'Close': [101 + i for i in range(10)] + [201 + i for i in range(10)],
        'Volume': [10000 + i * 100 for i in range(10)] + [20000 + i * 200 for i in range(10)],
    }
    df = pd.DataFrame(data)

    # Introduce some missing values and outliers for testing
    df.loc[3, 'Close'] = np.nan
    df.loc[12, 'Volume'] = np.nan
    df.loc[5, 'Open'] = 1000  # Outlier
    df.loc[15, 'High'] = 0.5  # Outlier

    # Add a duplicate
    df = pd.concat([df, df.iloc[0:1]], ignore_index=True)

    preprocessor = Preprocessor()

    print("Original DataFrame:")
    print(df)
    print("-" * 30)

    # 1. Clean Data
    cleaned_df = preprocessor.clean_data(df)
    print("\nCleaned DataFrame (duplicates removed, essential NaNs dropped):")
    print(cleaned_df)
    print("-" * 30)

    # 2. Handle Missing Values
    imputed_df = preprocessor.handle_missing_values(cleaned_df, method='interpolate')
    print("\nDataFrame after handling missing values (interpolate):")
    print(imputed_df)
    print("-" * 30)

    # 3. Handle Outliers
    outlier_handled_df = preprocessor.handle_outliers(imputed_df, method='zscore', threshold=2.0)
    print("\nDataFrame after handling outliers (zscore, threshold=2.0):")
    print(outlier_handled_df)
    print("-" * 30)

    # Verify outlier handling (manual check)
    print("\nVerifying outlier handling for AAPL Open (expected 1000 to be winsorized):")
    print(outlier_handled_df[outlier_handled_df['Symbol'] == 'AAPL'][['Date', 'Open']])
    print("\nVerifying outlier handling for MSFT High (expected 0.5 to be winsorized):")
    print(outlier_handled_df[outlier_handled_df['Symbol'] == 'MSFT'][['Date', 'High']])

    # 4. Normalize Data (fit_transform)
    normalized_df_minmax = preprocessor.fit_transform_normalize_data(outlier_handled_df.copy(), method='min-max')
    print("\nNormalized DataFrame (min-max):")
    print(normalized_df_minmax)
    print("-" * 30)

    # Save scalers
    scaler_filepath = "minmax_scaler.joblib"
    preprocessor.save_scalers(scaler_filepath)

    # Create a new preprocessor instance to load scalers
    new_preprocessor = Preprocessor()
    new_preprocessor.load_scalers(scaler_filepath)

    # Inverse Transform
    denormalized_df_minmax = new_preprocessor.inverse_transform_normalize_data(normalized_df_minmax.copy())
    print("\nDenormalized DataFrame (min-max inverse transform):")
    print(denormalized_df_minmax)
    print("\nAre original and denormalized values close? (should be True for numeric cols)")
    # Check if they are approximately equal due to floating point precision
    comparison = np.isclose(outlier_handled_df[preprocessor.numeric_columns],
                            denormalized_df_minmax[preprocessor.numeric_columns], atol=1e-6).all()
    print(comparison)
    print("-" * 30)

    # Test Z-score normalization
    normalized_df_zscore = preprocessor.fit_transform_normalize_data(outlier_handled_df.copy(), method='z-score')
    print("\nNormalized DataFrame (z-score):")
    print(normalized_df_zscore)

    # Test Robust normalization
    normalized_df_robust = preprocessor.fit_transform_normalize_data(outlier_handled_df.copy(), method='robust')
    print("\nNormalized DataFrame (robust):")
    print(normalized_df_robust)

    # Test with global normalization (groupby=None)
    print("\nTesting global (non-grouped) normalization:")
    global_norm_df = preprocessor.fit_transform_normalize_data(outlier_handled_df.copy(), method='z-score',
                                                               groupby=None)
    print(global_norm_df)
    global_denorm_df = preprocessor.inverse_transform_normalize_data(global_norm_df.copy(), groupby=None)
    print("\nGlobal denormalized:")
    print(global_denorm_df)

    # Clean up scaler file
    import os

    if os.path.exists(scaler_filepath):
        os.remove(scaler_filepath)
        logger.info(f"Removed temporary scaler file: {scaler_filepath}")