import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.stats import zscore
from config.logging_config import logger


class Preprocessor:
    """Data preprocessing class for financial time series data."""

    NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]
    VALID_METHODS = ["min-max", "z-score", "robust"]

    def __init__(self, numeric_columns: Optional[List[str]] = None):
        """
        Initialize preprocessor with configurable column names.

        Args:
            numeric_columns: List of numeric column names to process
        """
        self.numeric_columns = numeric_columns or self.NUMERIC_COLS

    def clean_data(self, df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
        """
        Clean data by handling dates, NaNs, and duplicates.

        Args:
            df: Input DataFrame
            date_column: Name of date column

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data: handling dates, NaNs, and duplicates")
        df = df.copy()

        if date_column not in df.columns:
            raise ValueError(f"DataFrame must contain a '{date_column}' column")

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            invalid_dates = df[date_column].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates")

        df.sort_values(by=[date_column, "Symbol"], inplace=True)
        missing_before = df.isna().sum()
        df.dropna(subset=[date_column, "Symbol"], inplace=True)

        duplicates = df.duplicated(subset=["Symbol", date_column], keep='last').sum()
        if duplicates > 0:
            logger.warning(f"Removing {duplicates} duplicate entries")
            df.drop_duplicates(subset=["Symbol", date_column], keep='last', inplace=True)

        missing_after = df.isna().sum()
        for col in df.columns:
            removed = missing_before[col] - missing_after[col]
            if removed > 0:
                logger.info(f"Removed {removed} missing values from {col}")

        df.reset_index(drop=True, inplace=True)
        return df

    def normalize_data(self,
                       df: pd.DataFrame,
                       method: str = "min-max",
                       groupby: Optional[str] = "Symbol") -> pd.DataFrame:
        """
        Normalize numeric columns using specified method and then denormalize.

        Args:
            df: Input DataFrame
            method: Normalization method ('min-max', 'z-score', or 'robust')
            groupby: Column to group by for normalization (None for global)

        Returns:
            DataFrame with cleaned but unnormalized values
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Method must be one of {self.VALID_METHODS}")

        logger.info(f"Normalizing and then denormalizing data using {method} method")
        df = df.copy()
        scaling_factors = {}

        def normalize_group(group):
            group_scaling = {}
            ticker = group['Symbol'].iloc[0] if 'Symbol' in group else 'global'

            for col in self.numeric_columns:
                if col not in group.columns:
                    continue

                if method == "min-max":
                    min_val = group[col].min()
                    max_val = group[col].max()
                    if max_val - min_val != 0:
                        group[col] = (group[col] - min_val) / (max_val - min_val)
                        group_scaling[col] = {'min': min_val, 'max': max_val}

                elif method == "z-score":
                    mean = group[col].mean()
                    std = group[col].std()
                    if std != 0:
                        group[col] = (group[col] - mean) / std
                        group_scaling[col] = {'mean': mean, 'std': std}

                elif method == "robust":
                    q75, q25 = group[col].quantile([0.75, 0.25])
                    iqr = q75 - q25
                    if iqr != 0:
                        group[col] = (group[col] - q25) / iqr
                        group_scaling[col] = {'q25': q25, 'iqr': iqr}

            scaling_factors[ticker] = group_scaling
            return group

        if groupby:
            df = df.groupby(groupby, group_keys=False).apply(normalize_group)
        else:
            df = normalize_group(df)

        def denormalize_group(group):
            ticker = group['Symbol'].iloc[0] if 'Symbol' in group else 'global'
            group_scaling = scaling_factors[ticker]

            for col in self.numeric_columns:
                if col not in group.columns or col not in group_scaling:
                    continue

                if method == "min-max":
                    min_val = group_scaling[col]['min']
                    max_val = group_scaling[col]['max']
                    group[col] = group[col] * (max_val - min_val) + min_val

                elif method == "z-score":
                    mean = group_scaling[col]['mean']
                    std = group_scaling[col]['std']
                    group[col] = group[col] * std + mean

                elif method == "robust":
                    q25 = group_scaling[col]['q25']
                    iqr = group_scaling[col]['iqr']
                    group[col] = group[col] * iqr + q25

            return group

        if groupby:
            df = df.groupby(groupby, group_keys=False).apply(denormalize_group)
        else:
            df = denormalize_group(df)

        return df

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
        logger.info(f"Winsorizing outliers using {method} method")
        df = df.copy()

        def winsorize_group(group):
            for col in self.numeric_columns:
                if col not in group.columns:
                    continue

                if method == "zscore":
                    z_scores = zscore(group[col], nan_policy='omit')
                    upper_mask = z_scores > threshold
                    lower_mask = z_scores < -threshold

                    if upper_mask.any() or lower_mask.any():
                        upper_bound = group[col][~upper_mask].max()
                        lower_bound = group[col][~lower_mask].min()

                        n_outliers = (upper_mask | lower_mask).sum()

                        group.loc[upper_mask, col] = upper_bound
                        group.loc[lower_mask, col] = lower_bound

                        logger.info(f"Winsorized {n_outliers} outliers in {col}")

                elif method == "iqr":
                    q75, q25 = group[col].quantile([0.75, 0.25])
                    iqr = q75 - q25
                    lower_bound = q25 - threshold * iqr
                    upper_bound = q75 + threshold * iqr

                    upper_mask = group[col] > upper_bound
                    lower_mask = group[col] < lower_bound

                    if upper_mask.any() or lower_mask.any():
                        n_outliers = (upper_mask | lower_mask).sum()

                        group.loc[upper_mask, col] = upper_bound
                        group.loc[lower_mask, col] = lower_bound

                        logger.info(f"Winsorized {n_outliers} outliers in {col}")

            return group

        if groupby:
            return df.groupby(groupby, group_keys=False).apply(winsorize_group)
        return winsorize_group(df)