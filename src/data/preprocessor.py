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

    def clean_data(self, df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
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
        Normalize numeric columns using specified method.

        Args:
            df: Input DataFrame
            method: Normalization method ('min-max', 'z-score', or 'robust')
            groupby: Column to group by for normalization (None for global)

        Returns:
            Normalized DataFrame
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Method must be one of {self.VALID_METHODS}")

        logger.info(f"Normalizing data using {method} method")
        df = df.copy()

        def normalize_group(group):
            for col in self.numeric_columns:
                if col not in group.columns:
                    continue

                if method == "min-max":
                    min_val = group[col].min()
                    max_val = group[col].max()
                    if max_val - min_val != 0:
                        group[col] = (group[col] - min_val) / (max_val - min_val)
                elif method == "z-score":
                    std = group[col].std()
                    if std != 0:
                        group[col] = (group[col] - group[col].mean()) / std
                elif method == "robust":
                    q75, q25 = group[col].quantile([0.75, 0.25])
                    iqr = q75 - q25
                    if iqr != 0:
                        group[col] = (group[col] - q25) / iqr

            return group

        if groupby:
            return df.groupby(groupby, group_keys=False).apply(normalize_group)
        return normalize_group(df)

    def handle_outliers(self,
                        df: pd.DataFrame,
                        method: str = "zscore",
                        threshold: float = 3.0,
                        groupby: Optional[str] = "Symbol") -> pd.DataFrame:
        """
        Handle outliers using specified method.

        Args:
            df: Input DataFrame
            method: Method to detect outliers ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            groupby: Column to group by for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers using {method} method")
        df = df.copy()

        def handle_group(group):
            mask = pd.Series(True, index=group.index)

            for col in self.numeric_columns:
                if col not in group.columns:
                    continue

                if method == "zscore":
                    z_scores = zscore(group[col], nan_policy='omit')
                    mask &= abs(z_scores) < threshold
                elif method == "iqr":
                    q75, q25 = group[col].quantile([0.75, 0.25])
                    iqr = q75 - q25
                    lower = q25 - threshold * iqr
                    upper = q75 + threshold * iqr
                    mask &= (group[col] >= lower) & (group[col] <= upper)

            removed = (~mask).sum()
            if removed > 0:
                logger.info(f"Removed {removed} outliers")

            return group[mask]

        if groupby:
            return df.groupby(groupby, group_keys=False).apply(handle_group)
        return handle_group(df)