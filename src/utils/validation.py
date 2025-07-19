import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from config.logging_config import logger  # Assuming this is configured


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in each column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A Series showing the count of missing values per column.
                   Includes percentage of missing values in the debug log.
    """
    logger.info("Checking for missing values in the DataFrame.")
    missing_counts = df.isnull().sum()
    total_rows = len(df)

    if missing_counts.sum() > 0:
        missing_report = missing_counts[missing_counts > 0]
        missing_percentage = (missing_report / total_rows * 100).round(2)

        logger.warning(f"Missing values detected in the following columns:\n{missing_report}\n"
                       f"Percentage missing:\n{missing_percentage}%")
    else:
        logger.debug("No missing values found in the DataFrame.")

    return missing_counts


def check_data_types(df: pd.DataFrame) -> pd.Series:
    """
    Check the data types of each column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A Series showing the data type of each column.
    """
    logger.info("Checking data types of DataFrame columns.")
    dtypes = df.dtypes
    logger.debug(f"Data types:\n{dtypes}")
    return dtypes


def check_outliers(df: pd.DataFrame, method: str = 'IQR', factor: float = 1.5) -> pd.DataFrame:
    """
    Identify outliers in numeric columns using IQR or Z-score method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The method to use for outlier detection ('IQR' or 'Z-score'). Case-insensitive.
        factor (float): Multiplier for IQR (e.g., 1.5) or threshold for Z-score (e.g., 3.0).

    Returns:
        pd.DataFrame: A boolean DataFrame of the same shape as `df` indicating True for outliers.
                      Non-numeric columns will have False.
    """
    logger.info(f"Checking for outliers using method='{method}', factor={factor}.")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("No numeric columns found in the DataFrame. Skipping outlier detection.")
        return pd.DataFrame(False, index=df.index, columns=df.columns)

    outliers_mask = pd.DataFrame(False, index=numeric_df.index, columns=numeric_df.columns)

    if method.lower() == 'iqr':
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Handle cases where IQR is zero (e.g., constant column)
        for col in numeric_df.columns:
            if IQR[col] == 0:
                outliers_mask[col] = False  # No outliers if IQR is zero
                logger.debug(f"IQR for column '{col}' is zero. No outliers detected for this column.")
            else:
                mask_lower = numeric_df[col] < (Q1[col] - factor * IQR[col])
                mask_upper = numeric_df[col] > (Q3[col] + factor * IQR[col])
                outliers_mask[col] = (mask_lower | mask_upper)

    elif method.lower() == 'z-score':
        from scipy.stats import zscore
        for col in numeric_df.columns:
            # Handle cases where std dev is zero (e.g., constant column)
            if numeric_df[col].std() == 0:
                outliers_mask[col] = False  # No outliers if std dev is zero
                logger.debug(f"Standard deviation for column '{col}' is zero. No outliers detected for this column.")
            else:
                z = np.abs(zscore(numeric_df[col].dropna()))  # Z-score on non-NaN values
                outliers_mask.loc[numeric_df[col].dropna().index, col] = z > factor
    else:
        raise ValueError(f"Method '{method}' not supported. Choose 'IQR' or 'Z-score'.")

    # Create a full mask with False for non-numeric columns and merge numeric outliers
    final_outliers_df = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in outliers_mask.columns:
        final_outliers_df[col] = outliers_mask[col]

    total_outlier_points = final_outliers_df.sum().sum()
    if total_outlier_points > 0:
        logger.warning(f"Outliers detected: {total_outlier_points} total outlier points.")
        # Optionally, log which columns have outliers
        # logger.debug(f"Outliers by column:\n{final_outliers_df.sum()[final_outliers_df.sum() > 0]}")
    else:
        logger.debug("No outliers detected in numeric columns.")

    return final_outliers_df


def validate_dataframe(df: pd.DataFrame,
                       methods: List[str] = ['missing', 'dtype', 'outlier'],
                       outlier_method: str = 'IQR',
                       outlier_factor: float = 1.5) -> bool:
    """
    Run a series of validation checks on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        methods (List[str]): A list of checks to perform ('missing', 'dtype', 'outlier').
        outlier_method (str): Method for outlier detection ('IQR' or 'Z-score').
        outlier_factor (float): Factor/threshold for outlier detection.

    Returns:
        bool: True if the DataFrame passes all specified checks, False otherwise.
    """
    logger.info("Starting DataFrame integrity validation.")
    valid = True

    if df.empty:
        logger.error("Input DataFrame is empty. Validation failed.")
        return False

    if 'missing' in methods:
        missing = check_missing_values(df)
        if missing.sum() > 0:
            valid = False

    if 'dtype' in methods:
        dtypes = check_data_types(df)
        # Assuming typical OHLCV data for expected numeric columns.
        # This list could be made configurable if needed in the future.
        expected_numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in expected_numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Data type check failed: Column '{col}' is not numeric (found: {df[col].dtype}).")
                valid = False
            elif col in df.columns and pd.api.types.is_object_dtype(df[col]) and df[col].apply(
                    lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit()).any():
                logger.warning(
                    f"Data type check failed: Column '{col}' is object type and contains non-numeric strings.")
                valid = False

    if 'outlier' in methods:
        outliers_mask = check_outliers(df, method=outlier_method, factor=outlier_factor)
        if outliers_mask.any().any():
            valid = False

    if valid:
        logger.info("DataFrame validation passed all selected checks.")
    else:
        logger.warning("DataFrame validation failed one or more checks. Review warnings above.")

    return valid