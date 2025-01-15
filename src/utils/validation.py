"""
Validation Module

Checks data integrity (missing values, data types, outliers).
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from config.logging_config import logger


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in each column of 'df'.
    """
    logger.info("Checking for missing values.")
    missing = df.isnull().sum()
    logger.debug(f"Missing values per column:\n{missing}")
    return missing


def check_data_types(df: pd.DataFrame) -> pd.Series:
    """
    Check the data types of each column in 'df'.
    """
    logger.info("Checking data types of dataframe columns.")
    dtypes = df.dtypes
    logger.debug(f"Data types:\n{dtypes}")
    return dtypes


def check_outliers(df: pd.DataFrame, method: str = 'IQR', factor: float = 1.5) -> pd.DataFrame:
    """
    Identify outliers using IQR or Z-score method.

    Returns a boolean DataFrame: True => outlier.
    """
    logger.info(f"Checking for outliers using method={method}, factor={factor}.")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("No numeric columns => skipping outlier detection.")
        return pd.DataFrame(False, index=df.index, columns=df.columns)

    if method.lower() == 'iqr':
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        mask_lower = numeric_df < (Q1 - factor * IQR)
        mask_upper = numeric_df > (Q3 + factor * IQR)
        outliers = (mask_lower | mask_upper)
    elif method.lower() == 'z-score':
        from scipy.stats import zscore
        z = numeric_df.apply(zscore)
        outliers = z.abs() > factor
    else:
        logger.error(f"Outlier detection method '{method}' not supported.")
        raise ValueError(f"Method '{method}' not supported.")

    merged_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in numeric_df.columns:
        merged_mask[col] = outliers[col]
    logger.debug(f"Outliers detected:\n{merged_mask.sum().sum()} total outlier points.")
    return merged_mask


def validate_dataframe(df: pd.DataFrame,
                       methods: List[str] = ['missing', 'dtype', 'outlier'],
                       outlier_method: str = 'IQR',
                       outlier_factor: float = 1.5) -> bool:
    """
    Run a series of validation checks on 'df'.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        methods (List[str]): Which validations to run. Options: ['missing', 'dtype', 'outlier'].
        outlier_method (str): 'IQR' or 'Z-score'.
        outlier_factor (float): Factor for outlier detection.

    Returns:
        bool: True if all checks pass, False if any check fails.
    """
    logger.info("Validating dataframe integrity.")
    valid = True

    if 'missing' in methods:
        missing = check_missing_values(df)
        if missing.sum() > 0:
            logger.warning(f"Dataframe has missing values:\n{missing[missing>0]}")
            valid = False

    if 'dtype' in methods:
        dtypes = check_data_types(df)
        non_numeric_cols = dtypes[~dtypes.apply(lambda x: np.issubdtype(x, np.number))]
        if not non_numeric_cols.empty:
            logger.warning(f"Non-numeric columns detected:\n{non_numeric_cols}")
            valid = False

    if 'outlier' in methods:
        outliers_mask = check_outliers(df, method=outlier_method, factor=outlier_factor)
        if outliers_mask.any().any():
            logger.warning(f"Dataframe contains outliers using {outlier_method} method.")
            valid = False

    if valid:
        logger.info("Dataframe validation passed all checks.")
    else:
        logger.warning("Dataframe validation failed one or more checks.")

    return valid
