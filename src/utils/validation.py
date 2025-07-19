import pandas as pd
import numpy as np
from typing import List
from config.logging_config import logger


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in each column of 'df'.
    """
    logger.info("Checking for missing values.")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.debug(f"Missing values per column:\n{missing[missing > 0]}")
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
        raise ValueError(f"Method '{method}' not supported.")

    merged_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in numeric_df.columns:
        merged_mask[col] = outliers[col]
    
    if merged_mask.sum().sum() > 0:
        logger.debug(f"Outliers detected:\n{merged_mask.sum().sum()} total outlier points.")
    return merged_mask


def validate_dataframe(df: pd.DataFrame,
                       methods: List[str] = ['missing', 'dtype', 'outlier'],
                       outlier_method: str = 'IQR',
                       outlier_factor: float = 1.5) -> bool:
    """
    Run a series of validation checks on 'df'.
    """
    logger.info("Validating dataframe integrity.")
    valid = True

    if 'missing' in methods:
        missing = check_missing_values(df)
        if missing.sum() > 0:
            logger.warning(f"Dataframe has missing values:\n{missing[missing>0]}")
            valid = False

    if 'dtype' in methods:
        expected_numeric = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in expected_numeric:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column '{col}' is not numeric. Found type: {df[col].dtype}")
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