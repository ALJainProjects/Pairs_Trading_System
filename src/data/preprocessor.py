import pandas as pd
import numpy as np
from config.logging_config import logger

class Preprocessor:
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data: dropping NaNs, duplicates, sorting by date.")
        if "date" not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column.")

        # Convert date if needed
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        df.sort_values(by="date", inplace=True)

        before_drop = len(df)
        df.dropna(inplace=True)
        after_drop = len(df)
        if before_drop - after_drop > 0:
            logger.warning(f"Dropped {before_drop - after_drop} rows due to NaNs.")

        before_dup = len(df)
        df.drop_duplicates(subset=["ticker", "date"], inplace=True)
        after_dup = len(df)
        if before_dup - after_dup > 0:
            logger.warning(f"Removed {before_dup - after_dup} duplicates.")

        df.reset_index(drop=True, inplace=True)
        return df

    def normalize_data(self, df: pd.DataFrame, method: str = "min-max") -> pd.DataFrame:
        logger.info(f"Normalizing data using {method} method.")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        if method == "min-max":
            for col in numeric_cols:
                min_val, max_val = df[col].min(), df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-12)
        elif method == "z-score":
            for col in numeric_cols:
                mean, std = df[col].mean(), df[col].std() + 1e-12
                df[col] = (df[col] - mean) / std
        else:
            raise ValueError(f"Unsupported normalization method {method}")
        return df

    def handle_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
        logger.info(f"Removing outliers with z_threshold={z_threshold}")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        from scipy.stats import zscore
        zscores = df[numeric_cols].apply(zscore).abs()
        filtered = df[(zscores < z_threshold).all(axis=1)]
        logger.debug(f"Filtered out {len(df) - len(filtered)} outlier rows.")
        return filtered
