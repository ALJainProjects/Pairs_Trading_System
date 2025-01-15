"""
Enhanced Technical Indicator Module

This module provides feature engineering for financial time series with:
1. Proper NaN handling strategies
2. Input validation
3. Configurable parameters
4. Multiple calculation methods
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from config.logging_config import logger


class FeatureEngineer:
    """Class for generating technical indicators with proper NaN handling."""

    def __init__(self,
                 min_periods: Optional[int] = None,
                 fill_method: str = 'backfill',
                 validate: bool = True):
        """
        Initialize the feature engineer.

        Args:
            min_periods: Minimum periods for rolling calculations
                      (None uses window size)
            fill_method: How to handle NaN values ('drop', 'backfill', or None)
            validate: Whether to validate inputs
        """
        self.min_periods = min_periods
        self.fill_method = fill_method
        self.validate = validate

    def _validate_data(self, df: pd.DataFrame,
                       required_columns: List[str]) -> None:
        """Validate input data."""
        if df.empty:
            raise ValueError("Empty DataFrame provided")

        missing_cols = [col for col in required_columns
                        if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.isnull().any().any():
            logger.warning("Input contains NaN values")

    def _handle_nans(self, df: pd.DataFrame,
                     feature_columns: List[str]) -> pd.DataFrame:
        """Handle NaN values according to specified method."""
        if self.fill_method == 'drop':
            df = df.dropna(subset=feature_columns)
        elif self.fill_method == 'backfill':
            df[feature_columns] = df[feature_columns].bfill()
        elif self.fill_method is not None:
            raise ValueError(f"Unknown fill method: {self.fill_method}")

        return df

    def add_moving_average(self,
                           df: pd.DataFrame,
                           window: int = 20,
                           column: str = "Close",
                           ma_type: str = "simple") -> pd.DataFrame:
        """
        Add moving average with configurable type.

        Args:
            df: Input DataFrame
            window: Window size
            column: Column to use
            ma_type: Type of MA ('simple', 'weighted', or 'exp')
        """
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()
        ma_col = f"{ma_type.upper()}_MA_{window}"
        min_periods = self.min_periods or window

        if ma_type == 'simple':
            df[ma_col] = df[column].rolling(
                window=window,
                min_periods=min_periods
            ).mean()
        elif ma_type == 'weighted':
            weights = np.arange(1, window + 1)
            df[ma_col] = df[column].rolling(
                window=window,
                min_periods=min_periods
            ).apply(lambda x: np.sum(weights[:len(x)] * x) / weights[:len(x)].sum(), raw=True)
        elif ma_type == 'exp':
            df[ma_col] = df[column].ewm(
                span=window,
                min_periods=min_periods,
                adjust=False
            ).mean()
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

        return self._handle_nans(df, [ma_col])

    def add_rsi(self,
                df: pd.DataFrame,
                window: int = 14,
                column: str = "Close",
                method: str = "wilder") -> pd.DataFrame:
        """
        Add RSI with configurable calculation method.

        Args:
            df: Input DataFrame
            window: Window size
            column: Column to use
            method: Calculation method ('wilder' or 'cutler')
        """
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()
        min_periods = self.min_periods or window

        delta = df[column].diff()

        if method == 'wilder':
            gain = delta.where(delta > 0, 0).ewm(
                alpha=1 / window,
                min_periods=min_periods,
                adjust=False
            ).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(
                alpha=1 / window,
                min_periods=min_periods,
                adjust=False
            ).mean()
        elif method == 'cutler':
            gain = delta.where(delta > 0, 0).rolling(
                window=window,
                min_periods=min_periods
            ).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(
                window=window,
                min_periods=min_periods
            ).mean()
        else:
            raise ValueError(f"Unknown RSI method: {method}")

        rs = gain / (loss + 1e-12)
        df["RSI"] = 100 - (100 / (1 + rs))

        return self._handle_nans(df, ["RSI"])

    def add_macd(self,
                 df: pd.DataFrame,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 column: str = "Close") -> pd.DataFrame:
        """Add MACD indicator."""
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()
        min_periods = self.min_periods or slow_period

        fast_ema = df[column].ewm(
            span=fast_period,
            min_periods=min_periods,
            adjust=False
        ).mean()
        slow_ema = df[column].ewm(
            span=slow_period,
            min_periods=min_periods,
            adjust=False
        ).mean()

        df["MACD"] = fast_ema - slow_ema
        df["Signal_Line"] = df["MACD"].ewm(
            span=signal_period,
            min_periods=self.min_periods or signal_period,
            adjust=False
        ).mean()
        df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

        return self._handle_nans(df, ["MACD", "Signal_Line", "MACD_Histogram"])

    def add_bollinger_bands(self,
                            df: pd.DataFrame,
                            window: int = 20,
                            num_std: float = 2.0,
                            column: str = "Close") -> pd.DataFrame:
        """Add Bollinger Bands."""
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()
        min_periods = self.min_periods or window

        sma = df[column].rolling(
            window=window,
            min_periods=min_periods
        ).mean()

        std = df[column].rolling(
            window=window,
            min_periods=min_periods
        ).std()

        df["BB_Middle"] = sma
        df["BB_Upper"] = sma + (std * num_std)
        df["BB_Lower"] = sma - (std * num_std)

        df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
        df["%B"] = (df[column] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

        return self._handle_nans(df, ["BB_Middle", "BB_Upper", "BB_Lower",
                                      "BB_Bandwidth", "%B"])

    def add_volume_indicators(self,
                              df: pd.DataFrame,
                              window: int = 20) -> pd.DataFrame:
        """Add volume-based indicators."""
        if self.validate:
            self._validate_data(df, ["Close", "Volume"])

        df = df.copy()
        min_periods = self.min_periods or window

        df["Volume_SMA"] = df["Volume"].rolling(
            window=window,
            min_periods=min_periods
        ).mean()

        df["OBV"] = (np.sign(df["Close"].diff()) *
                     df["Volume"]).fillna(0).cumsum()

        df["VPT"] = (df["Volume"] *
                     df["Close"].pct_change()).fillna(0).cumsum()

        return self._handle_nans(df, ["Volume_SMA", "OBV", "VPT"])

    def generate_features(self,
                          df: pd.DataFrame,
                          features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate selected technical indicators.

        Args:
            df: Input DataFrame
            features: List of feature_engineering to generate (None for all)
        """
        logger.info("Generating technical indicators")

        if self.validate:
            self._validate_data(df, ["Close"])

        df = df.copy()

        all_features = {
            'sma': lambda: self.add_moving_average(df, ma_type='simple'),
            'ema': lambda: self.add_moving_average(df, ma_type='exp'),
            'wma': lambda: self.add_moving_average(df, ma_type='weighted'),
            'rsi': lambda: self.add_rsi(df),
            'macd': lambda: self.add_macd(df),
            'bbands': lambda: self.add_bollinger_bands(df),
            'volume': lambda: self.add_volume_indicators(df)
        }

        features = features or list(all_features.keys())
        invalid_features = set(features) - set(all_features.keys())
        if invalid_features:
            raise ValueError(f"Unknown feature_engineering: {invalid_features}")

        for feature in features:
            try:
                df = all_features[feature]()
                logger.info(f"Generated {feature} feature_engineering")
            except Exception as e:
                logger.error(f"Error generating {feature}: {str(e)}")
                raise

        return df


def main():
    """Example usage."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)

    data = pd.DataFrame({
        'Close': np.random.randn(1000).cumsum(),
        'Volume': np.random.randint(1000, 10000, 1000)
    })

    engineer = FeatureEngineer(
        min_periods=10,
        fill_method='backfill',
        validate=True
    )

    df_all = engineer.generate_features(data)

    df_selected = engineer.generate_features(
        data,
        features=['sma', 'rsi', 'bbands']
    )

    return df_all, df_selected


if __name__ == "__main__":
    df_all, df_selected = main()
    print(df_all.head(100))
    print(df_selected.head(100))