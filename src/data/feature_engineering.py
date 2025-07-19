import pandas as pd
import numpy as np
from typing import Optional, List, Set
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming logging_config.py exists and configures a logger
try:
    from config.logging_config import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Added basicConfig for standalone use

# Import technical indicator functions from the 'ta' library
import ta.volatility
import ta.momentum
import ta.trend
import ta.volume # Explicitly import volume for clarity


class FeatureEngineer:
    """Class for generating technical indicators with proper NaN handling."""

    def __init__(self,
                 min_periods: Optional[int] = None,
                 fill_method: str = 'backfill',
                 validate: bool = True):
        """
        Initialize the feature engineer.

        Args:
            min_periods (Optional[int]): Minimum periods for rolling calculations.
                                         If None, uses the window size specified for each indicator.
            fill_method (str): How to handle NaN values after calculation ('drop', 'backfill', or None).
                               'drop': Drops rows containing NaNs in generated features.
                               'backfill': Fills NaNs in generated features backwards.
                               None: Leaves NaNs as they are.
            validate (bool): Whether to validate inputs (e.g., empty DataFrame, missing columns).
        """
        self._min_periods_override = min_periods # Store the override value
        self.fill_method = fill_method
        self.validate = validate
        logger.info(f"FeatureEngineer initialized with min_periods={min_periods}, fill_method='{fill_method}', validate={validate}.")

    @property
    def _min_periods_actual(self) -> Optional[int]:
        """
        Internal property to get the minimum periods override.
        Individual indicator methods will use this or their own `window` if this is None.
        """
        return self._min_periods_override

    def _validate_data(self, df: pd.DataFrame,
                       required_columns: List[str]) -> None:
        """
        Validate input DataFrame for common issues before feature generation.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_columns (List[str]): A list of column names that must be present.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        missing_cols = [col for col in required_columns
                        if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

        if df.isnull().any().any():
            logger.warning("Input DataFrame contains NaN values. This might affect indicator calculations.")

    def _handle_nans(self, df: pd.DataFrame,
                     feature_columns: List[str]) -> pd.DataFrame:
        """
        Handle NaN values in the specified feature columns according to the specified method.

        Args:
            df (pd.DataFrame): The DataFrame containing features.
            feature_columns (List[str]): List of columns on which to apply NaN handling.

        Returns:
            pd.DataFrame: DataFrame with NaNs handled.
        """
        if not feature_columns:
            logger.debug("No feature columns specified for NaN handling. Returning DataFrame as is.")
            return df

        # Ensure columns exist before trying to access them
        existing_feature_columns = [col for col in feature_columns if col in df.columns]
        if not existing_feature_columns:
            logger.debug(f"Specified feature columns {feature_columns} not found in DataFrame. No NaN handling applied.")
            return df

        if self.fill_method == 'drop':
            initial_rows = len(df)
            df = df.dropna(subset=existing_feature_columns)
            if len(df) < initial_rows:
                logger.info(f"Dropped {initial_rows - len(df)} rows due to NaNs in features using 'drop' method.")
        elif self.fill_method == 'backfill':
            # Apply backfill only to the feature columns.
            # .bfill() operates in-place if no assignment, but assignment is clearer.
            for col in existing_feature_columns:
                df[col] = df[col].bfill()
            logger.debug(f"Backfilled NaNs in columns: {existing_feature_columns}.")
        elif self.fill_method is None:
            logger.debug("NaN values are not handled (fill_method is None).")
            pass # Do nothing, leave NaNs
        else:
            raise ValueError(f"Unknown fill method: '{self.fill_method}'. Must be 'drop', 'backfill', or None.")

        return df

    def add_moving_average(self,
                           df: pd.DataFrame,
                           window: int = 20,
                           column: str = "Adj_Close",
                           ma_type: str = "simple") -> pd.DataFrame:
        """
        Add a moving average with configurable type.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): Window size for the moving average calculation.
            column (str): The column name to use for calculation (e.g., "Adj_Close").
            ma_type (str): Type of MA ('simple', 'weighted', or 'exp').

        Returns:
            pd.DataFrame: DataFrame with the new moving average column.
        """
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()
        ma_col = f"{ma_type.upper()}_MA_{window}"
        # Use the overridden min_periods or default to window
        min_periods_actual = self._min_periods_actual if self._min_periods_actual is not None else window

        if ma_type == 'simple':
            df[ma_col] = df[column].rolling(
                window=window,
                min_periods=min_periods_actual
            ).mean()
        elif ma_type == 'weighted':
            # Weighted Moving Average (WMA)
            # This implementation assumes linearly increasing weights.
            # The 'raw=True' in apply passes a NumPy array to the lambda, which is faster.
            df[ma_col] = df[column].rolling(
                window=window,
                min_periods=min_periods_actual
            ).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.arange(1, len(x) + 1).sum(),
                raw=True
            )
        elif ma_type == 'exp':
            df[ma_col] = df[column].ewm(
                span=window,
                min_periods=min_periods_actual,
                adjust=False # Standard EWM calculation
            ).mean()
        else:
            raise ValueError(f"Unknown MA type: '{ma_type}'. Must be 'simple', 'weighted', or 'exp'.")

        logger.debug(f"Added {ma_col} to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_rsi(self,
                df: pd.DataFrame,
                window: int = 14,
                column: str = "Adj_Close",
                method: str = "wilder") -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) with configurable calculation method.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): Window size for RSI calculation.
            column (str): The column name to use for calculation (e.g., "Adj_Close").
            method (str): Calculation method ('wilder' for Wilder's smoothing, 'cutler' for simple average).
                          Note: 'ta' library's `rsi` function primarily implements Wilder's smoothing.
        Returns:
            pd.DataFrame: DataFrame with the new RSI column.
        """
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()

        # 'ta' library's rsi is based on Wilder's smoothing.
        # min_periods is handled internally by ta.
        if method == 'wilder':
            df["RSI"] = ta.momentum.rsi(close=df[column], window=window, fillna=False)
        elif method == 'cutler':
            # For a true Cutler's RSI (Simple Moving Average of gains/losses),
            # a custom implementation would be needed. Sticking to 'ta' for simplicity
            # and consistency with other 'ta' library indicators.
            logger.warning("RSI 'cutler' method is not directly supported by 'ta' library's rsi function. Using 'wilder' calculation instead.")
            df["RSI"] = ta.momentum.rsi(close=df[column], window=window, fillna=False)
        else:
            raise ValueError(f"Unknown RSI method: '{method}'. Must be 'wilder' or 'cutler'.")

        logger.debug(f"Added RSI (window={window}, method='{method}') to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_macd(self,
                 df: pd.DataFrame,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 column: str = "Adj_Close") -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence) indicator and its components.

        Args:
            df (pd.DataFrame): Input DataFrame.
            fast_period (int): Window size for the fast EMA.
            slow_period (int): Window size for the slow EMA.
            signal_period (int): Window size for the signal line EMA of the MACD.
            column (str): The column name to use for calculation (e.g., "Adj_Close").

        Returns:
            pd.DataFrame: DataFrame with 'MACD', 'Signal_Line', and 'MACD_Histogram' columns.
        """
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()

        macd_indicator = ta.trend.MACD(
            close=df[column],
            window_fast=fast_period,
            window_slow=slow_period,
            window_sign=signal_period,
            fillna=False # We handle NaNs later
        )

        df["MACD"] = macd_indicator.macd()
        df["Signal_Line"] = macd_indicator.macd_signal()
        df["MACD_Histogram"] = macd_indicator.macd_diff()

        logger.debug(f"Added MACD (fast={fast_period}, slow={slow_period}, signal={signal_period}) to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_bollinger_bands(self,
                            df: pd.DataFrame,
                            window: int = 20,
                            num_std: float = 2.0,
                            column: str = "Adj_Close") -> pd.DataFrame:
        """
        Add Bollinger Bands (Middle, Upper, Lower, Bandwidth, %B).

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): Window size for the Bollinger Bands calculation (moving average and standard deviation).
            num_std (float): Number of standard deviations for the upper and lower bands.
            column (str): The column name to use for calculation (e.g., "Adj_Close").

        Returns:
            pd.DataFrame: DataFrame with 'BB_Middle', 'BB_Upper', 'BB_Lower',
                          'BB_Bandwidth', and '%B' columns.
        """
        if self.validate:
            self._validate_data(df, [column])

        df = df.copy()

        bb_indicator = ta.volatility.BollingerBands(
            close=df[column],
            window=window,
            window_dev=num_std,
            fillna=False # We handle NaNs later
        )

        df["BB_Middle"] = bb_indicator.bollinger_mavg()
        df["BB_Upper"] = bb_indicator.bollinger_hband()
        df["BB_Lower"] = bb_indicator.bollinger_lband()
        df["BB_Bandwidth"] = bb_indicator.bollinger_wband()
        df["%B"] = bb_indicator.bollinger_pband()

        logger.debug(f"Added Bollinger Bands (window={window}, std={num_std}) to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_volume_indicators(self,
                              df: pd.DataFrame,
                              window: int = 20) -> pd.DataFrame:
        """
        Add volume-based indicators: Volume Simple Moving Average (Volume_SMA),
        On-Balance Volume (OBV), and Volume Price Trend (VPT).

        Args:
            df (pd.DataFrame): Input DataFrame with 'Adj_Close' and 'Volume' columns.
            window (int): Window size for Volume_SMA.

        Returns:
            pd.DataFrame: DataFrame with 'Volume_SMA', 'OBV', and 'VPT' columns.
        """
        if self.validate:
            self._validate_data(df, ["Adj_Close", "Volume"])

        df = df.copy()
        # Use the overridden min_periods or default to window for Volume_SMA
        min_periods_actual = self._min_periods_actual if self._min_periods_actual is not None else window

        # SMA for Volume
        df["Volume_SMA"] = df["Volume"].rolling(
            window=window,
            min_periods=min_periods_actual
        ).mean()

        # OBV from ta library
        df["OBV"] = ta.volume.on_balance_volume(
            close=df["Adj_Close"],
            volume=df["Volume"],
            fillna=False
        )

        # VPT from ta library
        df["VPT"] = ta.volume.volume_price_trend(
            close=df["Adj_Close"],
            volume=df["Volume"],
            fillna=False
        )

        logger.debug(f"Added volume indicators (Volume_SMA window={window}) to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_atr(self,
                df: pd.DataFrame,
                window: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR).

        Args:
            df (pd.DataFrame): Input DataFrame with 'High', 'Low', 'Adj_Close' columns.
            window (int): Window size for ATR calculation.

        Returns:
            pd.DataFrame: DataFrame with the 'ATR' column.
        """
        if self.validate:
            self._validate_data(df, ["High", "Low", "Adj_Close"])

        df = df.copy()

        df["ATR"] = ta.volatility.average_true_range(
            high=df["High"],
            low=df["Low"],
            close=df["Adj_Close"],
            window=window,
            fillna=False
        )
        logger.debug(f"Added ATR (window={window}) to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_cci(self,
                df: pd.DataFrame,
                window: int = 20) -> pd.DataFrame:
        """
        Add Commodity Channel Index (CCI).

        Args:
            df (pd.DataFrame): Input DataFrame with 'High', 'Low', 'Adj_Close' columns.
            window (int): Window size for CCI calculation.

        Returns:
            pd.DataFrame: DataFrame with the 'CCI' column.
        """
        if self.validate:
            self._validate_data(df, ["High", "Low", "Adj_Close"])

        df = df.copy()

        df["CCI"] = ta.trend.cci(
            high=df["High"],
            low=df["Low"],
            close=df["Adj_Close"],
            window=window,
            fillna=False
        )
        logger.debug(f"Added CCI (window={window}) to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_stochastic_oscillator(self,
                                  df: pd.DataFrame,
                                  window: int = 14,
                                  smooth_window: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator (%K and %D).

        Args:
            df (pd.DataFrame): Input DataFrame with 'High', 'Low', 'Adj_Close' columns.
            window (int): Window size for %K calculation.
            smooth_window (int): Window size for %D (signal line) calculation.

        Returns:
            pd.DataFrame: DataFrame with '%K' and '%D' columns.
        """
        if self.validate:
            self._validate_data(df, ["High", "Low", "Adj_Close"])

        df = df.copy()

        stoch_indicator = ta.momentum.StochasticOscillator(
            high=df["High"],
            low=df["Low"],
            close=df["Adj_Close"],
            window=window,
            smooth_window=smooth_window,
            fillna=False
        )

        df["%K"] = stoch_indicator.stoch()
        df["%D"] = stoch_indicator.stoch_signal()

        logger.debug(f"Added Stochastic Oscillator (window={window}, smooth={smooth_window}) to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_adx(self,
                df: pd.DataFrame,
                window: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) and its components (+DI, -DI).

        Args:
            df (pd.DataFrame): Input DataFrame with 'High', 'Low', 'Adj_Close' columns.
            window (int): Window size for ADX calculation.

        Returns:
            pd.DataFrame: DataFrame with 'ADX', '+DI', and '-DI' columns.
        """
        if self.validate:
            self._validate_data(df, ["High", "Low", "Adj_Close"])

        df = df.copy()

        adx_indicator = ta.trend.ADXIndicator(
            high=df["High"],
            low=df["Low"],
            close=df["Adj_Close"],
            window=window,
            fillna=False
        )

        df["ADX"] = adx_indicator.adx()
        df["+DI"] = adx_indicator.adx_pos()
        df["-DI"] = adx_indicator.adx_neg()

        logger.debug(f"Added ADX (window={window}) to DataFrame.")
        return df # NaN handling will be done in generate_features

    def add_pca_factors(self,
                        df: pd.DataFrame,
                        n_components: Optional[int] = None,
                        variance_threshold: Optional[float] = None,
                        column: str = "Adj_Close") -> pd.DataFrame:
        """
        Add Principal Component Analysis (PCA) derived factors.

        This function performs PCA on the *returns* of multiple assets
        to extract common underlying factors. Factors are derived from
        the pivoted DataFrame (Date x Symbol returns) and then merged
        back to the original long format.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Date', 'Symbol', and the specified `column`.
                Must contain data for multiple symbols for PCA to be meaningful.
            n_components (Optional[int]): Number of principal components to extract. If None,
                                          `variance_threshold` must be specified.
            variance_threshold (Optional[float]): If specified, PCA will select the minimum number
                                                  of components that explain at least this much
                                                  variance (e.g., 0.95 for 95%). If None,
                                                  `n_components` must be specified.
            column (str): The column to calculate returns from for PCA (e.g., "Adj_Close").

        Returns:
            pd.DataFrame: DataFrame with added PCA factors. Factors are named 'PCA_Factor_0', 'PCA_Factor_1', etc.
                          These factors are common across all symbols for a given date.
        """
        if self.validate:
            self._validate_data(df, ['Date', 'Symbol', column])
            if n_components is None and variance_threshold is None:
                raise ValueError("Either 'n_components' or 'variance_threshold' must be specified for PCA.")
            if n_components is not None and n_components < 1:
                raise ValueError("'n_components' must be a positive integer.")
            if variance_threshold is not None and not (0 < variance_threshold <= 1):
                raise ValueError("'variance_threshold' must be between 0 and 1.")
            if df['Symbol'].nunique() < 2:
                logger.warning("PCA requires at least two unique symbols to be meaningful. Skipping PCA.")
                return df # Return original DataFrame if only one symbol

        df_copy_for_pca = df.copy() # Operate on a copy to avoid side effects

        # Pivot the data to have symbols as columns and dates as index
        df_pivot = df_copy_for_pca.pivot(index='Date', columns='Symbol', values=column)

        # Calculate daily returns
        returns = df_pivot.pct_change()

        # Handle NaNs from pct_change (first row) and potential missing data
        initial_returns_rows = len(returns)
        returns = returns.dropna(how='any') # Drop rows with any NaN
        if returns.empty:
            logger.warning("Not enough valid data to calculate returns for PCA after dropping NaNs. Skipping PCA.")
            return df # Return original if no valid returns
        if len(returns) < initial_returns_rows:
            logger.info(f"Dropped {initial_returns_rows - len(returns)} rows from returns for PCA due to NaNs.")

        if returns.shape[1] < 2:
            logger.warning(f"After cleaning, only {returns.shape[1]} symbols remain for PCA. PCA requires at least 2 for meaningful decomposition. Skipping PCA.")
            return df # Return original if only one symbol left after cleaning

        # Standardize returns: crucial for PCA
        scaler = StandardScaler()
        # Use .loc to ensure alignment if original returns had NaNs that were dropped
        returns_scaled = scaler.fit_transform(returns)
        returns_scaled_df = pd.DataFrame(returns_scaled, index=returns.index, columns=returns.columns)

        # Apply PCA
        if variance_threshold is not None:
            pca = PCA(n_components=variance_threshold)
            logger.info(f"Performing PCA to explain {variance_threshold*100:.2f}% of variance.")
        else:
            pca = PCA(n_components=n_components)
            logger.info(f"Performing PCA to extract {n_components} components.")

        try:
            pca_components = pca.fit_transform(returns_scaled_df)
        except ValueError as e:
            logger.error(f"PCA fit_transform failed: {e}. This might happen if all returns are identical or constant for a symbol.")
            return df # Return original DataFrame if PCA fails

        # Create DataFrame for PCA factors
        num_actual_components = pca_components.shape[1]
        pca_factor_cols = [f"PCA_Factor_{i}" for i in range(num_actual_components)]
        pca_factors_df = pd.DataFrame(pca_components, index=returns.index, columns=pca_factor_cols)

        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        logger.info(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_)}")

        # Merge PCA factors back to the original DataFrame structure (long format)
        # This requires merging on 'Date'.
        # Set 'Date' as index temporarily for merge, then reset.
        df_copy_for_pca = df_copy_for_pca.set_index('Date')
        df_with_pca = df_copy_for_pca.merge(pca_factors_df, left_index=True, right_index=True, how='left')
        df_with_pca = df_with_pca.reset_index() # Bring 'Date' back as a column

        logger.debug(f"Added {num_actual_components} PCA factors to DataFrame.")
        # NaN handling for PCA factors will be done in generate_features
        return df_with_pca


    def generate_features(self,
                          df: pd.DataFrame,
                          features: Optional[List[str]] = None,
                          pca_n_components: Optional[int] = None,
                          pca_variance_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Generate selected technical indicators and factor decomposition for a DataFrame
        that may contain multiple symbols.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'Date', 'Symbol', and OHLCV data.
            features (Optional[List[str]]): List of feature names to generate. If None, all basic
                                            technical indicators will be generated.
                                            Accepted values: 'sma', 'ema', 'wma', 'rsi', 'macd',
                                            'bbands', 'atr', 'cci', 'stoch', 'adx', 'volume_indicators', 'pca'.
            pca_n_components (Optional[int]): Number of principal components for 'pca' factor decomposition.
                                               Used only if 'pca' is in `features` or `features` is None.
            pca_variance_threshold (Optional[float]): Variance threshold for 'pca' factor decomposition.
                                                      Used only if 'pca' in `features` or `features` is None,
                                                      and `pca_n_components` is None.

        Returns:
            pd.DataFrame: DataFrame with added technical indicators and PCA factors.
                          The DataFrame will be sorted by 'Date' and 'Symbol'.
        """
        logger.info("Starting feature generation.")

        # Ensure essential columns are present for comprehensive feature generation
        required_cols_for_all = ["Date", "Symbol", "Adj_Close", "High", "Low", "Volume"]
        if self.validate:
            self._validate_data(df, required_cols_for_all)

        df_processed = df.copy()
        generated_feature_cols: Set[str] = set() # To track all new feature columns added

        # Determine which features to generate
        if features is None:
            features_to_generate = {
                'sma', 'ema', 'wma', 'rsi', 'macd', 'bbands',
                'atr', 'cci', 'stoch', 'adx', 'volume_indicators', 'pca'
            }
        else:
            features_to_generate = set(features)
            # Validate requested features
            valid_features = {
                'sma', 'ema', 'wma', 'rsi', 'macd', 'bbands',
                'atr', 'cci', 'stoch', 'adx', 'volume_indicators', 'pca'
            }
            invalid_features = [f for f in features_to_generate if f not in valid_features]
            if invalid_features:
                raise ValueError(f"Invalid feature names requested: {invalid_features}. "
                                 f"Accepted features are: {', '.join(sorted(list(valid_features)))}.")

        # Handle PCA separately as it's a cross-sectional calculation
        # It needs to be done *before* splitting by symbol, then merged back.
        pca_factors_generated = False
        pca_factor_col_names: List[str] = []

        if 'pca' in features_to_generate:
            if df_processed['Symbol'].nunique() > 1:
                try:
                    df_processed = self.add_pca_factors(
                        df_processed,
                        n_components=pca_n_components,
                        variance_threshold=pca_variance_threshold,
                        column="Adj_Close" # PCA generally works best on price returns
                    )
                    # Identify PCA factor column names for later NaN handling
                    # Assuming PCA factors are named 'PCA_Factor_X'
                    pca_factor_col_names = [col for col in df_processed.columns if col.startswith('PCA_Factor_')]
                    if pca_factor_col_names:
                        generated_feature_cols.update(pca_factor_col_names)
                        pca_factors_generated = True
                except ValueError as e:
                    logger.error(f"Error generating PCA factors: {e}. Skipping PCA factor generation.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during PCA factor generation: {e}. Skipping PCA.")
            else:
                logger.warning("Skipping PCA: Only one symbol found in the DataFrame. PCA requires multiple symbols.")
            features_to_generate.discard('pca') # PCA is handled, remove from list for symbol-wise processing

        # --- Process individual symbols ---
        # Instead of concatenating iteratively, collect results and concat once at the end.
        symbol_processed_dfs = []

        for symbol in df_processed['Symbol'].unique():
            logger.debug(f"Processing symbol: {symbol}")
            symbol_mask = df_processed['Symbol'] == symbol
            # Important: Ensure 'Date' is sorted for rolling window calculations
            symbol_data = df_processed[symbol_mask].sort_values('Date').copy()

            # Track columns before adding new features for this symbol
            initial_cols_for_symbol_set = set(symbol_data.columns)

            try:
                # Moving Averages
                if 'sma' in features_to_generate:
                    symbol_data = self.add_moving_average(symbol_data, ma_type='simple')
                if 'ema' in features_to_generate:
                    symbol_data = self.add_moving_average(symbol_data, ma_type='exp')
                if 'wma' in features_to_generate:
                    symbol_data = self.add_moving_average(symbol_data, ma_type='weighted')

                # Momentum and Oscillators
                if 'rsi' in features_to_generate:
                    symbol_data = self.add_rsi(symbol_data)
                if 'macd' in features_to_generate:
                    symbol_data = self.add_macd(symbol_data)
                if 'stoch' in features_to_generate:
                    symbol_data = self.add_stochastic_oscillator(symbol_data)
                if 'cci' in features_to_generate:
                    symbol_data = self.add_cci(symbol_data)

                # Volatility
                if 'bbands' in features_to_generate:
                    symbol_data = self.add_bollinger_bands(symbol_data)
                if 'atr' in features_to_generate:
                    symbol_data = self.add_atr(symbol_data)

                # Trend
                if 'adx' in features_to_generate:
                    symbol_data = self.add_adx(symbol_data)

                # Volume Indicators
                if 'volume_indicators' in features_to_generate:
                    symbol_data = self.add_volume_indicators(symbol_data)

                # Identify newly created feature columns for this symbol (excluding PCA if generated earlier)
                feature_cols_added_for_symbol = [
                    col for col in symbol_data.columns
                    if col not in initial_cols_for_symbol_set and not col.startswith('PCA_Factor_')
                ]

                # Rename feature columns to be symbol-specific (only for non-PCA features)
                rename_mapping = {}
                for col in feature_cols_added_for_symbol:
                    rename_mapping[col] = f"{symbol}_{col}"
                symbol_data = symbol_data.rename(columns=rename_mapping)

                # Update the set of all generated feature columns
                generated_feature_cols.update(rename_mapping.values())

                symbol_processed_dfs.append(symbol_data)

            except Exception as e:
                logger.error(f"Error generating features for {symbol}: {e}. Skipping this symbol.")
                continue

        if not symbol_processed_dfs:
            logger.warning("No symbols were processed successfully. Returning original DataFrame.")
            return df.sort_values(['Date', 'Symbol']).reset_index(drop=True)


        # Concatenate all processed symbol DataFrames
        final_df = pd.concat(symbol_processed_dfs, ignore_index=True)

        # Apply final NaN handling to all generated feature columns (including PCA)
        if generated_feature_cols:
            final_df = self._handle_nans(final_df, list(generated_feature_cols))
        else:
            logger.info("No new features were generated, skipping final NaN handling for new features.")

        final_df = final_df.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        logger.info("Feature generation complete.")
        return final_df


def main():
    """Example usage."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)

    # Create dummy data with required columns for TA library functions
    # Make data for multiple symbols properly interleaved by date for PCA
    all_dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
    symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

    realistic_data_list = []
    for date in all_dates:
        for symbol in symbols:
            # Create a base row with random values
            # Using cumulative sum to create more realistic trends
            adj_close_base = 100 + 0.1 * np.random.randn() # Base for current date
            if realistic_data_list:
                # Find previous adj_close for this symbol to make it somewhat continuous
                prev_adj_close = next((item['Adj_Close'] for item in reversed(realistic_data_list) if item['Symbol'] == symbol), adj_close_base)
                adj_close = prev_adj_close * (1 + np.random.normal(0, 0.005))
            else:
                adj_close = adj_close_base

            high = adj_close * (1 + np.abs(np.random.normal(0, 0.005)))
            low = adj_close * (1 - np.abs(np.random.normal(0, 0.005)))
            volume = np.random.randint(100000, 5000000)

            # Ensure price logic (Low <= Adj_Close <= High)
            if high < low: # Should not happen with above logic but defensive
                high, low = low, high
            if adj_close < low:
                adj_close = low
            if adj_close > high:
                adj_close = high

            realistic_data_list.append({
                'Date': date,
                'Symbol': symbol,
                'Adj_Close': adj_close,
                'High': high,
                'Low': low,
                'Volume': volume
            })
    realistic_data = pd.DataFrame(realistic_data_list)
    realistic_data['Date'] = pd.to_datetime(realistic_data['Date'])
    realistic_data = realistic_data.sort_values(by=['Date', 'Symbol']).reset_index(drop=True)


    engineer = FeatureEngineer(
        min_periods=10,
        fill_method='backfill',
        validate=True
    )

    print("\n--- Generating All Features with 2 PCA Factors ---")
    df_all_with_pca = engineer.generate_features(
        realistic_data.copy(),
        features=None, # None means all default features + pca if specified
        pca_n_components=2
    )
    # Check for NaNs only in the *generated* columns
    generated_cols_all = [col for col in df_all_with_pca.columns if col not in realistic_data.columns]
    print(f"Total rows: {len(df_all_with_pca)}. NaNs in generated columns:\n{df_all_with_pca[generated_cols_all].isnull().sum()}")
    print("DataFrame with all features including 2 PCA factors (head):\n", df_all_with_pca.head(25))
    print("DataFrame with all features including 2 PCA factors (tail):\n", df_all_with_pca.tail(15))


    print("\n--- Generating Selected Features with PCA (80% variance explained) ---")
    df_selected_with_pca = engineer.generate_features(
        realistic_data.copy(),
        features=['sma', 'rsi', 'bbands', 'pca'], # Explicitly ask for PCA
        pca_variance_threshold=0.80
    )
    generated_cols_selected = [col for col in df_selected_with_pca.columns if col not in realistic_data.columns]
    print(f"Total rows: {len(df_selected_with_pca)}. NaNs in generated columns:\n{df_selected_with_pca[generated_cols_selected].isnull().sum()}")
    print("DataFrame with selected features including PCA (80% variance explained) (head):\n", df_selected_with_pca.head(25))
    print("DataFrame with selected features including PCA (80% variance explained) (tail):\n", df_selected_with_pca.tail(15))

    print("\n--- Example with NaNs in input (with PCA and backfill) ---")
    data_with_nans = realistic_data.copy()
    data_with_nans.loc[100:110, 'Adj_Close'] = np.nan # Introduce NaNs for a block of data
    data_with_nans.loc[500:510, 'Volume'] = np.nan # Introduce NaNs for a block of data
    df_nans_handled = engineer.generate_features(data_with_nans, features=['rsi', 'macd', 'pca'], pca_n_components=1)
    generated_cols_nans = [col for col in df_nans_handled.columns if col not in realistic_data.columns]
    print(f"Total rows: {len(df_nans_handled)}. NaNs in generated columns:\n{df_nans_handled[generated_cols_nans].isnull().sum()}")
    print("DataFrame with NaNs in input, after feature engineering (with PCA and backfill) (head):\n", df_nans_handled.head(25))
    print("DataFrame with NaNs in input, after feature engineering (with PCA and backfill) (tail):\n", df_nans_handled.tail(15))

    print("\n--- Example with fill_method='drop' ---")
    engineer_drop = FeatureEngineer(fill_method='drop')
    df_nans_dropped = engineer_drop.generate_features(data_with_nans.copy(), features=['rsi', 'macd'], pca_n_components=1)
    print(f"Total rows: {len(df_nans_dropped)}. NaNs in generated columns:\n{df_nans_dropped[generated_cols_nans].isnull().sum()}")
    print("DataFrame with NaNs in input, after feature engineering (with 'drop' method) (head):\n", df_nans_dropped.head(25))
    print("DataFrame with NaNs in input, after feature engineering (with 'drop' method) (tail):\n", df_nans_dropped.tail(15))


if __name__ == "__main__":
    main()