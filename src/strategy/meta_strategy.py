import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Set

from .base import BaseStrategy
from .static_pairs_strategy import StaticPairsStrategy
from .advanced_ml_strategy import AdvancedMLStrategy
from config.logging_config import logger


class MarketRegimeDetector:
    """
    Detects market regimes based on volatility and trend using configurable parameters.
    Regimes are classified into: 'Trending_Volatile', 'Trending_Quiet', 'Ranging_Volatile', 'Ranging_Quiet'.
    """

    def __init__(self,
                 lookback_period: int = 60,  # For short-term MA and current volatility
                 long_trend_period: int = 120,  # For longer-term MA slope for trend detection
                 volatility_quantile_window: int = 252,  # Window for historical volatility distribution
                 volatility_quantile_threshold: float = 0.75,  # Percentile for classifying as 'Volatile'
                 trend_slope_threshold: float = 0.0001  # Min absolute slope of long MA for "strong" trending
                 ):
        """
        Initializes the MarketRegimeDetector.

        Args:
            lookback_period (int): The number of periods to look back for short-term MA and current volatility.
            long_trend_period (int): The number of periods for the longer-term MA used in trend slope calculation.
                                     Should typically be >= lookback_period.
            volatility_quantile_window (int): The window size to calculate the historical volatility distribution.
                                              Used to determine the threshold for 'Volatile' vs 'Quiet'.
            volatility_quantile_threshold (float): The quantile (e.g., 0.75 for 75th percentile)
                                                   used to classify 'Volatile' vs 'Quiet'.
            trend_slope_threshold (float): Minimum absolute slope (per period) of the long MA
                                           to classify as a strong "Trending" regime.
        """
        if lookback_period < 2:
            raise ValueError("lookback_period must be at least 2 for MA and volatility calculations.")
        if long_trend_period < 2:
            raise ValueError("long_trend_period must be at least 2.")
        if long_trend_period < lookback_period:
            logger.warning(
                "long_trend_period is less than lookback_period. This might lead to inconsistent trend detection.")
        if volatility_quantile_window < 2:
            raise ValueError("volatility_quantile_window must be at least 2.")
        if not (0 <= volatility_quantile_threshold <= 1):
            raise ValueError("volatility_quantile_threshold must be between 0 and 1.")
        if trend_slope_threshold < 0:
            raise ValueError("trend_slope_threshold cannot be negative.")

        self.lookback = lookback_period
        self.long_trend_period = long_trend_period
        self.volatility_quantile_window = volatility_quantile_window
        self.volatility_quantile_threshold = volatility_quantile_threshold
        self.trend_slope_threshold = trend_slope_threshold

        logger.info(f"MarketRegimeDetector initialized with lookback={lookback_period}, "
                    f"long_trend_period={long_trend_period}, vol_quantile_window={volatility_quantile_window}, "
                    f"vol_quantile_threshold={volatility_quantile_threshold}, trend_slope_threshold={trend_slope_threshold}.")

    def detect(self, market_data: pd.Series) -> str:
        """
        Classifies the market into one of four regimes:
        'Trending_Volatile', 'Trending_Quiet', 'Ranging_Volatile', 'Ranging_Quiet'.

        Args:
            market_data (pd.Series): A time series of market prices (e.g., SPY adjusted close).
                                     Must have sufficient history for rolling calculations.
                                     Assumed to be sorted by index.

        Returns:
            str: The detected market regime string. Defaults to 'Ranging_Quiet' if data is insufficient
                 for reliable detection.
        """
        if market_data.empty:
            logger.warning("Market data is empty. Cannot detect regime. Returning 'Ranging_Quiet'.")
            return "Ranging_Quiet"

        # Ensure market_data is sorted by index (though caller should ideally ensure this)
        market_data = market_data.sort_index()

        returns = market_data.pct_change().dropna()  # Drop the first NaN from pct_change

        # Determine the minimum required data for all calculations
        min_required_data = max(self.lookback, self.long_trend_period,
                                self.volatility_quantile_window) + 1  # +1 for pct_change to have returns for lookback

        if len(market_data) < min_required_data:
            logger.warning(f"Insufficient market data ({len(market_data)} points) for regime detection lookbacks. "
                           f"Required at least {min_required_data} points for reliable calculation. Returning 'Ranging_Quiet'.")
            return "Ranging_Quiet"

        # --- Trend Detection ---
        # Using longer MA slope for primary trend
        long_ma = market_data.rolling(window=self.long_trend_period,
                                      min_periods=max(1, self.long_trend_period // 2)).mean()

        is_trending_flag = False
        trend_direction_ma_cross = None
        trend_slope_ma = None

        if not long_ma.empty and not pd.isna(long_ma.iloc[-1]) and len(long_ma) >= 2:
            # Calculate slope of the last few points of the long MA
            # Use a short window (e.g., 5 periods) for recent MA slope
            ma_slope_window = min(5, len(long_ma))  # Ensure window doesn't exceed available data
            if ma_slope_window >= 2:
                # Calculate simple linear regression slope
                y_coords = long_ma.iloc[-ma_slope_window:].values
                x_coords = np.arange(ma_slope_window)
                if len(y_coords) == len(x_coords) and len(x_coords) > 1 and np.std(x_coords) > 0:
                    try:
                        # np.polyfit returns coefficients [slope, intercept] for degree 1
                        trend_slope_ma = np.polyfit(x_coords, y_coords, 1)[0]
                    except np.linalg.LinAlgError:
                        logger.debug(
                            "LinAlgError in polyfit for MA slope. Data might be constant. Skipping slope trend.")
                        trend_slope_ma = None
                else:
                    logger.debug(f"Insufficient data ({len(y_coords)} points) for MA slope calculation. Skipping.")

            # MA cross for confirmation of direction (short_ma_value > long_ma_value)
            short_ma_value = \
            market_data.rolling(window=self.lookback // 2, min_periods=max(1, self.lookback // 4)).mean().iloc[-1]
            long_ma_value = long_ma.iloc[-1]

            if not pd.isna(short_ma_value) and not pd.isna(long_ma_value):
                if short_ma_value > long_ma_value:
                    trend_direction_ma_cross = "Up"
                elif short_ma_value < long_ma_value:
                    trend_direction_ma_cross = "Down"
                else:
                    trend_direction_ma_cross = "Flat"

        if trend_slope_ma is not None and abs(trend_slope_ma) > self.trend_slope_threshold:
            # Strong trend detected by slope
            is_trending = "Trending"
            is_trending_flag = True
        elif trend_direction_ma_cross in ["Up", "Down"]:
            # Weaker trend detected by MA cross, but still trending
            is_trending = "Trending"
            is_trending_flag = True
        else:
            is_trending = "Ranging"

        # --- Volatility Detection ---
        current_volatility_series = returns.rolling(window=self.lookback, min_periods=max(1, self.lookback // 2)).std()
        current_volatility = current_volatility_series.iloc[-1] if not current_volatility_series.empty else np.nan

        # If current_volatility is NaN, it's due to insufficient data or constant prices in window
        if pd.isna(current_volatility) or current_volatility == 0:
            logger.debug(f"Current volatility is NaN/Zero for lookback {self.lookback}. Defaulting to 'Quiet'.")
            is_volatile = "Quiet"
        else:
            historical_volatility_distribution = returns.rolling(window=self.volatility_quantile_window,
                                                                 min_periods=max(1,
                                                                                 self.volatility_quantile_window // 2)).std().dropna()

            # If historical distribution is empty or mostly NaNs, cannot calculate quantile reliably
            if historical_volatility_distribution.empty:
                logger.warning(
                    f"Historical volatility distribution could not be calculated for window {self.volatility_quantile_window}. Defaulting to 'Quiet'.")
                volatility_threshold = 0.0  # Effectively make it quiet
            else:
                volatility_threshold = historical_volatility_distribution.quantile(self.volatility_quantile_threshold)
                if pd.isna(volatility_threshold):  # Fallback if quantile results in NaN (e.g., all same vol)
                    logger.warning(
                        f"Volatility threshold is NaN. Falling back to 0.0. Current vol: {current_volatility:.4f}.")
                    volatility_threshold = 0.0

            is_volatile = "Volatile" if current_volatility > volatility_threshold else "Quiet"

        regime = f"{is_trending}_{is_volatile}"
        logger.info(f"Market Regime Detected: {regime} "
                    f"(Current Vol: {current_volatility:.4f}, Vol Threshold: {volatility_threshold:.4f}, "
                    f"Long MA Slope: {trend_slope_ma:.6f} [Threshold: {self.trend_slope_threshold}], "
                    f"MA Cross Trend: {trend_direction_ma_cross})")
        return regime


class MetaStrategy(BaseStrategy):
    """
    A 'strategy of strategies' that dynamically selects a sub-strategy
    based on the detected market regime.
    """

    def __init__(self,
                 name: str = "MetaStrategy",
                 market_symbol: str = 'SPY',
                 regime_detector_params: Optional[Dict[str, Any]] = None,
                 sub_strategy_params: Optional[Dict[str, Dict[str, Any]]] = None
                 ):
        """
        Initializes the MetaStrategy.

        Args:
            name (str): A unique identifier for the strategy.
            market_symbol (str): The symbol of the market index to use for regime detection (e.g., 'SPY').
            regime_detector_params (Optional[Dict[str, Any]]): Parameters for the MarketRegimeDetector.
                                                                E.g., {'lookback_period': 90, 'volatility_quantile_threshold': 0.8}.
            sub_strategy_params (Optional[Dict[str, Dict[str, Any]]]): A dictionary where keys are regime names
                                                                        and values are dictionaries of parameters
                                                                        for the corresponding sub-strategy.
                                                                        E.g., {'Ranging_Quiet': {'zscore_entry': 1.5}}.
        """
        super().__init__(name)
        self.market_symbol = market_symbol

        # Initialize MarketRegimeDetector with provided parameters or defaults
        self.regime_detector_params = regime_detector_params if regime_detector_params is not None else {}
        self.regime_detector = MarketRegimeDetector(**self.regime_detector_params)

        # Default sub-strategy parameters
        default_sub_strategy_params = {
            "Trending_Volatile": {},  # AdvancedMLStrategy defaults
            "Trending_Quiet": {},  # AdvancedMLStrategy defaults
            "Ranging_Volatile": {'zscore_entry': 2.5},  # StaticPairsStrategy specific
            "Ranging_Quiet": {'zscore_entry': 2.0}  # StaticPairsStrategy specific
        }
        self.sub_strategy_params = default_sub_strategy_params
        if sub_strategy_params:
            # Merge provided params, allowing user to override or add specific params
            for regime, params in sub_strategy_params.items():
                if regime in self.sub_strategy_params:
                    # Merge dictionaries: existing parameters take precedence if not overridden
                    self.sub_strategy_params[regime].update(params)
                else:
                    self.sub_strategy_params[regime] = params  # Add new regime if defined

        # Instantiate sub-strategies with their specific parameters
        self.sub_strategies: Dict[str, BaseStrategy] = {
            "Trending_Volatile": AdvancedMLStrategy(**self.sub_strategy_params.get("Trending_Volatile", {})),
            "Trending_Quiet": AdvancedMLStrategy(**self.sub_strategy_params.get("Trending_Quiet", {})),
            "Ranging_Volatile": StaticPairsStrategy(**self.sub_strategy_params.get("Ranging_Volatile", {})),
            "Ranging_Quiet": StaticPairsStrategy(**self.sub_strategy_params.get("Ranging_Quiet", {}))
        }
        self.current_strategy_name: Optional[str] = None  # Name of the currently active sub-strategy
        self._current_active_strategy: Optional[BaseStrategy] = None  # Reference to the current active strategy object

        logger.info(f"{self.name} initialized. Sub-strategies: {list(self.sub_strategies.keys())}")

    def fit(self, historical_data: pd.DataFrame):
        """
        Fits all underlying sub-strategies using the provided historical data.
        This prepares all potential strategies for activation.

        Args:
            historical_data (pd.DataFrame): The full historical data available for fitting.
                                            This data is passed to all sub-strategies.
        """
        logger.info(f"{self.name}: Fitting all sub-strategies...")
        # Ensure market symbol data is available for initial regime detection if needed by sub-strategies
        if self.market_symbol not in historical_data.columns:
            logger.warning(f"Market symbol '{self.market_symbol}' not found in historical_data. "
                           "Sub-strategies depending on it might not fit correctly.")

        for name, strategy in self.sub_strategies.items():
            logger.info(f"Fitting sub-strategy: {name} ({strategy.name})...")
            try:
                # Pass the entire historical_data to each sub-strategy for its fitting logic
                strategy.fit(historical_data)
                logger.info(f"Sub-strategy {name} ({strategy.name}) fitted successfully.")
            except Exception as e:
                logger.error(f"Error fitting sub-strategy {name} ({strategy.name}): {e}. "
                             "This sub-strategy might not function correctly.")
        self.is_fitted = True
        logger.info(f"{self.name} fitting complete. All sub-strategies initialized.")

    @property
    def active_tradeable_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns the list of tradeable pairs from the currently active sub-strategy.
        This provides external access to the pairs being considered by the active strategy.

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple represents a tradeable pair (asset1, asset2).
                                    Returns an empty list if no strategy is active or if the active strategy
                                    does not have a 'tradeable_pairs' attribute.
        """
        # Ensure _current_active_strategy is not None and has the 'tradeable_pairs' attribute
        # Use hasattr for robustness as not all BaseStrategy subclasses might implement it directly.
        if self._current_active_strategy and hasattr(self._current_active_strategy, 'tradeable_pairs'):
            # Access the attribute directly. Type ignore is used because mypy might not infer
            # that self._current_active_strategy actually has this attribute dynamically.
            return self._current_active_strategy.tradeable_pairs  # type: ignore
        return []

    def generate_signals(self,
                         current_data_window: pd.DataFrame,
                         portfolio_context: Dict) -> Dict[Tuple[str, str], float]:
        """
        First, detect the current market regime. Then, delegate signal generation
        to the appropriate sub-strategy based on the detected regime.

        Args:
            current_data_window (pd.DataFrame): A rolling window of historical data
                                                up to the current time point.
                                                Must include the `market_symbol` column.
            portfolio_context (Dict): The current state of the portfolio.

        Returns:
            Dict[Tuple[str, str], float]: A dictionary mapping asset pairs to target positions.
                                          This is the consolidated signal from the active sub-strategy.
        """
        if not self.is_fitted:
            logger.error("MetaStrategy must be fitted first before generating signals.")
            raise RuntimeError("MetaStrategy must be fitted first.")

        # --- 1. Detect Market Regime ---
        market_series: pd.Series
        if self.market_symbol not in current_data_window.columns:
            logger.warning(f"Market symbol '{self.market_symbol}' not found in current_data_window. "
                           "Averaging all available series for regime detection.")
            # Fallback: Use mean of all available numeric columns for regime detection
            numeric_cols = current_data_window.select_dtypes(include=np.number).columns
            if numeric_cols.empty:
                logger.error(
                    "No numeric columns found in current_data_window for regime detection. Returning no signals.")
                return {}
            market_series = current_data_window[numeric_cols].mean(axis=1)
        else:
            market_series = current_data_window[self.market_symbol]

        # Ensure market_series is not empty and has enough data for detector's lookbacks
        min_regime_data = max(self.regime_detector.lookback, self.regime_detector.long_trend_period,
                              self.regime_detector.volatility_quantile_window)
        if market_series.empty or len(market_series) < min_regime_data:
            logger.warning(f"Market series for regime detection is too short or empty ({len(market_series)} points). "
                           f"Required at least {min_regime_data} points for detector. Returning no signals.")
            return {}

        current_regime = self.regime_detector.detect(market_series)

        # --- 2. Select Active Sub-Strategy ---
        active_strategy = self.sub_strategies.get(current_regime)
        if active_strategy is None:
            logger.warning(
                f"No specific sub-strategy defined for regime '{current_regime}'. Defaulting to 'Ranging_Quiet' strategy.")
            active_strategy = self.sub_strategies["Ranging_Quiet"]  # Fallback

        # Log strategy switch only if it's actually changing
        # Compare strategy objects directly to avoid issues with identical names but different instances
        if self._current_active_strategy != active_strategy:
            previous_strategy_name = self.current_strategy_name
            self.current_strategy_name = active_strategy.name
            self._current_active_strategy = active_strategy  # Update reference to active strategy object
            logger.info(
                f"Market regime changed to '{current_regime}'. Switching active strategy from '{previous_strategy_name}' to '{self.current_strategy_name}'.")
        else:
            logger.debug(
                f"Current strategy '{self.current_strategy_name}' remains active for regime '{current_regime}'.")

        # --- 3. Delegate Signal Generation ---
        # Call generate_signals on the chosen sub-strategy
        # Pass the full current_data_window and portfolio_context
        signals = active_strategy.generate_signals(current_data_window, portfolio_context)

        # Log the number of signals generated by the active strategy
        if signals:
            logger.debug(f"Active strategy '{self.current_strategy_name}' generated {len(signals)} signals.")
        else:
            logger.debug(f"Active strategy '{self.current_strategy_name}' generated no signals.")

        return signals

    def reset(self):
        """
        Resets the internal state of the MetaStrategy and all its sub-strategies.
        """
        super().reset()  # Resets self.is_fitted
        self.current_strategy_name = None
        self._current_active_strategy = None  # Clear active strategy reference
        logger.info(f"{self.name}: Resetting all sub-strategies...")
        for name, strategy in self.sub_strategies.items():
            strategy.reset()
            logger.debug(f"Sub-strategy {name} reset.")
        logger.info(f"{self.name} reset complete.")