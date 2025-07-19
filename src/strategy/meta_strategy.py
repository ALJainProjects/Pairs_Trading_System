import pandas as pd
from typing import Dict, Tuple

from .base import BaseStrategy
from .static_pairs_strategy import StaticPairsStrategy
from .advanced_ml_strategy import AdvancedMLStrategy
from config.logging_config import logger


class MarketRegimeDetector:
    """Detects market regimes based on volatility and trend."""

    def __init__(self, lookback_period: int = 60):
        self.lookback = lookback_period

    def detect(self, market_data: pd.Series) -> str:
        """
        Classifies the market into one of four regimes.
        """
        returns = market_data.pct_change()

        short_ma = market_data.rolling(window=20).mean().iloc[-1]
        long_ma = market_data.rolling(window=self.lookback).mean().iloc[-1]
        is_trending = "Trending" if short_ma > long_ma else "Ranging"

        volatility = returns.rolling(window=self.lookback).std().iloc[-1]
        volatility_q75 = returns.rolling(window=252).std().quantile(0.75)
        is_volatile = "Volatile" if volatility > volatility_q75 else "Quiet"

        regime = f"{is_trending}_{is_volatile}"
        logger.info(f"Market Regime Detected: {regime}")
        return regime


class MetaStrategy(BaseStrategy):
    """
    A 'strategy of strategies' that dynamically selects a sub-strategy
    based on the detected market regime.
    """

    def __init__(self, name="MetaStrategy", market_symbol: str = 'SPY'):
        super().__init__(name)
        self.market_symbol = market_symbol
        self.regime_detector = MarketRegimeDetector()

        self.sub_strategies = {
            "Trending_Volatile": AdvancedMLStrategy(),
            "Trending_Quiet": AdvancedMLStrategy(),
            "Ranging_Volatile": StaticPairsStrategy(zscore_entry=2.5),
            "Ranging_Quiet": StaticPairsStrategy(zscore_entry=2.0)
        }
        self.current_strategy_name = None

    def fit(self, historical_data: pd.DataFrame):
        """Fit all underlying sub-strategies."""
        logger.info("MetaStrategy: Fitting all sub-strategies...")
        for name, strategy in self.sub_strategies.items():
            logger.info(f"Fitting sub-strategy: {name}")
            strategy.fit(historical_data)
        self.is_fitted = True

    def generate_signals(self, current_data_window: pd.DataFrame, portfolio_context: Dict) -> Dict[
        Tuple[str, str], float]:
        """
        First, detect the market regime. Then, delegate signal generation
        to the appropriate sub-strategy.
        """
        if not self.is_fitted:
            raise RuntimeError("MetaStrategy must be fitted first.")

        if self.market_symbol not in current_data_window.columns:
            market_series = current_data_window.mean(axis=1)
        else:
            market_series = current_data_window[self.market_symbol]

        current_regime = self.regime_detector.detect(market_series)

        active_strategy = self.sub_strategies.get(current_regime)
        if active_strategy is None:
            logger.warning(f"No strategy defined for regime '{current_regime}'. Defaulting to Ranging_Quiet.")
            active_strategy = self.sub_strategies["Ranging_Quiet"]

        if self.current_strategy_name != active_strategy.name:
            self.current_strategy_name = active_strategy.name
            logger.info(f"Switching to strategy: {self.current_strategy_name} for regime: {current_regime}")

        return active_strategy.generate_signals(current_data_window, portfolio_context)