import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from config.logging_config import logger


class PortfolioRiskManager:
    """
    Manages risk at both the individual position and overall portfolio level.
    This class is responsible for position sizing, enforcing risk limits, and
    applying dynamic stop-losses.
    """

    def __init__(self,
                 initial_capital: float,
                 max_portfolio_exposure: float = 1.0,  # Max leverage
                 max_position_concentration: float = 0.2,  # Max % of portfolio in one asset
                 default_position_risk_pct: float = 0.02,  # Risk 2% of portfolio per trade
                 atr_stop_multiplier: float = 3.0,  # ATR-based stop loss multiplier
                 atr_period: int = 20):

        self.initial_capital = initial_capital
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_position_concentration = max_position_concentration
        self.default_position_risk_pct = default_position_risk_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_period = atr_period

    def calculate_target_positions(self,
                                   raw_signals: Dict[Tuple[str, str], float],
                                   current_prices: Dict[str, float],
                                   historical_data: pd.DataFrame,
                                   portfolio_equity: float) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Translates raw strategy signals into target dollar positions based on risk.

        Args:
            raw_signals (Dict): Raw signals from the strategy (e.g., -1, 0, 1).
            current_prices (Dict): Current market prices of all assets.
            historical_data (pd.DataFrame): Historical data for ATR calculation.
            portfolio_equity (float): Current total equity of the portfolio.

        Returns:
            Dict: A dictionary mapping pairs to their target dollar positions for each leg.
        """
        target_positions = {}

        for pair, signal in raw_signals.items():
            if signal == 0:
                continue

            asset1, asset2 = pair
            if asset1 not in current_prices or asset2 not in current_prices:
                continue

            # Calculate ATR for the spread to determine stop-loss distance
            spread_history = historical_data[asset1] - historical_data[asset2]  # Simple spread for risk sizing
            atr = self.calculate_atr(spread_history, self.atr_period)
            if atr == 0: continue

            stop_loss_distance = atr * self.atr_stop_multiplier

            # Position sizing based on volatility
            risk_per_unit = stop_loss_distance
            risk_amount = portfolio_equity * self.default_position_risk_pct

            num_units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

            # Calculate dollar value for each leg
            position_value1 = num_units * current_prices[asset1]
            position_value2 = num_units * current_prices[asset2]

            # Apply signal direction
            target_positions[pair] = {
                asset1: signal * position_value1,
                asset2: -signal * position_value2,
            }

        return self.enforce_portfolio_limits(target_positions, portfolio_equity)

    def enforce_portfolio_limits(self,
                                 target_positions: Dict,
                                 portfolio_equity: float) -> Dict:
        """Enforces portfolio-level exposure and concentration limits."""

        # 1. Concentration Limit
        asset_exposure = {}
        for pair, legs in target_positions.items():
            for asset, value in legs.items():
                asset_exposure[asset] = asset_exposure.get(asset, 0) + abs(value)

        max_allowed_asset_value = portfolio_equity * self.max_position_concentration
        for asset, exposure in asset_exposure.items():
            if exposure > max_allowed_asset_value:
                scale_down = max_allowed_asset_value / exposure
                logger.warning(f"Breached concentration limit for {asset}. Scaling down by {scale_down:.2f}.")
                for pair, legs in target_positions.items():
                    if asset in legs:
                        legs[asset] *= scale_down

        # 2. Total Exposure Limit
        total_exposure = sum(abs(v) for legs in target_positions.values() for v in legs.values())
        max_allowed_exposure = portfolio_equity * self.max_portfolio_exposure

        if total_exposure > max_allowed_exposure:
            scale_down = max_allowed_exposure / total_exposure
            logger.warning(f"Breached total exposure limit. Scaling all positions by {scale_down:.2f}.")
            for pair, legs in target_positions.items():
                for asset in legs:
                    legs[asset] *= scale_down

        return target_positions

    @staticmethod
    def calculate_atr(series: pd.Series, period: int) -> float:
        """Calculates the Average True Range for a given series."""
        if len(series) < period: return 0.0
        high_low = series.rolling(window=period).max() - series.rolling(window=period).min()
        return high_low.mean()