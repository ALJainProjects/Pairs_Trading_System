import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
from config.logging_config import logger


class PortfolioRiskManager:
    """
    Manages risk at both the individual position and overall portfolio level for pairs trading.
    This class handles position sizing, enforces risk limits, and calculates dynamic stop-loss distances.
    """

    def __init__(self,
                 initial_capital: float,
                 max_portfolio_exposure: float = 1.0,
                 max_position_concentration: float = 0.2,
                 default_position_risk_pct: float = 0.02,
                 stop_distance_multiplier: float = 3.0,
                 volatility_period: int = 60,
                 min_volatility_obs: int = 30):
        """
        Initializes the PortfolioRiskManager.

        Args:
            initial_capital (float): The starting capital of the portfolio.
            max_portfolio_exposure (float): Maximum total absolute dollar exposure (leverage).
            max_position_concentration (float): Maximum percentage of portfolio equity in any single asset.
            default_position_risk_pct (float): Percentage of total portfolio equity to risk per pair trade.
            stop_distance_multiplier (float): Multiplier for the spread's volatility to define stop-loss.
            volatility_period (int): Rolling window period for calculating spread volatility.
            min_volatility_obs (int): Minimum observations for volatility calculation.
        """
        if initial_capital <= 0: raise ValueError("Initial capital must be positive.")
        if not (0 <= max_portfolio_exposure): raise ValueError("Max portfolio exposure must be non-negative.")
        if not (0 <= max_position_concentration <= 1): raise ValueError(
            "Max position concentration must be between 0 and 1.")
        if not (0 < default_position_risk_pct <= 1): raise ValueError(
            "Default position risk percentage must be between 0 and 1 (exclusive of 0).")
        if not (0 < stop_distance_multiplier): raise ValueError("Stop distance multiplier must be positive.")
        if volatility_period < 2: raise ValueError("Volatility period must be at least 2.")
        if min_volatility_obs < 1 or min_volatility_obs > volatility_period: raise ValueError(
            "Min volatility observations must be between 1 and volatility_period.")

        self.initial_capital = initial_capital
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_position_concentration = max_position_concentration
        self.default_position_risk_pct = default_position_risk_pct
        self.stop_distance_multiplier = stop_distance_multiplier
        self.volatility_period = volatility_period
        self.min_volatility_obs = min_volatility_obs

        logger.info(
            f"PortfolioRiskManager initialized (Capital: {initial_capital}, Exposure: {max_portfolio_exposure}).")

    def calculate_target_positions(self,
                                   raw_signals: Dict[Tuple[str, str], float],
                                   current_prices: Dict[str, float],
                                   historical_data: pd.DataFrame,
                                   portfolio_equity: float) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Translates raw strategy signals into target dollar positions for each leg of a pair,
        based on calculated risk.

        Args:
            raw_signals (Dict): Raw signals from the strategy (-1: short spread, 0: flat, 1: long spread).
            current_prices (Dict): Current market prices of all assets.
            historical_data (pd.DataFrame): Historical price data for all assets (columns are symbols, index is date).
            portfolio_equity (float): Current total equity of the portfolio.

        Returns:
            Dict: A dictionary mapping pairs to a nested dictionary of target dollar positions for each leg.
                  E.g., {('AAPL', 'MSFT'): {'AAPL': 1000.0, 'MSFT': -980.0}}.
        """
        target_positions: Dict[Tuple[str, str], Dict[str, float]] = {}
        historical_data = historical_data.sort_index()

        for pair, signal in raw_signals.items():
            if signal == 0: continue

            asset1, asset2 = pair
            if asset1 not in current_prices or asset2 not in current_prices:
                logger.warning(f"Current prices missing for {pair}. Skipping.")
                continue
            if asset1 not in historical_data.columns or asset2 not in historical_data.columns:
                logger.warning(f"Historical data missing for {pair}. Skipping.")
                continue

            # Calculate spread history and its volatility for stop-loss distance
            spread_history = historical_data[asset1] - historical_data[asset2]
            spread_volatility = self.calculate_spread_volatility(
                spread_history, self.volatility_period, self.min_volatility_obs
            )

            if spread_volatility <= 1e-9:
                logger.warning(f"Spread volatility for {pair} is zero or near zero. Skipping.")
                continue

            # Define stop-loss distance in spread units
            stop_loss_distance = spread_volatility * self.stop_distance_multiplier

            # Risk amount based on portfolio equity
            risk_amount = portfolio_equity * self.default_position_risk_pct

            # Number of 'spread units' (conceptual shares) based on risk
            num_shares = risk_amount / stop_loss_distance

            # Calculate dollar value for each leg, assuming 1:1 shares per spread unit
            # This is a simplification; a true hedge ratio would be needed for dollar neutrality.
            position_value1 = num_shares * current_prices[asset1]
            position_value2 = num_shares * current_prices[asset2]

            target_positions[pair] = {
                asset1: signal * position_value1,
                asset2: -signal * position_value2,
            }
            logger.debug(
                f"Pair {pair}: Raw target: {asset1}={signal * position_value1:.2f}, {asset2}={-signal * position_value2:.2f}")

        return self.enforce_portfolio_limits(target_positions, portfolio_equity)

    def enforce_portfolio_limits(self,
                                 target_positions: Dict[Tuple[str, str], Dict[str, float]],
                                 portfolio_equity: float) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Enforces portfolio-level total exposure and individual asset concentration limits.

        Args:
            target_positions (Dict): Mapping of pairs to target dollar positions.
            portfolio_equity (float): Current total equity of the portfolio.

        Returns:
            Dict: Adjusted target positions after enforcing limits.
        """
        if not target_positions: return {}
        adjusted_positions = {pair: legs.copy() for pair, legs in target_positions.items()}

        # 1. Concentration Limit (per single asset)
        asset_exposure: Dict[str, float] = {}
        for pair, legs in adjusted_positions.items():
            for asset, value in legs.items():
                asset_exposure[asset] = asset_exposure.get(asset, 0.0) + abs(value)

        for asset, exposure in asset_exposure.items():
            max_allowed_asset_value = portfolio_equity * self.max_position_concentration
            if exposure > max_allowed_asset_value:
                scale_down_factor = max_allowed_asset_value / exposure
                logger.warning(f"Breached concentration limit for {asset}. Scaling down by {scale_down_factor:.2f}.")
                for pair, legs in adjusted_positions.items():
                    if asset in legs:
                        legs[asset] *= scale_down_factor

        # 2. Total Exposure Limit
        total_exposure = sum(abs(v) for legs in adjusted_positions.values() for v in legs.values())
        max_allowed_total_exposure = portfolio_equity * self.max_portfolio_exposure

        if total_exposure > max_allowed_total_exposure:
            scale_down_factor = max_allowed_total_exposure / total_exposure
            logger.warning(f"Breached total exposure limit. Scaling all positions by {scale_down_factor:.2f}.")
            for pair, legs in adjusted_positions.items():
                for asset in legs:
                    legs[asset] *= scale_down_factor

        logger.info(
            f"Final total exposure after limits: {sum(abs(v) for legs in adjusted_positions.values() for v in legs.values()):.2f}")
        return adjusted_positions

    @staticmethod
    def calculate_spread_volatility(spread_series: pd.Series, period: int, min_obs: int) -> float:
        """
        Calculates the rolling standard deviation of the spread series.

        Args:
            spread_series (pd.Series): A time series of spread values.
            period (int): Rolling window period for standard deviation.
            min_obs (int): Minimum observations required within the rolling window.

        Returns:
            float: The latest rolling standard deviation of the spread. Returns 0.0 if insufficient data or zero volatility.
        """
        spread_clean = spread_series.dropna()
        if len(spread_clean) < period or len(spread_clean) < min_obs: return 0.0

        rolling_std = spread_clean.rolling(window=period, min_periods=min_obs).std()

        latest_std = rolling_std.iloc[-1] if not rolling_std.empty and pd.notna(rolling_std.iloc[-1]) else 0.0

        if latest_std < 1e-9: return 0.0  # Treat very small std dev as zero

        return latest_std