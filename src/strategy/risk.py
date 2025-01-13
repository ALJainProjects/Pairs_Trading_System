"""
Enhanced Risk Management for Pairs Trading

Handles:
- Pair-specific position sizing
- Spread-based stop losses
- Portfolio-level risk for multiple pairs
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from config.logging_config import logger

class PairRiskManager:
    """Risk manager specifically designed for pairs trading."""

    def __init__(
        self,
        max_position_size: float = 0.05,  # 5% of portfolio per pair
        max_drawdown: float = 0.20,       # 20% max drawdown
        stop_loss_threshold: float = 0.10, # 10% stop-loss on spread
        max_correlation: float = 0.7,      # Maximum correlation between pairs
        leverage_limit: float = 2.0        # Maximum total leverage
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum size per pair
            max_drawdown: Maximum portfolio drawdown
            stop_loss_threshold: Stop loss on spread movement
            max_correlation: Maximum correlation between active pairs
            leverage_limit: Maximum total leverage
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_threshold = stop_loss_threshold
        self.max_correlation = max_correlation
        self.leverage_limit = leverage_limit

    def position_sizing(self,
                       portfolio_value: float,
                       pair_prices: Tuple[float, float],
                       num_active_pairs: int,
                       pair_volatility: Optional[float] = None) -> float:
        """
        Calculate position size for a pair trade.

        Args:
            portfolio_value: Current portfolio value
            pair_prices: Tuple of (price1, price2)
            num_active_pairs: Number of currently active pairs
            pair_volatility: Optional volatility of the pair spread

        Returns:
            float: Position size in units
        """
        # Adjust position size based on number of pairs
        adjusted_size = self.max_position_size / max(1, num_active_pairs)

        # Calculate base position value
        max_position_value = portfolio_value * adjusted_size

        # Adjust for pair volatility if provided
        if pair_volatility is not None:
            vol_adjustment = 1.0 / (1.0 + pair_volatility)
            max_position_value *= vol_adjustment

        # Convert to quantity based on pair prices
        price1, price2 = pair_prices
        pair_price = price1 + price2  # Total cost for one unit of the pair

        if pair_price <= 0:
            return 0

        return max_position_value / pair_price

    def check_pair_correlation(self,
                             new_pair_returns: pd.DataFrame,
                             active_pairs_returns: pd.DataFrame) -> bool:
        """
        Check if adding a new pair would exceed correlation limits.

        Args:
            new_pair_returns: Returns of the new pair
            active_pairs_returns: Returns of currently active pairs

        Returns:
            bool: True if correlation is acceptable
        """
        if active_pairs_returns.empty:
            return True

        # Calculate correlation between new pair and active pairs
        correlation = new_pair_returns.corrwith(active_pairs_returns)
        max_corr = abs(correlation).max()

        return max_corr < self.max_correlation

    def check_spread_stop_loss(self,
                             entry_spread: float,
                             current_spread: float,
                             position_type: str) -> bool:
        """
        Check spread-based stop loss.

        Args:
            entry_spread: Spread at entry
            current_spread: Current spread value
            position_type: 'long' or 'short'

        Returns:
            bool: True if stop loss is triggered
        """
        if entry_spread == 0:
            return False

        if position_type == 'long':
            loss = (entry_spread - current_spread) / abs(entry_spread)
        else:  # short
            loss = (current_spread - entry_spread) / abs(entry_spread)

        return loss >= self.stop_loss_threshold

    def calculate_pair_exposure(self,
                              positions: Dict[str, Dict],
                              current_prices: Dict[str, float]) -> float:
        """
        Calculate total exposure from all pair positions.

        Args:
            positions: Dictionary of current positions
            current_prices: Dictionary of current prices

        Returns:
            float: Total exposure
        """
        total_exposure = 0

        for pair, position in positions.items():
            asset1, asset2 = pair
            quantity = position['quantity']

            # Calculate exposure for both legs
            exposure1 = abs(quantity * current_prices[asset1])
            exposure2 = abs(quantity * current_prices[asset2])
            total_exposure += exposure1 + exposure2

        return total_exposure

    def monitor_portfolio_risk(self,
                             equity_curve: pd.Series,
                             positions: Dict,
                             current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Monitor portfolio-wide risk constraints.

        Args:
            equity_curve: Portfolio equity curve
            positions: Current positions
            current_prices: Current asset prices

        Returns:
            Tuple of (risk_violated, reason)
        """
        if equity_curve.empty:
            return False, ""

        current_equity = equity_curve.iloc[-1]

        # 1. Check drawdown
        drawdown = self.calculate_drawdown(equity_curve)
        if drawdown > self.max_drawdown:
            return True, f"Max drawdown exceeded: {drawdown:.2%}"

        # 2. Check leverage
        total_exposure = self.calculate_pair_exposure(positions, current_prices)
        current_leverage = total_exposure / current_equity

        if current_leverage > self.leverage_limit:
            return True, f"Leverage limit exceeded: {current_leverage:.2f}x"

        # 3. Check position sizes
        for pair, position in positions.items():
            asset1, asset2 = pair
            quantity = position['quantity']
            position_value = quantity * (
                current_prices[asset1] + current_prices[asset2]
            )
            position_size = position_value / current_equity

            if position_size > self.max_position_size:
                return True, f"Position size exceeded for {pair}: {position_size:.2%}"

        return False, ""

    def calculate_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peaks = equity_curve.expanding().max()
        drawdowns = (equity_curve - peaks) / peaks
        return abs(float(drawdowns.min()))

    def generate_risk_report(self,
                           equity_curve: pd.Series,
                           positions: Dict,
                           trades: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk report.

        Args:
            equity_curve: Portfolio equity curve
            positions: Current positions
            trades: Trade history

        Returns:
            Dictionary with risk metrics
        """
        # Calculate basic metrics
        drawdown = self.calculate_drawdown(equity_curve)
        returns = equity_curve.pct_change().dropna()

        # Calculate exposure and leverage
        latest_equity = equity_curve.iloc[-1]
        exposures = []

        for date in equity_curve.index:
            date_trades = trades[trades['Date'] == date]
            exposure = date_trades['Quantity'].abs() * date_trades['Price']
            exposures.append(exposure.sum())

        exposure_series = pd.Series(exposures, index=equity_curve.index)
        leverage_series = exposure_series / equity_curve

        report = {
            'Current Metrics': {
                'Drawdown': drawdown,
                'Volatility': returns.std() * np.sqrt(252),
                'Current Leverage': leverage_series.iloc[-1],
                'Active Pairs': len(positions)
            },
            'Risk Parameters': {
                'Max Position Size': self.max_position_size,
                'Max Drawdown': self.max_drawdown,
                'Stop Loss': self.stop_loss_threshold,
                'Max Correlation': self.max_correlation,
                'Leverage Limit': self.leverage_limit
            },
            'Time Series': {
                'Leverage': leverage_series,
                'Exposure': exposure_series
            }
        }

        return report