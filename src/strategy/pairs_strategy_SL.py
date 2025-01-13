"""
Statistical Pairs Trading Module (Corrected for Look-Ahead Bias)

This module implements statistical methods for pairs trading with:
1. Rolling cointegration testing
2. Proper temporal spread calculation
3. Forward-looking bias prevention in signal generation
4. Walk-forward validation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from config.logging_config import logger


class StatisticalPairsTrader:
    """Statistical pairs trading with proper temporal handling."""

    def __init__(self,
                 lookback_window: int = 252,  # 1 year
                 zscore_window: int = 20,
                 zscore_threshold: float = 2.0,
                 min_half_life: int = 5,
                 max_half_life: int = 126):  # ~6 months
        """
        Initialize the pairs trader.

        Args:
            lookback_window: Window for parameter estimation
            zscore_window: Window for z-score calculation
            zscore_threshold: Threshold for trading signals
            min_half_life: Minimum half-life for mean reversion
            max_half_life: Maximum half-life for mean reversion
        """
        self.lookback_window = lookback_window
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def calculate_rolling_cointegration(self,
                                        asset1: pd.Series,
                                        asset2: pd.Series,
                                        min_window: int = 126) -> pd.Series:
        """
        Calculate rolling cointegration p-values.

        Args:
            asset1: First asset prices
            asset2: Second asset prices
            min_window: Minimum window for test

        Returns:
            Series of p-values
        """
        if len(asset1) != len(asset2):
            raise ValueError("Series must have same length")

        pvalues = []
        for i in range(min_window, len(asset1)):
            # Use only past data for the test
            p_val = self._single_coint_test(
                asset1.iloc[i - min_window:i],
                asset2.iloc[i - min_window:i]
            )
            pvalues.append(p_val)

        return pd.Series(
            pvalues,
            index=asset1.index[min_window:],
            name='coint_pvalue'
        )

    def _single_coint_test(self,
                           asset1: pd.Series,
                           asset2: pd.Series) -> float:
        """Perform single cointegration test."""
        try:
            _, p_val, _ = coint(asset1, asset2)
            return p_val
        except:
            return 1.0  # Non-cointegrated case

    def calculate_rolling_hedge_ratio(self,
                                      asset1: pd.Series,
                                      asset2: pd.Series,
                                      window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate rolling hedge ratios using only past data.

        Args:
            asset1: First asset prices
            asset2: Second asset prices
            window: Rolling window size

        Returns:
            DataFrame with alpha and beta coefficients
        """
        window = window or self.lookback_window

        alphas = []
        betas = []

        for i in range(window, len(asset1)):
            # Use only past data for regression
            X = add_constant(asset2.iloc[i - window:i])
            y = asset1.iloc[i - window:i]

            try:
                model = OLS(y, X).fit()
                alpha, beta = model.params
            except:
                # Use previous values or defaults
                alpha = alphas[-1] if alphas else 0
                beta = betas[-1] if betas else 1

            alphas.append(alpha)
            betas.append(beta)

        return pd.DataFrame({
            'alpha': alphas,
            'beta': betas
        }, index=asset1.index[window:])

    def calculate_rolling_spread(self,
                                 asset1: pd.Series,
                                 asset2: pd.Series,
                                 use_ratio: bool = False) -> pd.DataFrame:
        """
        Calculate rolling spread using only past data.

        Args:
            asset1: First asset prices
            asset2: Second asset prices
            use_ratio: Whether to use ratio instead of regression

        Returns:
            DataFrame with spread and parameters
        """
        if use_ratio:
            spread = asset1 / asset2
            result = pd.DataFrame({
                'spread': spread,
                'beta': 1.0,
                'alpha': 0.0
            })
        else:
            # Calculate rolling hedge ratios
            params = self.calculate_rolling_hedge_ratio(
                asset1, asset2, self.lookback_window
            )

            # Calculate spread using rolling parameters
            spread = pd.Series(index=params.index)
            for i in range(len(params)):
                alpha = params.iloc[i]['alpha']
                beta = params.iloc[i]['beta']
                spread.iloc[i] = (
                        asset1.iloc[i + self.lookback_window] -
                        (beta * asset2.iloc[i + self.lookback_window] + alpha)
                )

            result = pd.DataFrame({
                'spread': spread,
                'beta': params['beta'],
                'alpha': params['alpha']
            })

        return result

    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life using OLS."""
        lagged_spread = spread.shift(1)
        delta_spread = spread - lagged_spread

        X = add_constant(lagged_spread.iloc[1:])
        y = delta_spread.iloc[1:]

        model = OLS(y, X).fit()
        gamma = model.params[1]  # Speed of mean reversion

        half_life = -np.log(2) / gamma if gamma < 0 else np.inf
        return half_life

    def calculate_rolling_zscore(self,
                                 spread: pd.Series,
                                 window: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling z-score using only past data.

        Args:
            spread: Spread series
            window: Window for z-score calculation

        Returns:
            Series of z-scores
        """
        window = window or self.zscore_window

        roll_mean = spread.rolling(window=window, min_periods=window).mean()
        roll_std = spread.rolling(window=window, min_periods=window).std()

        return (spread - roll_mean) / roll_std

    def generate_signals(self,
                         asset1: pd.Series,
                         asset2: pd.Series,
                         validation_start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Generate trading signals using walk-forward validation.

        Args:
            asset1: First asset prices
            asset2: Second asset prices
            validation_start: Start of validation period

        Returns:
            DataFrame with signals and metrics
        """
        if validation_start is None:
            validation_start = asset1.index[self.lookback_window]

        # Initialize results
        signals = pd.DataFrame(index=asset1.index)
        signals['position'] = 0

        # Calculate rolling cointegration
        coint_pvals = self.calculate_rolling_cointegration(
            asset1, asset2, min_window=self.lookback_window
        )

        # Calculate rolling spread
        spread_data = self.calculate_rolling_spread(asset1, asset2)

        # Calculate rolling z-scores
        zscores = self.calculate_rolling_zscore(
            spread_data['spread'],
            self.zscore_window
        )

        # Generate signals for validation period
        validation_mask = signals.index >= validation_start

        for i in signals.index[validation_mask]:
            # Check cointegration
            if coint_pvals.loc[i] > 0.05:
                continue

            # Check half-life
            current_spread = spread_data['spread'].loc[:i]
            half_life = self.calculate_half_life(current_spread)

            if not (self.min_half_life <= half_life <= self.max_half_life):
                continue

            # Generate signal based on z-score
            z_val = zscores.loc[i]
            if z_val > self.zscore_threshold:
                signals.loc[i, 'position'] = -1  # Short the spread
            elif z_val < -self.zscore_threshold:
                signals.loc[i, 'position'] = 1  # Long the spread

        # Add metrics
        signals['spread'] = spread_data['spread']
        signals['zscore'] = zscores
        signals['coint_pvalue'] = coint_pvals
        signals['hedge_ratio'] = spread_data['beta']

        return signals

    def calculate_returns(self,
                          signals: pd.DataFrame,
                          asset1: pd.Series,
                          asset2: pd.Series) -> pd.DataFrame:
        """
        Calculate strategy returns without look-ahead bias.

        Args:
            signals: DataFrame with trading signals
            asset1: First asset prices
            asset2: Second asset prices

        Returns:
            DataFrame with returns and metrics
        """
        results = signals.copy()

        # Calculate asset returns
        asset1_rets = asset1.pct_change()
        asset2_rets = asset2.pct_change()

        # Calculate strategy returns
        long_returns = results['position'].shift(1) * (
                asset1_rets - results['hedge_ratio'] * asset2_rets
        )

        results['strategy_returns'] = long_returns
        results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()

        return results


def main():
    """Example usage with proper temporal handling."""
    # Load data
    asset1_prices = pd.Series(...)  # First asset prices
    asset2_prices = pd.Series(...)  # Second asset prices

    # Initialize trader
    trader = StatisticalPairsTrader(
        lookback_window=252,
        zscore_window=20,
        zscore_threshold=2.0
    )

    # Generate signals using walk-forward validation
    signals = trader.generate_signals(
        asset1_prices,
        asset2_prices,
        validation_start=pd.Timestamp('2022-01-01')
    )

    # Calculate returns
    results = trader.calculate_returns(
        signals,
        asset1_prices,
        asset2_prices
    )

    return results


if __name__ == "__main__":
    main()