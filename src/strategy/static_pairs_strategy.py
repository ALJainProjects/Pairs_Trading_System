import pandas as pd
from typing import Dict, Tuple, List

from .base import BaseStrategy
from src.analysis.cointegration import find_cointegrated_pairs
from src.models.statistical import StatisticalModel


class StaticPairsStrategy(BaseStrategy):
    """
    A statistical arbitrage pairs trading strategy.

    This strategy identifies cointegrated pairs during an initial 'fit' phase
    and then generates trading signals based on the mean reversion of their spread's z-score.
    """

    def __init__(self,
                 name="StaticPairsStrategy",
                 lookback_window: int = 60,
                 zscore_entry: float = 2.0,
                 zscore_exit: float = 0.5,
                 max_pairs: int = 10):

        super().__init__(name,
                         lookback_window=lookback_window,
                         zscore_entry=zscore_entry,
                         zscore_exit=zscore_exit,
                         max_pairs=max_pairs)

        self.stat_model = StatisticalModel()
        self.tradeable_pairs: List[Tuple[str, str]] = []

    def fit(self, historical_data: pd.DataFrame):
        """
        Finds the top N cointegrated pairs from the historical data.
        These pairs will be traded for the entire duration of the backtest.
        """
        pairs_found = find_cointegrated_pairs(historical_data)

        if not pairs_found:
            raise ValueError("No cointegrated pairs found in the provided historical data.")

        self.tradeable_pairs = [tuple(p.values())[:2] for p in pairs_found[:self.params['max_pairs']]]
        self.is_fitted = True

    def generate_signals(self,
                         current_data_window: pd.DataFrame,
                         portfolio_context: Dict) -> Dict[Tuple[str, str], float]:
        """
        Generates trading signals for the statically defined pairs.
        """
        if not self.is_fitted:
            raise RuntimeError("Strategy has not been fitted. Call fit() before running the backtest.")

        signals = {}
        # Extract symbols from the position objects for checking
        open_symbols = {asset for pos in portfolio_context['positions'] for asset in (pos.symbol,)}

        for pair in self.tradeable_pairs:
            asset1, asset2 = pair

            hedge_ratio = self.stat_model.calculate_rolling_hedge_ratio(
                current_data_window[asset1],
                current_data_window[asset2],
                window=self.params['lookback_window']
            )
            spread = self.stat_model.calculate_spread(
                current_data_window[asset1],
                current_data_window[asset2],
                hedge_ratio
            )
            z_score = self.stat_model.calculate_zscore(spread, window=self.params['lookback_window'])

            if z_score.empty or pd.isna(z_score.iloc[-1]):
                continue

            current_z = z_score.iloc[-1]

            is_open = asset1 in open_symbols or asset2 in open_symbols

            # Entry Logic
            if not is_open:
                if current_z > self.params['zscore_entry']:
                    signals[pair] = -1.0  # Short the spread
                elif current_z < -self.params['zscore_entry']:
                    signals[pair] = 1.0  # Long the spread

            # Exit Logic
            elif is_open:
                if abs(current_z) < self.params['zscore_exit']:
                    signals[pair] = 0.0  # Signal to close the position

        return signals