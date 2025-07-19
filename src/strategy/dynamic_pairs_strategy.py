import pandas as pd
from typing import Dict, Tuple, List

from .base import BaseStrategy
from src.analysis.cointegration import find_cointegrated_pairs
from src.models.statistical import StatisticalModel


class DynamicPairsStrategy(BaseStrategy):
    """
    A dynamic pairs trading strategy that periodically re-evaluates the universe
    of assets to find the best trading pairs based on current market conditions.
    """

    def __init__(self,
                 name="DynamicPairsStrategy",
                 reselection_period: int = 60,  # Days between pair re-evaluation
                 lookback_window: int = 60,
                 zscore_entry: float = 2.0,
                 zscore_exit: float = 0.5,
                 max_pairs: int = 10):

        super().__init__(name,
                         reselection_period=reselection_period,
                         lookback_window=lookback_window,
                         zscore_entry=zscore_entry,
                         zscore_exit=zscore_exit,
                         max_pairs=max_pairs)

        self.stat_model = StatisticalModel()
        self.tradeable_pairs: List[Tuple[str, str]] = []
        self.last_reselection_date = None

    def _should_reselect_pairs(self, current_date: pd.Timestamp) -> bool:
        """Checks if it's time to re-evaluate and select new pairs."""
        if self.last_reselection_date is None:
            return True
        return (current_date - self.last_reselection_date).days >= self.params['reselection_period']

    def generate_signals(self,
                         current_data_window: pd.DataFrame,
                         portfolio_context: Dict) -> Dict[Tuple[str, str], float]:
        """
        Dynamically selects pairs if needed, then generates trading signals.
        """
        current_date = current_data_window.index[-1]
        open_positions = portfolio_context['positions']
        open_pairs = {tuple(sorted((p.symbol, next((s for s in p.symbol if s != p.symbol), None)))) for p in
                      open_positions}  # A bit complex, needs pair info in Position object

        signals = {}

        if self._should_reselect_pairs(current_date):
            self.last_reselection_date = current_date

            pairs_found = find_cointegrated_pairs(current_data_window)
            if pairs_found:
                new_pairs = [tuple(p.values())[:2] for p in pairs_found[:self.params['max_pairs']]]

                pairs_to_close = open_pairs - set(map(tuple, map(sorted, new_pairs)))
                for pair in pairs_to_close:
                    signals[pair] = 0.0

                self.tradeable_pairs = new_pairs
            else:  # Close all positions if no pairs are found
                for pair in open_pairs:
                    signals[pair] = 0.0

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

            if z_score.empty or pd.isna(z_score.iloc[-1]): continue
            current_z = z_score.iloc[-1]

            is_open = pair in open_pairs

            if not is_open:
                if current_z > self.params['zscore_entry']:
                    signals[pair] = -1.0
                elif current_z < -self.params['zscore_entry']:
                    signals[pair] = 1.0
            elif is_open:
                if abs(current_z) < self.params['zscore_exit']:
                    signals[pair] = 0.0

        return signals