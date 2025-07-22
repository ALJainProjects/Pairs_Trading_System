import pandas as pd
from typing import Dict, Tuple, List, Optional

from .base import BaseStrategy
from src.analysis.cointegration import find_cointegrated_pairs  # Assuming this is the correct path
from src.models.statistical import StatisticalModel  # Assuming this is the correct path
from config.logging_config import logger  # Assuming logger is configured


class StaticPairsStrategy(BaseStrategy):
    """
    A statistical arbitrage pairs trading strategy.

    This strategy identifies cointegrated pairs during an initial 'fit' phase
    and then generates trading signals based on the mean reversion of their spread's z-score.
    """

    def __init__(self,
                 name: str = "StaticPairsStrategy",
                 lookback_window: int = 60,
                 zscore_entry: float = 2.0,
                 zscore_exit: float = 0.5,
                 max_pairs: int = 10,
                 # Parameters for find_cointegrated_pairs
                 coint_p_threshold: float = 0.05,
                 coint_min_half_life: Optional[int] = 5,
                 coint_max_half_life: Optional[int] = 252,
                 coint_integration_test_min_obs: int = 20,
                 coint_adf_regression_type: str = 'c'
                 ):

        super().__init__(name,
                         lookback_window=lookback_window,
                         zscore_entry=zscore_entry,
                         zscore_exit=zscore_exit,
                         max_pairs=max_pairs,
                         # Pass cointegration parameters to self.params as well
                         coint_p_threshold=coint_p_threshold,
                         coint_min_half_life=coint_min_half_life,
                         coint_max_half_life=coint_max_half_life,
                         coint_integration_test_min_obs=coint_integration_test_min_obs,
                         coint_adf_regression_type=coint_adf_regression_type
                         )

        self.stat_model = StatisticalModel()
        # Stores the actual list of tradeable pairs after fitting.
        # Format: List[Tuple[str, str]]
        self.tradeable_pairs: List[Tuple[str, str]] = []
        logger.info(f"{self.name} initialized with lookback_window={lookback_window}, "
                    f"zscore_entry={zscore_entry}, zscore_exit={zscore_exit}, max_pairs={max_pairs}.")

    def fit(self, historical_data: pd.DataFrame):
        """
        Finds the top N cointegrated pairs from the provided historical data.
        These pairs will be stored and used for generating signals throughout the backtest.

        Args:
            historical_data (pd.DataFrame): DataFrame of historical asset prices,
                                            where columns are asset symbols and index is datetime.
        """
        logger.info(f"Fitting {self.name}: searching for cointegrated pairs...")

        # Extract parameters for find_cointegrated_pairs from self.params
        pairs_found = find_cointegrated_pairs(
            prices=historical_data,
            p_threshold=self.params['coint_p_threshold'],
            min_half_life=self.params['coint_min_half_life'],
            max_half_life=self.params['coint_max_half_life'],
            integration_test_min_obs=self.params['coint_integration_test_min_obs'],
            adf_regression_type=self.params['coint_adf_regression_type']
        )

        if not pairs_found:
            logger.error("No cointegrated pairs found in the provided historical data based on criteria.")
            raise ValueError("No cointegrated pairs found in the provided historical data.")

        # Select the top 'max_pairs' and extract only the asset tuple
        # Ensure correct key access ('asset1', 'asset2') for robustness
        self.tradeable_pairs = [
            (p['asset1'], p['asset2'])
            for p in pairs_found[:self.params['max_pairs']]
        ]

        self.is_fitted = True
        logger.info(
            f"Fitting complete. {len(self.tradeable_pairs)} tradeable pairs identified for {self.name}: {self.tradeable_pairs}.")

    def generate_signals(self,
                         current_data_window: pd.DataFrame,
                         portfolio_context: Dict) -> Dict[Tuple[str, str], float]:
        """
        Generates trading signals for the statically defined pairs based on spread's z-score.

        Args:
            current_data_window (pd.DataFrame): A rolling window of historical data
                                                up to the current time point. Expected to
                                                contain price data for the assets in tradeable_pairs.
            portfolio_context (Dict): The current state of the portfolio, typically including:
                                      - 'cash': current cash balance.
                                      - 'positions': a list of objects or dicts representing open positions.
                                                     Each position should ideally have 'symbol' and 'direction'.

        Returns:
            Dict[Tuple[str, str], float]: A dictionary where keys are asset pairs (e.g., ('AAPL', 'MSFT'))
                                          and values are target positions for the spread:
                                          - -1.0: Short the spread (sell Asset1, buy Asset2)
                                          - 1.0: Long the spread (buy Asset1, sell Asset2)
                                          - 0.0: Close any existing position for the pair or no action.
        """
        if not self.is_fitted:
            logger.error("Strategy has not been fitted. Call fit() before running the backtest.")
            raise RuntimeError("Strategy has not been fitted. Call fit() before running the backtest.")

        signals = {}

        # Extract symbols from the portfolio's open positions for quick lookup
        # Assuming 'positions' in portfolio_context is a list of objects/dicts
        # where each has a 'symbol' attribute/key.
        # This approach assumes individual asset positions. For pairs, it's more complex.
        # A more robust system would track pair positions directly.
        open_positions_symbols = {pos.symbol for pos in portfolio_context.get('positions', [])}

        for pair in self.tradeable_pairs:
            asset1, asset2 = pair

            # Check if current_data_window contains data for both assets in the pair
            if asset1 not in current_data_window.columns or asset2 not in current_data_window.columns:
                logger.warning(f"Skipping pair {pair}: Not all assets found in current_data_window. "
                               f"Available: {current_data_window.columns.tolist()}.")
                signals[pair] = 0.0  # No action if data missing
                continue

            # Check if there's enough data in the window for rolling calculations
            if len(current_data_window) < self.params['lookback_window']:
                logger.debug(f"Skipping pair {pair}: current_data_window length ({len(current_data_window)}) "
                             f"is less than lookback_window ({self.params['lookback_window']}).")
                signals[pair] = 0.0
                continue

            # Calculate rolling hedge ratio, spread, and z-score
            try:
                hedge_ratio_series = self.stat_model.calculate_rolling_hedge_ratio(
                    current_data_window[asset1],
                    current_data_window[asset2],
                    window=self.params['lookback_window']
                )

                # If hedge_ratio_series is empty or all NaN, subsequent calculations will fail.
                # Use the last valid hedge ratio.
                if hedge_ratio_series.empty or pd.isna(hedge_ratio_series.iloc[-1]):
                    logger.debug(
                        f"Skipping pair {pair}: Rolling hedge ratio is NaN or empty for the current window. Insufficient data or constant series.")
                    signals[pair] = 0.0
                    continue

                current_hedge_ratio = hedge_ratio_series.iloc[-1]

                spread_series = self.stat_model.calculate_spread(
                    current_data_window[asset1],
                    current_data_window[asset2],
                    current_hedge_ratio
                )

                z_score_series = self.stat_model.calculate_zscore(
                    spread_series,
                    window=self.params['lookback_window']
                )

                if z_score_series.empty or pd.isna(z_score_series.iloc[-1]):
                    logger.debug(
                        f"Skipping pair {pair}: Z-score is NaN or empty for the current window. Insufficient data or constant spread.")
                    signals[pair] = 0.0
                    continue

                current_z = z_score_series.iloc[-1]
                logger.debug(f"Pair {pair}: Current Z-score = {current_z:.2f}")

            except Exception as e:
                logger.error(
                    f"Error calculating spread/z-score for pair {pair}: {e}. Skipping signal generation for this pair.")
                signals[pair] = 0.0
                continue

            # Determine if a position for this pair is currently open
            # This check is basic; a more advanced system would track pair positions explicitly.
            # Assuming if either asset of the pair is in open_positions_symbols, the pair is considered "open".
            is_pair_open = asset1 in open_positions_symbols or asset2 in open_positions_symbols

            # Entry Logic
            if not is_pair_open:
                if current_z > self.params['zscore_entry']:
                    signals[pair] = -1.0  # Short the spread (Sell Asset1, Buy Asset2)
                    logger.info(
                        f"Signal for {pair}: SHORT SPREAD (Z-score {current_z:.2f} > Entry {self.params['zscore_entry']})")
                elif current_z < -self.params['zscore_entry']:
                    signals[pair] = 1.0  # Long the spread (Buy Asset1, Sell Asset2)
                    logger.info(
                        f"Signal for {pair}: LONG SPREAD (Z-score {current_z:.2f} < -Entry {-self.params['zscore_entry']})")
                else:
                    signals[pair] = 0.0  # No entry signal if not triggered
                    logger.debug(f"Signal for {pair}: NO ENTRY (Z-score {current_z:.2f} within entry thresholds)")

            # Exit Logic (if a position is currently open for this pair)
            else:
                # If Z-score reverts towards zero (within exit threshold)
                if abs(current_z) < self.params['zscore_exit']:
                    signals[pair] = 0.0  # Signal to close the position (flat)
                    logger.info(
                        f"Signal for {pair}: CLOSE POSITION (Z-score {current_z:.2f} within Exit {self.params['zscore_exit']})")
                else:
                    # If position is open but Z-score is not yet within exit threshold, hold the position.
                    # This implies no new signal, which the backtester interprets as "hold current position".
                    # However, explicitly signaling "hold" can make logic clearer for complex backtesters.
                    # For a basic backtester expecting 0.0 for close and anything else for entry/hold:
                    # you might return the current position direction, or just omit the pair.
                    # For this template, if no explicit close/entry, we don't add to signals dict,
                    # which implies holding the position for that pair.
                    logger.debug(f"Signal for {pair}: HOLD POSITION (Z-score {current_z:.2f} outside exit threshold)")
                    # If you explicitly need to send a 'hold' signal, you'd need the current direction:
                    # current_pair_direction = self._get_current_pair_direction(pair, portfolio_context)
                    # signals[pair] = current_pair_direction # Requires `_get_current_pair_direction`

        return signals