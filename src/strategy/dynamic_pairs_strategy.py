import pandas as pd
import numpy as np  # Import numpy for np.nan and general numeric ops
from typing import Dict, Tuple, List, Set, Optional

from .base import BaseStrategy
from src.analysis.cointegration import find_cointegrated_pairs  # Assuming this is the correct path
from src.models.statistical import StatisticalModel  # Assuming this is the correct path
from config.logging_config import logger  # Assuming logger is configured


class DynamicPairsStrategy(BaseStrategy):
    """
    A dynamic pairs trading strategy that periodically re-evaluates the universe
    of assets to find the best trading pairs based on current market conditions.
    """

    def __init__(self,
                 name: str = "DynamicPairsStrategy",
                 reselection_period: int = 60,  # Days between pair re-evaluation
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
                         reselection_period=reselection_period,
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
        # Stores the current list of tradeable pairs. This list is updated dynamically.
        self.tradeable_pairs: List[Tuple[str, str]] = []
        self.last_reselection_date: Optional[pd.Timestamp] = None  # Tracks when pairs were last reselected

        logger.info(f"{self.name} initialized with reselection_period={reselection_period}, "
                    f"lookback_window={lookback_window}, zscore_entry={zscore_entry}, "
                    f"zscore_exit={zscore_exit}, max_pairs={max_pairs}.")

    def fit(self, historical_data: pd.DataFrame):
        """
        Performs an initial pair selection based on the provided historical data.
        This method is called once at the start of the backtest.

        Args:
            historical_data (pd.DataFrame): The full historical data available for initial fitting.
                                            Used to establish the first set of tradeable pairs.
        """
        logger.info(f"Initial fitting for {self.name}: performing first pair selection...")

        # Use the initial historical data for the very first reselection
        # This sets up the initial tradeable_pairs and last_reselection_date
        self._reselect_pairs(historical_data)

        self.is_fitted = True
        logger.info(f"Initial fitting of {self.name} complete. {len(self.tradeable_pairs)} tradeable pairs identified.")

    def _should_reselect_pairs(self, current_date: pd.Timestamp) -> bool:
        """
        Checks if it's time to re-evaluate and select new pairs based on the reselection period.

        Args:
            current_date (pd.Timestamp): The current date in the backtest.

        Returns:
            bool: True if pair reselection should occur, False otherwise.
        """
        if self.last_reselection_date is None:
            # This should ideally be set in fit(), but as a fallback
            logger.warning("last_reselection_date is None. Forcing reselection on first call.")
            return True

        # Calculate the number of days since the last reselection
        days_since_last_reselection = (current_date - self.last_reselection_date).days

        should_reselect = days_since_last_reselection >= self.params['reselection_period']
        if should_reselect:
            logger.info(
                f"Reselection period met. {days_since_last_reselection} days since last reselection on {self.last_reselection_date}. Current date: {current_date}.")
        else:
            logger.debug(
                f"No reselection needed yet. {days_since_last_reselection} days since last reselection. Current date: {current_date}.")
        return should_reselect

    def _reselect_pairs(self, data_for_reselection: pd.DataFrame):
        """
        Performs the actual pair reselection process.

        Args:
            data_for_reselection (pd.DataFrame): The historical data window to use
                                                 for finding new cointegrated pairs.
        """
        logger.info(f"Executing pair reselection based on data ending {data_for_reselection.index[-1]}.")

        # Use parameters from __init__ for find_cointegrated_pairs
        pairs_found = find_cointegrated_pairs(
            prices=data_for_reselection,
            p_threshold=self.params['coint_p_threshold'],
            min_half_life=self.params['coint_min_half_life'],
            max_half_life=self.params['coint_max_half_life'],
            integration_test_min_obs=self.params['coint_integration_test_min_obs'],
            adf_regression_type=self.params['coint_adf_regression_type']
        )

        if not pairs_found:
            logger.warning(
                "No cointegrated pairs found during reselection period. Current tradeable pairs will be cleared.")
            self.tradeable_pairs = []
            return  # No new pairs to set, so just return

        # Select the top 'max_pairs' and ensure correct key access
        new_tradeable_pairs = [
            (p['asset1'], p['asset2'])
            for p in pairs_found[:self.params['max_pairs']]
        ]

        logger.info(f"Reselected {len(new_tradeable_pairs)} pairs: {new_tradeable_pairs}")
        self.tradeable_pairs = new_tradeable_pairs

    def generate_signals(self,
                         current_data_window: pd.DataFrame,
                         portfolio_context: Dict) -> Dict[Tuple[str, str], float]:
        """
        Dynamically selects pairs if needed, then generates trading signals.

        Args:
            current_data_window (pd.DataFrame): A rolling window of historical data
                                                up to the current time point. Expected to
                                                contain price data for all potentially tradeable assets.
            portfolio_context (Dict): The current state of the portfolio, typically including:
                                      - 'cash': current cash balance.
                                      - 'positions': a list of objects or dicts representing open positions.
                                                     Each position object should ideally have 'symbol'
                                                     and allow inference of the pair it belongs to.
                                      - 'open_pair_symbols': (Suggested) A set of tuples (asset1, asset2)
                                                             representing currently open pairs for easier lookup.

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

        current_date = current_data_window.index[-1]
        signals = {}

        # 1. Determine open pairs from portfolio context
        # This part requires robust handling of `portfolio_context['positions']`
        # Assuming `portfolio_context` will contain `open_pair_symbols` for convenience in a real backtester.
        # If not, you'd need to infer from individual asset positions (e.g., if AAPL is long and MSFT is short, assume (AAPL, MSFT) pair is open).
        # For this template, let's assume `portfolio_context` can provide clear `open_pair_symbols`.
        open_pairs: Set[Tuple[str, str]] = portfolio_context.get('open_pair_symbols', set())
        # Example if you only have individual positions:
        # open_asset_symbols = {pos.symbol for pos in portfolio_context.get('positions', [])}
        # You'd then loop through self.tradeable_pairs and check if both assets are in open_asset_symbols
        # and in the correct long/short configuration, which is much harder to infer generically.

        # 2. Reselect pairs if the reselection period has been met
        if self._should_reselect_pairs(current_date):
            previous_tradeable_pairs = set(self.tradeable_pairs)  # Capture current pairs before reselection
            self._reselect_pairs(current_data_window)  # Reselect pairs based on current window
            self.last_reselection_date = current_date  # Update the last reselection date

            current_tradeable_pairs_set = set(self.tradeable_pairs)

            # Identify pairs that are no longer tradeable and should be closed
            pairs_to_close = previous_tradeable_pairs - current_tradeable_pairs_set
            for pair in pairs_to_close:
                if pair in open_pairs:  # Only close if actually open
                    signals[pair] = 0.0  # Signal to close the position
                    logger.info(f"Signal for {pair}: CLOSE POSITION (Pair no longer tradeable after reselection).")
                else:
                    logger.debug(f"Pair {pair} is no longer tradeable but not currently open.")

            # If no new pairs are found, ensure all open positions are closed
            if not self.tradeable_pairs:
                logger.info("No tradeable pairs found after reselection. Signaling to close all open pair positions.")
                for pair in open_pairs:
                    signals[pair] = 0.0

        # 3. Generate signals for the currently active tradeable pairs
        for pair in self.tradeable_pairs:
            asset1, asset2 = pair

            # Check if current_data_window contains data for both assets in the pair
            if asset1 not in current_data_window.columns or asset2 not in current_data_window.columns:
                logger.warning(f"Skipping pair {pair}: Not all assets found in current_data_window. "
                               f"Available: {current_data_window.columns.tolist()}.")
                signals[pair] = 0.0  # No action if data missing
                continue

            # Ensure enough data for rolling calculations
            if len(current_data_window) < self.params['lookback_window']:
                logger.debug(f"Skipping pair {pair}: current_data_window length ({len(current_data_window)}) "
                             f"is less than lookback_window ({self.params['lookback_window']}).")
                signals[pair] = 0.0
                continue

            # Calculate rolling hedge ratio, spread, and z-score
            try:
                # Calculate rolling hedge ratio
                hedge_ratio_series = self.stat_model.calculate_rolling_hedge_ratio(
                    current_data_window[asset1],
                    current_data_window[asset2],
                    window=self.params['lookback_window']
                )

                # Use the last valid hedge ratio. If none, skip.
                if hedge_ratio_series.empty or pd.isna(hedge_ratio_series.iloc[-1]):
                    logger.debug(
                        f"Skipping pair {pair}: Rolling hedge ratio is NaN or empty for the current window. Insufficient data or constant series.")
                    signals[pair] = 0.0
                    continue
                current_hedge_ratio = hedge_ratio_series.iloc[-1]

                # Calculate spread
                spread_series = self.stat_model.calculate_spread(
                    current_data_window[asset1],
                    current_data_window[asset2],
                    current_hedge_ratio
                )

                # Calculate z-score
                z_score_series = self.stat_model.calculate_zscore(
                    spread_series,
                    window=self.params['lookback_window']
                )

                # Get the latest z-score. If NaN or empty, skip.
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

            is_pair_open = pair in open_pairs  # Check if this specific pair is open

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
                    # No signal is generated for holding, implying the backtester maintains the existing position.
                    logger.debug(f"Signal for {pair}: HOLD POSITION (Z-score {current_z:.2f} outside exit threshold)")

        return signals