import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
import logging
from streamlit_system.components.strategy_builder import MultiPairTradingSystem, PairModel, fill_missing_values

logger = logging.getLogger(__name__)

class DynamicPairTradingSystem(MultiPairTradingSystem):
    """Enhanced trading system with dynamic pair selection and management"""

    def __init__(
            self,
            prices: pd.DataFrame,
            initial_pairs: List[Tuple[str, str]] = None,
            initial_capital: float = 1500000,
            window_size: int = 90,
            threshold: float = 2.0,
            transaction_cost_bps: float = 1,
            stop_loss_pct: float = 0.05,
            capital_utilization: float = 0.8,
            max_holding_period: int = 30,
            profit_target_pct: float = 0.05,
            loss_limit_pct: float = 0.03,
            capital_reallocation_freq: int = 60,
            max_active_pairs: int = 15,  # Maximum number of pairs to trade
            pair_evaluation_freq: int = 20,  # Check pairs every 20 days
            universe_reevaluation_freq: int = 60,  # Find new pairs every 60 days
            min_cointegration_pvalue: float = 0.05,  # Maximum p-value for cointegration
            min_correlation: float = 0.6,  # Minimum correlation threshold
            lookback_window: int = 252,  # Window for stability tests
            volatility_adjustment_factor: float = 1.5,  # How much to adjust allocation in volatile periods
            min_data_points: int = 252  # Minimum data history needed for analysis
    ):
        """
        Initialize the dynamic pair trading system with all parameters.

        Args:
            prices: DataFrame with price data for all available symbols
            initial_pairs: Optional list of pairs to start with
            min_data_points: Minimum number of data points needed for pair analysis
        """
        # Save universe-specific parameters
        self.all_symbols = prices.columns.tolist()
        self.max_active_pairs = max_active_pairs
        self.pair_evaluation_freq = pair_evaluation_freq
        self.universe_reevaluation_freq = universe_reevaluation_freq
        self.min_cointegration_pvalue = min_cointegration_pvalue
        self.min_correlation = min_correlation
        self.lookback_window = lookback_window
        self.volatility_adjustment_factor = volatility_adjustment_factor
        self.min_data_points = min_data_points

        # Save these parameters explicitly so they can be accessed in this class
        self.window_size = window_size
        self.threshold = threshold
        self.transaction_cost_bps = transaction_cost_bps
        self.stop_loss_pct = stop_loss_pct
        self.capital_utilization = capital_utilization
        self.max_holding_period = max_holding_period
        self.profit_target_pct = profit_target_pct
        self.loss_limit_pct = loss_limit_pct
        self.capital_reallocation_freq = capital_reallocation_freq

        # Tracking variables
        self.last_pair_evaluation = None
        self.last_universe_evaluation = None
        self.pair_quality_metrics = {}  # Store quality metrics for each pair
        self.market_volatility = None
        self.is_high_volatility = False
        self.inactive_pairs = set()  # Pairs we're not trading due to breakdown
        self.all_traded_pairs = set()  # Historical record of all pairs we've traded

        # Generate initial pairs if none provided
        active_pairs = []
        if initial_pairs and len(initial_pairs) > 0:
            # Validate the initial pairs
            active_pairs = [(s1, s2) for s1, s2 in initial_pairs
                            if s1 in self.all_symbols and s2 in self.all_symbols]

        # If we don't have enough valid pairs, find some automatically
        if len(active_pairs) < max_active_pairs:
            logger.info(f"Finding initial pairs from universe of {len(self.all_symbols)} symbols...")
            needed_pairs = max_active_pairs - len(active_pairs)
            new_pairs = self.find_best_pairs(prices, needed_pairs)
            active_pairs.extend(new_pairs)

        # Set pairs to trade
        self.active_pairs = active_pairs

        # Initialize base class with active pairs
        super().__init__(
            pairs=active_pairs,
            prices=prices,
            initial_capital=initial_capital,
            window_size=window_size,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
            stop_loss_pct=stop_loss_pct,
            capital_utilization=capital_utilization,
            max_holding_period=max_holding_period,
            profit_target_pct=profit_target_pct,
            loss_limit_pct=loss_limit_pct,
            capital_reallocation_freq=capital_reallocation_freq
        )

        # Add tracking for market state
        self.market_state_history = []

    def find_best_pairs(self, prices_df: pd.DataFrame, num_pairs: int) -> List[Tuple[str, str]]:
        """
        Find the best pairs to trade based on both correlation and cointegration.

        Args:
            prices_df: DataFrame with price data
            num_pairs: Number of pairs to find

        Returns:
            List of pairs that are both correlated and likely cointegrated
        """
        from statsmodels.tsa.stattools import adfuller, coint
        from itertools import combinations

        logger.info(f"Finding best {num_pairs} pairs from {len(prices_df.columns)} symbols...")

        # Function to check cointegration with Engle-Granger method
        def is_cointegrated(s1, s2, threshold=self.min_cointegration_pvalue):
            try:
                # Ensure minimum data and positive values
                s1_clean = fill_missing_values(s1)
                s2_clean = fill_missing_values(s2)

                # Only proceed if we have enough data
                if len(s1_clean) < self.min_data_points:
                    return False, 1.0

                # Take log of prices for better properties
                s1_log = np.log(s1_clean)
                s2_log = np.log(s2_clean)

                # Test for cointegration using Engle-Granger two-step approach
                result = coint(s1_log, s2_log)
                p_value = result[1]

                # Check residuals for stationarity as a double-check
                model = LinearRegression()
                X = s1_log.values.reshape(-1, 1)
                y = s2_log.values
                model.fit(X, y)

                # Calculate residuals (spread)
                spread = y - model.predict(X)

                # Test residuals for stationarity
                adf_result = adfuller(spread)
                adf_pvalue = adf_result[1]

                # Return the more conservative result
                return (p_value < threshold and adf_pvalue < threshold), max(p_value, adf_pvalue)
            except Exception as e:
                logger.error(f"Error testing cointegration: {e}")
                return False, 1.0

        # Get the most liquid symbols (based on non-NaN data points)
        symbol_liquidity = {}
        for symbol in prices_df.columns:
            symbol_liquidity[symbol] = prices_df[symbol].count()

        # Sort symbols by liquidity and take the top 50 (or whatever is available)
        top_symbols = sorted(symbol_liquidity.keys(),
                             key=lambda s: symbol_liquidity[s], reverse=True)

        # Limit to 50 most liquid symbols to reduce computation
        max_symbols = min(50, len(top_symbols))
        universe = top_symbols[:max_symbols]

        logger.info(f"Selected {len(universe)} most liquid symbols for pair analysis")

        # Calculate correlation matrix for all symbols in the universe
        returns_df = prices_df[universe].pct_change().dropna()
        corr_matrix = returns_df.corr()

        # Keep track of pairs we've analyzed and their metrics
        potential_pairs = []

        # Loop through all possible pairs
        count = 0
        total = len(universe) * (len(universe) - 1) // 2
        logger.info(f"Analyzing {total} potential pairs...")

        for i, j in combinations(range(len(universe)), 2):
            count += 1
            if count % 100 == 0:
                logger.info(f"Processed {count}/{total} pairs...")

            symbol1, symbol2 = universe[i], universe[j]

            # Skip if either symbol has too much missing data
            if (symbol_liquidity[symbol1] < self.min_data_points or
                    symbol_liquidity[symbol2] < self.min_data_points):
                continue

            # Check correlation first (faster calculation)
            correlation = corr_matrix.loc[symbol1, symbol2]

            # Only check cointegration if correlation is high enough
            if abs(correlation) >= self.min_correlation:
                # Get price data
                price1 = prices_df[symbol1]
                price2 = prices_df[symbol2]

                # Check for cointegration
                is_coint, pvalue = is_cointegrated(price1, price2)

                if is_coint:
                    # Save the pair with its metrics
                    potential_pairs.append({
                        'pair': (symbol1, symbol2),
                        'correlation': abs(correlation),
                        'pvalue': pvalue,
                        'score': abs(correlation) / (pvalue + 0.001)  # Combined score
                    })

        # Sort by combined score (higher is better)
        potential_pairs.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Found {len(potential_pairs)} qualified pairs")

        # Return pairs already in active_pairs + top new pairs
        existing_pairs = set(self.active_pairs if hasattr(self, 'active_pairs') else [])

        # Filter out pairs we've already tried and had problems with
        new_pairs = [p['pair'] for p in potential_pairs
                     if p['pair'] not in existing_pairs
                     and p['pair'] not in self.inactive_pairs
                     and (p['pair'][1], p['pair'][0]) not in self.inactive_pairs]

        needed = min(num_pairs, len(new_pairs))
        return new_pairs[:needed]

    def check_pair_stability(self, pair: Tuple[str, str], current_date) -> bool:
        """
        Check if a pair is still stable (cointegrated/correlated).

        Args:
            pair: Tuple containing the pair symbols
            current_date: Current date to define the lookback window

        Returns:
            bool: True if the pair is still stable
        """
        from statsmodels.tsa.stattools import adfuller

        symbol_x, symbol_y = pair
        model = self.pair_models[pair]

        try:
            # Get the lookback window indices
            current_idx = self.prices.index.get_loc(current_date)
            start_idx = max(0, current_idx - self.lookback_window)

            # Get price data for the window
            window_data = self.prices.iloc[start_idx:current_idx]

            # Check if we have enough data
            if len(window_data) < 60:  # Need at least 60 data points
                return True  # Assume stable if not enough data to check

            # Get the price series
            price_x = window_data[symbol_x]
            price_y = window_data[symbol_y]

            # Check correlation
            x_returns = price_x.pct_change().dropna()
            y_returns = price_y.pct_change().dropna()

            # Calculate recent correlation
            correlation = x_returns.corr(y_returns)

            # Fail if correlation is too low
            if abs(correlation) < self.min_correlation * 0.8:  # Allow some degradation
                logger.info(f"Pair {pair} correlation breakdown: {abs(correlation):.3f}")
                return False

            # Get the latest regression parameters
            beta = model.beta
            alpha = model.alpha

            if beta is None or alpha is None:
                logger.warning(f"No regression parameters for pair {pair}")
                return True  # Can't check without parameters

            # Calculate spread using current parameters
            price_x_clean = fill_missing_values(price_x)
            price_y_clean = fill_missing_values(price_y)

            # Log prices
            log_x = np.log(price_x_clean.apply(lambda x: max(x, 0.01)))
            log_y = np.log(price_y_clean.apply(lambda x: max(x, 0.01)))

            # Calculate spread
            spread = log_y - (beta * log_x + alpha)

            # Test for stationarity
            adf_result = adfuller(spread)
            p_value = adf_result[1]

            # Pair is stable if spread is still stationary
            if p_value > self.min_cointegration_pvalue * 2:  # Allow some degradation
                logger.info(f"Pair {pair} cointegration breakdown: {p_value:.3f}")
                return False

            # Check for structural break
            # Split the spread into two halves
            mid_point = len(spread) // 2
            spread1 = spread[:mid_point]
            spread2 = spread[mid_point:]

            # Calculate mean and std for both halves
            mean1, std1 = np.mean(spread1), np.std(spread1)
            mean2, std2 = np.mean(spread2), np.std(spread2)

            # Check if the means or stds have shifted significantly
            mean_shift = abs(mean2 - mean1) / std1
            vol_ratio = std2 / std1 if std1 > 0 else 1.0

            # If large shifts, consider a structural break
            if mean_shift > 1.5 or vol_ratio > 2.0 or vol_ratio < 0.5:
                logger.info(f"Pair {pair} structural break: mean_shift={mean_shift:.2f}, vol_ratio={vol_ratio:.2f}")
                return False

            # Store pair quality metrics for allocation decisions
            self.pair_quality_metrics[pair] = {
                'correlation': abs(correlation),
                'pvalue': p_value,
                'mean_shift': mean_shift,
                'vol_ratio': vol_ratio,
                'score': abs(correlation) / (p_value + 0.001)
            }

            return True

        except Exception as e:
            logger.error(f"Error checking stability for pair {pair}: {e}")
            return True  # Assume stable if there's an error

    def check_market_volatility(self, current_date):
        """
        Assess overall market volatility to adjust strategy.

        Args:
            current_date: Current date for the window

        Returns:
            is_volatile: Boolean indicating if market is in high volatility
        """
        try:
            # Get the lookback window indices
            current_idx = self.prices.index.get_loc(current_date)
            short_window = 20  # 1 month for recent volatility
            long_window = 252  # 1 year for baseline volatility

            # Get data for both windows
            short_idx = max(0, current_idx - short_window)
            long_idx = max(0, current_idx - long_window)

            short_data = self.prices.iloc[short_idx:current_idx]
            long_data = self.prices.iloc[long_idx:current_idx]

            # Calculate volatility (average of all symbols)
            short_vol = []
            long_vol = []

            # Sample a subset of symbols to make calculation faster
            symbols_to_check = random.sample(self.all_symbols, min(20, len(self.all_symbols)))

            for symbol in symbols_to_check:
                if symbol in short_data.columns:
                    # Calculate returns
                    short_returns = short_data[symbol].pct_change().dropna()
                    long_returns = long_data[symbol].pct_change().dropna()

                    # Calculate volatility (std of returns)
                    if len(short_returns) > 0:
                        short_vol.append(short_returns.std())
                    if len(long_returns) > 0:
                        long_vol.append(long_returns.std())

            # Calculate average volatilities
            avg_short_vol = np.mean(short_vol) if short_vol else 0
            avg_long_vol = np.mean(long_vol) if long_vol else 0

            # Volatility ratio (current vs historical)
            vol_ratio = avg_short_vol / avg_long_vol if avg_long_vol > 0 else 1.0

            # Store for tracking
            self.market_volatility = vol_ratio

            # Track market state
            self.market_state_history.append({
                'date': current_date,
                'volatility_ratio': vol_ratio,
                'short_vol': avg_short_vol,
                'long_vol': avg_long_vol,
                'is_high_volatility': vol_ratio > 1.5
            })

            # Determine if we're in a high volatility regime
            is_volatile = vol_ratio > 1.5  # 50% higher than baseline

            # Update state
            if is_volatile != self.is_high_volatility:
                self.is_high_volatility = is_volatile
                logger.info(f"Market volatility state change: {'HIGH' if is_volatile else 'NORMAL'} volatility")
                logger.info(f"Volatility ratio: {vol_ratio:.2f}")

            return is_volatile

        except Exception as e:
            logger.error(f"Error checking market volatility: {e}")
            return False

    def adjust_for_volatility(self):
        """
        Adjust strategy parameters based on current volatility regime.
        """
        if self.is_high_volatility:
            # In high volatility, we want to:
            # 1. Increase entry thresholds (wait for bigger deviations)
            # 2. Tighten stop losses
            # 3. Reduce holding periods
            # 4. Target smaller profits
            new_params = {
                'threshold': self.pair_models[self.pairs[0]].base_threshold * 1.5,
                'stop_loss_pct': 0.03,  # Tighter stop loss
                'max_holding_period': 15,  # Shorter holdings
                'profit_target_pct': 0.03,  # Smaller profit targets
                'loss_limit_pct': 0.02,  # More aggressive loss limits
                'capital_utilization': 0.5  # Reduce capital at risk
            }
        else:
            # In normal volatility, reset to default parameters
            new_params = {
                'threshold': self.pair_models[self.pairs[0]].base_threshold,
                'stop_loss_pct': 0.05,
                'max_holding_period': 30,
                'profit_target_pct': 0.05,
                'loss_limit_pct': 0.03,
                'capital_utilization': 0.8
            }

        # Update all models
        self.update_all_model_params(new_params)

    def manage_pair_rotation(self, current_date):
        """
        Add, remove, and rotate pairs based on stability checks and performance.

        Args:
            current_date: Current date for evaluation
        """
        # 1. Evaluate existing pairs for stability
        unstable_pairs = []
        for pair in self.pairs:
            # Skip if model has an active position
            if self.pair_models[pair].active:
                continue

            # Check if pair is still stable
            is_stable = self.check_pair_stability(pair, current_date)

            if not is_stable:
                unstable_pairs.append(pair)

        # 2. Track unstable pairs
        if unstable_pairs:
            logger.info(f"Found {len(unstable_pairs)} unstable pairs to remove")
            for pair in unstable_pairs:
                self.inactive_pairs.add(pair)
                self.inactive_pairs.add((pair[1], pair[0]))  # Add reversed pair too

        # 3. Calculate how many new pairs we need
        pairs_to_remove = []
        for pair in unstable_pairs:
            # Only remove if no active position
            if not self.pair_models[pair].active:
                pairs_to_remove.append(pair)

        # 4. Find replacements if needed
        if len(pairs_to_remove) > 0:
            # Remove unstable pairs
            logger.info(f"Removing {len(pairs_to_remove)} pairs")
            self.remove_pairs(pairs_to_remove)

            # Calculate how many new pairs we need
            current_count = len(self.pairs)
            needed_pairs = min(len(pairs_to_remove), self.max_active_pairs - current_count)

            if needed_pairs > 0:
                # Find and add new pairs
                logger.info(f"Finding {needed_pairs} new replacement pairs")
                new_pairs = self.find_best_pairs(self.prices, needed_pairs)
                self.add_pairs(new_pairs)

    def full_universe_evaluation(self, current_date):
        """
        Completely re-evaluate the universe for the best pairs.

        Args:
            current_date: Current date for evaluation
        """
        logger.info(f"Performing full universe evaluation on {current_date}")

        # 1. Check performance of current pairs
        performance_scores = {}
        for pair in self.pairs:
            model = self.pair_models[pair]
            metrics = model.get_metrics()

            # Only include pairs with enough trades
            if metrics.get('num_trades', 0) >= 3:
                # Calculate a performance score
                win_rate = metrics.get('win_rate', 50)
                profit_factor = metrics.get('profit_factor', 1.0)
                max_drawdown = metrics.get('max_drawdown', 0)

                # Penalize high drawdowns
                drawdown_factor = 1.0 - (max_drawdown / 100)
                drawdown_factor = max(0.5, drawdown_factor)  # Cap the penalty

                # Combined score
                score = (win_rate / 100) * profit_factor * drawdown_factor
                performance_scores[pair] = score

        # 2. Find underperforming pairs (but only consider inactive ones for removal)
        underperforming = []
        for pair, score in performance_scores.items():
            if score < 0.8 and not self.pair_models[pair].active:
                underperforming.append(pair)

        # 3. Get the best possible new pairs from the universe
        potential_new_count = min(5, self.max_active_pairs - len(self.pairs) + len(underperforming))
        if potential_new_count > 0:
            potential_new_pairs = self.find_best_pairs(self.prices, potential_new_count)

            # 4. Remove underperforming pairs
            if underperforming:
                logger.info(f"Removing {len(underperforming)} underperforming pairs")
                self.remove_pairs(underperforming)

            # 5. Add new pairs
            if potential_new_pairs:
                logger.info(f"Adding {len(potential_new_pairs)} new pairs from universe evaluation")
                self.add_pairs(potential_new_pairs)

    def add_pairs(self, new_pairs: List[Tuple[str, str]]):
        """
        Add new pairs to the system

        Args:
            new_pairs: List of pairs to add
        """
        for pair in new_pairs:
            if pair not in self.pairs and (pair[1], pair[0]) not in self.pairs:
                logger.info(f"Adding new pair: {pair[0]}-{pair[1]}")

                # Create a new model for this pair
                model = PairModel(
                    pair=pair,
                    prices=self.prices,
                    initial_capital=self.capital_per_pair,
                    window_size=self.window_size,
                    threshold=self.pair_models[self.pairs[0]].base_threshold if self.pairs else 2.0,
                    transaction_cost_bps=self.transaction_cost_bps,
                    stop_loss_pct=self.stop_loss_pct,
                    capital_utilization=self.capital_utilization,
                    max_holding_period=self.max_holding_period,
                    profit_target_pct=self.profit_target_pct,
                    loss_limit_pct=self.loss_limit_pct
                )

                # Add to our pairs list and model dictionary
                self.pairs.append(pair)
                self.pair_models[pair] = model

                # Record it was traded
                self.all_traded_pairs.add(pair)

    def remove_pairs(self, pairs_to_remove: List[Tuple[str, str]]):
        """
        Remove pairs from the system

        Args:
            pairs_to_remove: List of pairs to remove
        """
        for pair in pairs_to_remove:
            if pair in self.pairs:
                logger.info(f"Removing pair: {pair[0]}-{pair[1]}")

                # Close any open positions
                model = self.pair_models[pair]

                # Get current prices for closing positions
                latest_date = self.prices.index[-1]
                current_prices = {}
                for symbol in model.positions.keys():
                    price = self.prices.loc[latest_date, symbol]
                    if pd.notnull(price) and price > 0:
                        current_prices[symbol] = price
                    else:
                        # Use last valid price
                        last_valid = self.prices[symbol].dropna()
                        current_prices[symbol] = last_valid.iloc[-1] if len(last_valid) > 0 else 0.01

                # Close position if active
                if model.active:
                    model.close_position(current_prices, latest_date, "pair removed")

                # Remove from our collections
                self.pairs.remove(pair)

                # Don't delete the model so we keep the history

    def update(self, date, current_prices: Dict[str, float]):
        """
        Enhanced update method that manages pair rotation and volatility adjustments

        Args:
            date: Current date
            current_prices: Dictionary of current prices for all symbols
        """
        # Check if we need to evaluate pairs
        if (self.last_pair_evaluation is None or
                (date - self.last_pair_evaluation).days >= self.pair_evaluation_freq):
            logger.info(f"\nEvaluating pairs on {date}")
            self.manage_pair_rotation(date)
            self.last_pair_evaluation = date

        # Check if we need to do a full universe evaluation
        if (self.last_universe_evaluation is None or
                (date - self.last_universe_evaluation).days >= self.universe_reevaluation_freq):
            logger.info(f"\nPerforming universe evaluation on {date}")
            self.full_universe_evaluation(date)
            self.last_universe_evaluation = date

        # Check market volatility and adjust strategy if needed
        is_volatile = self.check_market_volatility(date)

        # Only adjust strategy parameters if volatility state has changed
        if is_volatile != self.is_high_volatility:
            logger.info(
                f"Volatility state changed from {self.is_high_volatility} to {is_volatile}. Adjusting strategy.")
            self.is_high_volatility = is_volatile
            self.adjust_for_volatility()

        # Check if capital reallocation is needed
        days_since_last_reallocation = 0
        if self.last_reallocation_date is not None:
            days_since_last_reallocation = (date - self.last_reallocation_date).days

        if self.last_reallocation_date is None or days_since_last_reallocation >= self.capital_reallocation_freq:
            self.update_capital_allocation(date)
            self.last_reallocation_date = date

        # Update each pair model with the current data
        total_portfolio_value = 0
        active_pairs = 0

        for pair, model in self.pair_models.items():
            try:
                model.update(date, current_prices)

                # Get latest portfolio value from this model
                if model.portfolio_history:
                    total_portfolio_value += model.portfolio_history[-1]['portfolio_value']
                    active_pairs += 1 if model.portfolio_history[-1]['active'] else 0
            except Exception as e:
                logger.error(f"Error updating model for {pair} on {date}: {e}")
                self.processing_errors += 1

        # Record overall portfolio statistics
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': total_portfolio_value,
            'active_pairs': active_pairs
        })

    def run_backtest(self):
        """Override to ensure proper initialization before running"""
        # Set initial dates to avoid immediate evaluation
        first_date = self.prices.index[0]
        self.last_pair_evaluation = first_date
        self.last_universe_evaluation = first_date

        # Run the base backtest
        super().run_backtest()

    def get_portfolio_metrics(self) -> Dict:
        """Enhanced metrics with pair stability and rotation info"""
        base_metrics = super().get_portfolio_metrics()

        # Add pair rotation metrics
        base_metrics['Pair Rotation'] = {
            'Total Pairs Traded': len(self.all_traded_pairs),
            'Current Active Pairs': len(self.pairs),
            'Inactive Pairs': len(self.inactive_pairs),
            'Average Pair Correlation': np.mean([m.get('correlation', 0)
                                                 for m in self.pair_quality_metrics.values()])
            if self.pair_quality_metrics else 0,
            'Average Pair Score': np.mean([m.get('score', 0)
                                           for m in self.pair_quality_metrics.values()])
            if self.pair_quality_metrics else 0,
        }

        # Add volatility regime metrics
        if self.market_state_history:
            vol_df = pd.DataFrame(self.market_state_history)
            high_vol_days = vol_df[vol_df['is_high_volatility']].shape[0]
            total_days = vol_df.shape[0]

            base_metrics['Volatility Regimes'] = {
                'High Volatility Days': high_vol_days,
                'Normal Volatility Days': total_days - high_vol_days,
                'High Volatility Percentage': (high_vol_days / total_days * 100) if total_days > 0 else 0,
                'Average Volatility Ratio': vol_df['volatility_ratio'].mean(),
                'Max Volatility Ratio': vol_df['volatility_ratio'].max(),
            }

        return base_metrics

    def plot_pair_rotation(self):
        """Create visualization showing pair rotation over time"""
        if not self.portfolio_history:
            logger.warning("No portfolio history to plot pair rotation")
            return

        # Convert all trade history to DataFrame
        all_trades = self.trade_history

        if not all_trades:
            logger.warning("No trade history to plot pair rotation")
            return

        trades_df = pd.DataFrame(all_trades)

        # Get unique pair IDs from trades
        all_pair_ids = trades_df['pair_id'].unique() if 'pair_id' in trades_df.columns else []

        if not all_pair_ids.size:
            logger.warning("No pair IDs found in trade history")
            return

        # Create a timeline of active pairs
        dates = [d['date'] for d in self.portfolio_history]
        date_range = pd.date_range(start=min(dates), end=max(dates), freq='D')

        # Create a dataframe for pair activity
        pair_activity = {}
        for pair_id in all_pair_ids:
            # Filter trades for this pair
            pair_trades = trades_df[trades_df['pair_id'] == pair_id]

            # Create a series of 1s for active days, 0s for inactive
            activity = pd.Series(0, index=date_range)

            # Group by date to handle multiple trades on same day
            daily_trades = pair_trades.groupby('date').size()

            # Set active days as 1
            for date in daily_trades.index:
                activity[date] = 1

            pair_activity[f"{pair_id[0]}-{pair_id[1]}"] = activity

        # Create a dataframe with all pair activity
        if pair_activity:
            activity_df = pd.DataFrame(pair_activity)

            # Plot the activity
            plt.figure(figsize=(14, 10))
            ax = plt.gca()

            # Plot each pair's activity as a horizontal line
            for i, pair_name in enumerate(activity_df.columns):
                active_dates = activity_df.index[activity_df[pair_name] == 1]
                if len(active_dates) > 0:
                    plt.scatter(active_dates, [i] * len(active_dates),
                                marker='|', s=100, label=pair_name)

            # Add pair names to y-axis
            plt.yticks(range(len(activity_df.columns)), activity_df.columns)

            # Format x-axis as dates
            plt.xlabel('Date')
            plt.ylabel('Pair')
            plt.title('Pair Trading Activity Timeline')

            # Add grid for readability
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()
        else:
            logger.warning("No pair activity data to plot")

    def plot_market_volatility(self):
        """Plot market volatility over time with regime changes"""
        if not self.market_state_history:
            logger.warning("No market state history to plot")
            return

        # Convert to DataFrame
        vol_df = pd.DataFrame(self.market_state_history)
        vol_df.set_index('date', inplace=True)

        # Create the plot
        plt.figure(figsize=(14, 6))

        # Plot volatility ratio
        plt.plot(vol_df.index, vol_df['volatility_ratio'],
                 color='blue', label='Volatility Ratio')

        # Add a horizontal line at the threshold
        plt.axhline(y=1.5, color='red', linestyle='--',
                    label='High Volatility Threshold')

        # Shade high volatility regions
        high_vol_regions = vol_df['is_high_volatility']

        # Find the start and end of each high volatility period
        switches = high_vol_regions.diff().fillna(0)
        switch_points = switches[switches != 0].index

        if high_vol_regions.iloc[0]:
            # First point is already in high vol state
            # FIX: Correct the list extension operation
            switch_points_list = [vol_df.index[0]]
            switch_points_list.extend(switch_points)
            switch_points = switch_points_list

        if len(switch_points) % 2 == 1:
            # Add the end date if we're still in a high vol state
            switch_points = list(switch_points)
            switch_points.append(vol_df.index[-1])

        # Shade the high volatility regions
        for i in range(0, len(switch_points), 2):
            if i + 1 < len(switch_points):
                plt.axvspan(switch_points[i], switch_points[i + 1],
                            alpha=0.3, color='red', label='_' * i + 'High Volatility Period')

        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Volatility Ratio')
        plt.title('Market Volatility Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def run_strategy(self):
        """Main method to run the entire strategy"""
        # Run the backtest
        self.run_backtest()

        # Calculate and display portfolio metrics
        metrics = self.get_portfolio_metrics()

        # Print overall performance metrics
        logger.info("\nPortfolio Performance:")
        logger.info("-" * 50)
        for metric, value in metrics['Portfolio Metrics'].items():
            logger.info(f"{metric}: {value:,.2f}" if isinstance(value, (int, float)) else f"{metric}: {value}")

        logger.info("\nTrading Activity:")
        logger.info("-" * 50)
        for metric, value in metrics['Trading Activity'].items():
            logger.info(f"{metric}: {value:,.2f}" if isinstance(value, (int, float)) else f"{metric}: {value}")

        logger.info("\nPair Rotation:")
        logger.info("-" * 50)
        for metric, value in metrics['Pair Rotation'].items():
            logger.info(f"{metric}: {value:,.2f}" if isinstance(value, (int, float)) else f"{metric}: {value}")

        if 'Volatility Regimes' in metrics:
            logger.info("\nVolatility Regimes:")
            logger.info("-" * 50)
            for metric, value in metrics['Volatility Regimes'].items():
                logger.info(f"{metric}: {value:,.2f}" if isinstance(value, (int, float)) else f"{metric}: {value}")

        # Create visualizations
        self.plot_extended_portfolio_overview()
        self.plot_pair_rotation()
        self.plot_market_volatility()
        self.plot_pair_metrics()

        # Plot individual pair analyses
        for pair, model in self.pair_models.items():
            model.plot_pair_analysis()

        return metrics

    def plot_extended_portfolio_overview(self):
        """Enhanced portfolio overview with volatility regions"""
        if not self.portfolio_history:
            logger.warning("No portfolio history available to plot")
            return

        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        # Create a merged dataframe with volatility info if available
        if self.market_state_history:
            vol_df = pd.DataFrame(self.market_state_history)
            vol_df.set_index('date', inplace=True)

            # Align the indices (may have different frequencies)
            # FIX: Use a safer joining method for merging indices
            try:
                merged = portfolio_df.join(vol_df[['is_high_volatility', 'volatility_ratio']], how='left')
            except:
                # Fallback for incompatible indices
                merged = portfolio_df.copy()
                merged['is_high_volatility'] = False
                merged['volatility_ratio'] = 1.0

            # Fill any missing volatility data
            merged['is_high_volatility'].fillna(False, inplace=True)
            merged['volatility_ratio'].fillna(1.0, inplace=True)
        else:
            merged = portfolio_df.copy()
            merged['is_high_volatility'] = False
            merged['volatility_ratio'] = 1.0

        # Plot the enhanced overview
        plt.figure(figsize=(14, 10))

        # Create a colormap for volatility background
        cmap = plt.cm.get_cmap('Reds', 100)

        # Plot portfolio value with volatility-based background
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(merged.index, merged['portfolio_value'],
                 color='blue', linewidth=2, label='Portfolio Value')

        # Add initial capital reference line
        ax1.axhline(y=self.initial_capital, color='black', linestyle='--',
                    label='Initial Capital')

        # Shade high volatility regions
        high_vol_regions = merged['is_high_volatility']

        # Find the start and end of each high volatility period
        if len(high_vol_regions) > 0:
            switches = high_vol_regions.diff().fillna(0)
            switch_points = switches[switches != 0].index

            if high_vol_regions.iloc[0]:
                # First point is already in high vol state
                # FIX: Correct the list extension operation
                switch_points_list = [merged.index[0]]
                switch_points_list.extend(switch_points)
                switch_points = switch_points_list

            if len(switch_points) % 2 == 1:
                # Add the end date if we're still in a high vol state
                switch_points = list(switch_points)
                switch_points.append(merged.index[-1])

            # Shade the high volatility regions
            for i in range(0, len(switch_points), 2):
                if i + 1 < len(switch_points):
                    ax1.axvspan(switch_points[i], switch_points[i + 1],
                                alpha=0.3, color='red', label='_' * i + 'High Volatility Period')

        ax1.set_title('Portfolio Value with Volatility Regime')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot the number of active pairs and volatility ratio
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(merged.index, merged['active_pairs'],
                 color='purple', linewidth=2, label='Active Pairs')

        # Add second y-axis for volatility ratio
        ax3 = ax2.twinx()
        ax3.plot(merged.index, merged['volatility_ratio'],
                 color='red', linewidth=1.5, label='Volatility Ratio')
        ax3.axhline(y=1.5, color='red', linestyle='--',
                    label='Volatility Threshold')

        # Set labels
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Active Pairs', color='purple')
        ax3.set_ylabel('Volatility Ratio', color='red')

        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_pair_metrics(self):
        """Plot quality metrics for pairs over time"""
        if not self.pair_quality_metrics:
            logger.warning("No pair quality metrics to plot")
            return

        # Prepare the data
        pairs = list(self.pair_quality_metrics.keys())
        metrics = ['correlation', 'pvalue', 'score']

        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Extract the metric for each pair
            for pair in pairs:
                pair_name = f"{pair[0]}-{pair[1]}"
                value = self.pair_quality_metrics[pair].get(metric, 0)
                ax.bar(pair_name, value, alpha=0.7)

            ax.set_title(f'Pair {metric.capitalize()}')
            ax.set_ylabel(metric)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Rotate x labels for readability if many pairs
            if len(pairs) > 5:
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()