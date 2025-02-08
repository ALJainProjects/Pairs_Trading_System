from enum import Enum
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import os

from src.strategy.base import BaseStrategy
from src.models.statistical import StatisticalModel
from config.logging_config import logger
from config.settings import DATA_DIR


@dataclass
class RegimeMetrics:
    """
    Track market regime characteristics and metrics.

    Attributes:
        volatility: Rolling volatility measure
        correlation: Average pairwise correlation
        trend_strength: Measure of trending behavior
        regime_type: Classified market regime
        start_date: Beginning of regime period
        end_date: End of regime period (if known)
    """
    volatility: float
    correlation: float
    trend_strength: float
    regime_type: str
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp] = None

class MarketRegimeDetector:
    """
    Detect market regimes using statistical thresholds and indicators.

    Uses rolling volatility, correlation, and trend metrics to classify
    market regimes based on their statistical characteristics.
    """

    def __init__(self, window: int = 63):
        """
        Initialize regime detector.

        Args:
            window: Rolling window size for calculations in days
        """
        self.window = window

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime detection features from price data.

        Args:
            data: DataFrame with columns Date, Symbol, Adj_Close

        Returns:
            DataFrame with regime features:
                - Rolling volatility
                - Average correlation
                - Trend strength
                - Volume profile
        """
        features = pd.DataFrame(index=data['Date'].unique())

        symbol_groups = data.groupby('Symbol')
        returns_by_symbol = {}

        for symbol, group in symbol_groups:
            group_sorted = group.sort_values('Date')
            returns_by_symbol[symbol] = pd.Series(
                group_sorted['Adj_Close'].pct_change().values,
                index=group_sorted['Date']
            )

        returns_matrix = pd.DataFrame(returns_by_symbol).fillna(0)

        features['volatility'] = returns_matrix.std(axis=1).rolling(window=self.window).mean()

        features['correlation'] = returns_matrix.rolling(window=self.window).apply(self._calculate_correlation)

        prices = data.pivot(index='Date', columns='Symbol', values='Adj_Close')
        trends = prices.pct_change().rolling(window=self.window).mean()
        features['trend_strength'] = trends.abs().mean(axis=1)

        volume_profile = data.pivot(index='Date', columns='Symbol', values='Volume')
        features['volume_intensity'] = volume_profile.rolling(window=self.window).mean().mean(axis=1)

        return features.ffill()

    def _calculate_correlation(self, x: pd.DataFrame) -> float:
        """
        Calculate average pairwise correlation from returns matrix.

        Args:
            x: Returns matrix for correlation calculation

        Returns:
            float: Average correlation value
        """
        if len(x.columns) < 2:
            return 0

        corr_matrix = x.corr()
        upper_tri = np.triu(corr_matrix.values, k=1)
        return np.mean(upper_tri[np.triu_indices_from(upper_tri, k=1)])

    def detect_regime(self, features: pd.DataFrame) -> str:
        """
        Detect current market regime using statistical thresholds.

        Args:
            features: DataFrame of calculated regime features

        Returns:
            str: Classified regime type
        """
        try:
            current = features.iloc[-1]

            vol_extreme = features['volatility'].quantile(0.9)
            vol_high = features['volatility'].quantile(0.7)
            vol_low = features['volatility'].quantile(0.3)

            corr_high = features['correlation'].quantile(0.7)
            corr_low = features['correlation'].quantile(0.3)

            trend_strong = features['trend_strength'].quantile(0.7)
            trend_weak = features['trend_strength'].quantile(0.3)

            volume_high = features['volume_intensity'].quantile(0.7)

            return self._classify_regime(
                current['volatility'],
                current['correlation'],
                current['trend_strength'],
                current['volume_intensity'],
                vol_extreme, vol_high, vol_low,
                corr_high, corr_low,
                trend_strong, trend_weak,
                volume_high
            )

        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return 'Unknown'

    def _classify_regime(
            self,
            vol: float,
            corr: float,
            trend: float,
            volume: float,
            vol_extreme: float,
            vol_high: float,
            vol_low: float,
            corr_high: float,
            corr_low: float,
            trend_strong: float,
            trend_weak: float,
            volume_high: float
    ) -> str:
        """
        Classify market regime based on feature thresholds.

        Args:
            vol: Current volatility
            corr: Current correlation
            trend: Current trend strength
            volume: Current volume intensity
            vol_extreme: 90th percentile volatility threshold
            vol_high: 70th percentile volatility threshold
            vol_low: 30th percentile volatility threshold
            corr_high: 70th percentile correlation threshold
            corr_low: 30th percentile correlation threshold
            trend_strong: 70th percentile trend threshold
            trend_weak: 30th percentile trend threshold
            volume_high: 70th percentile volume threshold

        Returns:
            str: Classified regime type
        """
        if vol > vol_extreme and corr > corr_high:
            return 'Crisis'

        if vol > vol_high:
            if corr > corr_high:
                if volume > volume_high:
                    return 'Panic'
                return 'Risk-Off'
            return 'High Volatility'

        if trend > trend_strong:
            if corr > corr_high:
                return 'Strong Trend'
            if vol < vol_low:
                return 'Low Vol Trend'
            return 'Trending'

        if vol < vol_low:
            if corr < corr_low:
                return 'Low Risk'
            if trend < trend_weak:
                return 'Range Bound'
            return 'Quiet'

        if trend < trend_weak and corr < corr_low:
            return 'Rotational'

        return 'Mean Reverting'


@dataclass
class PairStats:
    """
    Stores statistical metrics and position information for a trading pair.

    Attributes:
        hedge_ratio: The beta coefficient used to hedge the pair
        half_life: Mean reversion speed measure in days
        coint_pvalue: Cointegration test p-value
        spread_zscore: Current z-score of the spread
        spread_vol: Volatility of the spread
        correlation: Price correlation between assets
        last_update: Timestamp of last update
        composite_score: Combined ranking score
        position_sizes: Current position sizes for each asset
    """
    hedge_ratio: float
    half_life: float
    coint_pvalue: float
    spread_zscore: float
    spread_vol: float
    correlation: float
    last_update: pd.Timestamp
    composite_score: float = 0.0
    position_sizes: Dict[str, float] = field(default_factory=dict)


class PositionType(Enum):
    """
    Defines allowed position types for the strategy.

    Values:
        LONG_ONLY: Only long positions allowed
        SHORT_ONLY: Only short positions allowed
        MARKET_NEUTRAL: Both long and short positions allowed
    """
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    MARKET_NEUTRAL = "market_neutral"


class EnhancedStatPairsStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy using mean reversion signals.

    This strategy implements a pairs trading approach based on statistical arbitrage principles.
    It monitors the spread between pairs of assets for mean reversion opportunities and
    generates trading signals when the spread deviates significantly from its mean.

    Key features:
    - Cointegration-based pair selection
    - Z-score based entry and exit signals
    - Multiple exit conditions including stop-loss
    - Position sizing based on volatility and correlation
    - Comprehensive trade tracking and analysis

    The strategy can operate in long-only, short-only, or market-neutral modes.
    """

    def __init__(
            self,
            lookback_window: int = 252,
            zscore_entry: float = 2.0,
            zscore_exit: float = 0.5,
            min_half_life: int = 5,
            max_half_life: int = 126,
            max_spread_vol: float = 0.12,
            min_correlation: float = 0.45,
            coint_threshold: float = 0.05,
            max_pairs: int = 5,
            position_size: float = 0.1,
            stop_loss: float = 0.05,  # Increased from 0.02
            trailing_stop: float = 0.03,  # New parameter
            time_stop: int = 21,  # New parameter - minimum holding period
            profit_take: float = 0.1,  # New parameter
            max_drawdown: float = 0.2,
            close_on_signal_flip: bool = False,  # Changed from True
            signal_exit_threshold: float = 0.5,  # Increased from 0.3
            confirmation_periods: int = 3,  # Increased from 2
            close_on_regime_change: bool = True,
            cointegration_windows: List[int] = None,
            min_votes: int = None,
            regime_adaptation: bool = True,
            position_type: PositionType = PositionType.MARKET_NEUTRAL
    ):
        """
        Initialize the pairs trading strategy with specified parameters.

        Args:
            lookback_window: Days of historical data for calculations
            zscore_entry: Z-score threshold for position entry
            zscore_exit: Z-score threshold for position exit
            min_half_life: Minimum mean reversion half-life in days
            max_half_life: Maximum mean reversion half-life in days
            max_spread_vol: Maximum allowed spread volatility
            min_correlation: Minimum required correlation between pairs
            coint_threshold: Maximum p-value for cointegration test
            max_pairs: Maximum number of pairs to trade
            position_size: Base position size as fraction of portfolio
            stop_loss: Stop loss threshold as fraction of position value
            max_drawdown: Maximum allowed drawdown before reducing exposure
            cointegration_windows: List of windows for cointegration testing
            min_votes: Minimum required cointegration votes
            regime_adaptation: Whether to adapt to market regimes
            position_type: Type of positions allowed (long/short/neutral)
            close_on_signal_flip: Whether to close on signal reversal
            signal_exit_threshold: Secondary signal threshold for exits
            confirmation_periods: Required periods for signal confirmation
            close_on_regime_change: Whether to close on regime changes
        """
        super().__init__(
            name="EnhancedStatPairs",
            max_position_size=position_size,
            max_portfolio_exposure=1.0,
            transaction_cost_pct=0.001
        )

        # Core parameters
        self.lookback_window = lookback_window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.max_spread_vol = max_spread_vol
        self.min_correlation = min_correlation
        self.coint_threshold = coint_threshold
        self.position_type = position_type
        self.max_pairs = max_pairs
        self.position_size = position_size

        # Enhanced risk parameters
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.time_stop = time_stop
        self.profit_take = profit_take
        self.max_drawdown = max_drawdown

        # Signal parameters
        self.close_on_signal_flip = close_on_signal_flip
        self.signal_exit_threshold = signal_exit_threshold
        self.confirmation_periods = confirmation_periods
        self.close_on_regime_change = close_on_regime_change

        # Additional parameters
        self.cointegration_windows = cointegration_windows or [63, 126, 252]
        self.min_votes = min_votes or (len(self.cointegration_windows) // 2 + 1)
        self.regime_adaptation = regime_adaptation

        # Initialize components
        self.calculator = StatisticalModel()
        self.pairs: List[Tuple[str, str]] = []
        self.positions: Dict[Tuple[str, str], PairStats] = {}
        self.trades: List[Dict] = []
        self.zscore_history: Dict[Tuple[str, str], List[float]] = {}
        self.position_age: Dict[Tuple[str, str], int] = {}
        self.max_prices: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.entry_prices: Dict[Tuple[str, str], Dict[str, float]] = {}

    def _check_exit_conditions(
            self,
            pair: Tuple[str, str],
            current_prices: Dict[str, float],
            zscore: float,
            current_pos: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Enhanced exit condition checking with new parameters.
        """
        if pair not in self.position_age:
            return False, None

        position_age = self.position_age[pair]

        # Don't check stops during minimum holding period
        if position_age < self.time_stop:
            return False, None

        entry_prices = self.entry_prices[pair]
        max_prices = self.max_prices[pair]

        # Calculate returns for each asset
        returns = {}
        for asset in ['asset1', 'asset2']:
            current = current_prices[asset]
            entry = entry_prices[asset]
            returns[asset] = (current - entry) / entry if entry != 0 else 0

        # Calculate position-weighted return
        position_sizes = self.positions[pair].position_sizes
        total_return = sum(returns[asset] * position_sizes[asset]
                           for asset in ['asset1', 'asset2'])

        # Calculate trailing returns
        trailing_returns = {}
        for asset in ['asset1', 'asset2']:
            current = current_prices[asset]
            peak = max_prices[asset]
            trailing_returns[asset] = (current - peak) / peak if peak != 0 else 0

        # Calculate position-weighted trailing return
        trailing_total = sum(trailing_returns[asset] * position_sizes[asset]
                             for asset in ['asset1', 'asset2'])

        # Check profit target
        if total_return >= self.profit_take:
            return True, "Profit Target"

        # Check stop loss
        if total_return <= -self.stop_loss:
            return True, "Stop Loss"

        # Check trailing stop for profitable positions
        if max(max_prices.values()) > max(entry_prices.values()):
            if trailing_total <= -self.trailing_stop:
                return True, "Trailing Stop"

        # Check z-score exit with confirmation
        if (abs(zscore) <= self.zscore_exit and
                self._confirm_exit_signal(pair, zscore)):
            return True, "Z-Score Exit"

        return False, None

    def _confirm_exit_signal(self, pair: Tuple[str, str], zscore: float) -> bool:
        """
        Require multiple periods of confirmation before exit.
        """
        if pair not in self.zscore_history:
            self.zscore_history[pair] = []
            return False

        recent_scores = self.zscore_history[pair][-self.confirmation_periods:]
        if len(recent_scores) < self.confirmation_periods:
            return False

        # Check if z-score has been consistently near exit threshold
        confirm_count = sum(1 for z in recent_scores
                            if abs(z) <= self.zscore_exit * 1.2)
        return confirm_count >= self.confirmation_periods * 0.8

    def _record_trade(self, pair: Tuple[str, str], date: pd.Timestamp, reason: str,
                      current_pos: float, asset1_data: pd.Series, asset2_data: pd.Series,
                      zscore: float, pnl: float = None) -> None:
        """
        Record details of a trade execution.

        Args:
            pair: Tuple of asset symbols
            date: Trade execution timestamp
            reason: Reason for the trade (entry/exit)
            current_pos: Current position size
            asset1_data: Price data for first asset
            asset2_data: Price data for second asset
            zscore: Current z-score at trade time
            pnl: Profit/loss if known (for exits)
        """
        asset1, asset2 = pair
        stats = self.positions.get(pair)

        trade_info = {
            'Date': date,
            'Pair': f"{asset1}/{asset2}",
            'Action': 'EXIT' if current_pos != 0 else 'ENTRY',
            'Signal': -current_pos if current_pos != 0 else current_pos,
            'ZScore': zscore,
            'Price1': asset1_data.iloc[-1],
            'Price2': asset2_data.iloc[-1],
            'Position1': stats.position_sizes['asset1'] if stats else 0,
            'Position2': stats.position_sizes['asset2'] if stats else 0,
        }

        if trade_info['Action'] == 'EXIT':
            trade_info['ExitReason'] = reason
            if pnl is not None:
                trade_info['PnL'] = pnl

        self.trades.append(trade_info)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for all pairs.

        Processes historical price data to generate entry and exit signals.
        Handles position management including exits based on multiple criteria:
        - Z-score mean reversion
        - Stop losses
        - Signal reversals

        Args:
            prices: DataFrame with price history for all assets

        Returns:
            DataFrame with signals (-1, 0, 1) for each pair
        """
        dates = prices['Date'].unique()
        columns = pd.MultiIndex.from_tuples(self.pairs, names=['asset1', 'asset2'])
        signals = pd.DataFrame(0, index=dates, columns=columns)

        for date in dates:
            historical_data = prices[prices['Date'] <= date].copy()

            for pair in self.pairs:
                try:
                    asset1, asset2 = pair

                    # Get asset data
                    asset1_data = historical_data[historical_data['Symbol'] == asset1].set_index('Date')['Adj_Close']
                    asset2_data = historical_data[historical_data['Symbol'] == asset2].set_index('Date')['Adj_Close']
                    asset1_data, asset2_data = asset1_data.align(asset2_data, join='inner')

                    if len(asset1_data) < self.lookback_window:
                        continue

                    # Calculate core metrics
                    hedge_ratio = self.calculator.calculate_hedge_ratio(asset1_data, asset2_data)
                    spread = self.calculator.calculate_spread(asset1_data, asset2_data)
                    zscore = self.calculator.calculate_spread_zscore(spread)
                    current_zscore = zscore.iloc[-1]

                    # Update zscore history
                    if pair not in self.zscore_history:
                        self.zscore_history[pair] = []
                    self.zscore_history[pair].append(current_zscore)

                    # Generate signals
                    mr_signal = self.calculator.mean_reversion_signal(
                        spread, window=21, z_threshold=self.zscore_entry
                    )
                    bb_signal = self.calculator.bollinger_band_signal(
                        spread, window=21, num_std_dev=self.zscore_entry
                    )

                    current_signal = 0
                    if mr_signal.iloc[-1] == bb_signal.iloc[-1] and mr_signal.iloc[-1] != 0:
                        current_signal = mr_signal.iloc[-1]

                    # Position management
                    current_prices = {
                        'asset1': asset1_data.iloc[-1],
                        'asset2': asset2_data.iloc[-1]
                    }

                    if pair in self.positions:
                        prev_signals = signals.loc[signals.index < date, pair]
                        current_pos = prev_signals.iloc[-1] if len(prev_signals) > 0 else 0

                        # Update position age and max prices
                        self.position_age[pair] = self.position_age.get(pair, 0) + 1

                        if pair in self.max_prices:
                            self.max_prices[pair] = {
                                'asset1': max(self.max_prices[pair]['asset1'], current_prices['asset1']),
                                'asset2': max(self.max_prices[pair]['asset2'], current_prices['asset2'])
                            }

                        # Check exit conditions
                        exit_flag, exit_reason = self._check_exit_conditions(
                            pair, current_prices, current_zscore, current_pos
                        )

                        if exit_flag:
                            signals.loc[date, pair] = 0
                            self._record_trade(
                                pair, date, exit_reason, current_pos,
                                asset1_data, asset2_data, current_zscore,
                                self._calculate_pnl(pair, current_prices)
                            )

                            # Reset position tracking
                            if pair in self.positions:
                                del self.positions[pair]
                            if pair in self.position_age:
                                del self.position_age[pair]
                            if pair in self.max_prices:
                                del self.max_prices[pair]
                            if pair in self.entry_prices:
                                del self.entry_prices[pair]

                            continue

                    # Update or enter new position
                    signals.loc[date, pair] = current_signal

                    if current_signal != 0:
                        # Initialize or update position tracking
                        if pair not in self.entry_prices:
                            self.entry_prices[pair] = current_prices.copy()
                        if pair not in self.max_prices:
                            self.max_prices[pair] = current_prices.copy()
                        if pair not in self.position_age:
                            self.position_age[pair] = 0

                        # Update position stats
                        stats = PairStats(
                            hedge_ratio=hedge_ratio,
                            half_life=self.calculator.calculate_half_life(spread),
                            coint_pvalue=0.0,
                            spread_zscore=current_zscore,
                            spread_vol=spread.std(),
                            correlation=asset1_data.pct_change().corr(asset2_data.pct_change()),
                            last_update=pd.Timestamp(date),
                            position_sizes=self.calculate_position_sizes_v2(
                                pair, current_signal, historical_data
                            )
                        )
                        self.positions[pair] = stats

                except Exception as e:
                    logger.error(f"Error generating signals for pair {pair}: {str(e)}")
                    signals.loc[date, pair] = 0

        return signals

    def _calculate_pnl(self, pair: Tuple[str, str], current_prices: Dict[str, float]) -> float:
        """Calculate PnL for a position."""
        if pair not in self.positions or pair not in self.entry_prices:
            return 0.0

        position_sizes = self.positions[pair].position_sizes
        entry_prices = self.entry_prices[pair]

        pnl = 0.0
        for asset in ['asset1', 'asset2']:
            price_change = (current_prices[asset] - entry_prices[asset]) / entry_prices[asset]
            pnl += price_change * position_sizes[asset]

        return pnl

    def calculate_position_sizes_v2(
            self,
            pair: Tuple[str, str],
            signal: float,
            prices: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate position sizes for each asset in a pair.

        Determines the optimal position sizes based on the trading signal,
        asset prices, and hedge ratio. Implements risk controls based on
        the strategy's position type constraints.

        Args:
            pair: Tuple of asset symbols
            signal: Trading signal (-1, 0, 1)
            prices: DataFrame with current price data

        Returns:
            Dictionary with position sizes for each asset

        Raises:
            Exception: If error occurs during calculation
        """
        try:
            if ((self.position_type == PositionType.LONG_ONLY and signal < 0) or
                    (self.position_type == PositionType.SHORT_ONLY and signal > 0)):
                return {'asset1': 0.0, 'asset2': 0.0}

            asset1, asset2 = pair
            asset1_price = prices[prices['Symbol'] == asset1]['Adj_Close'].iloc[-1]
            asset2_price = prices[prices['Symbol'] == asset2]['Adj_Close'].iloc[-1]

            hedge_ratio = self.calculator.calculate_hedge_ratio(
                prices[prices['Symbol'] == asset1]['Adj_Close'],
                prices[prices['Symbol'] == asset2]['Adj_Close']
            )

            base_position = self.position_size * signal
            notional1 = base_position * asset1_price
            notional2 = notional1 * hedge_ratio

            shares1 = np.floor(notional1 / asset1_price)
            shares2 = np.floor(notional2 / asset2_price)

            return {
                'asset1': shares1,
                'asset2': -shares2 * hedge_ratio
            }

        except Exception as e:
            logger.error(f"Error calculating position sizes: {str(e)}")
            return {'asset1': 0.0, 'asset2': 0.0}

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Identify cointegrated pairs suitable for trading.

        Analyzes price data to find asset pairs that exhibit statistical
        properties suitable for pairs trading:
        - Cointegration relationship
        - Acceptable correlation
        - Mean reversion characteristics within defined bounds

        Args:
            prices: DataFrame with price history for all assets

        Returns:
            List of tuples containing valid trading pairs

        Note:
            Filters pairs based on strategy parameters:
            - Cointegration threshold
            - Minimum correlation
            - Half-life bounds
            - Maximum number of pairs
        """
        valid_pairs = []
        symbols = prices['Symbol'].unique()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                asset1, asset2 = symbols[i], symbols[j]
                try:
                    asset1_prices = prices[prices['Symbol'] == asset1].sort_values('Date')['Adj_Close']
                    asset2_prices = prices[prices['Symbol'] == asset2].sort_values('Date')['Adj_Close']

                    is_cointegrated = self.calculator.cointegration_test(
                        asset1_prices,
                        asset2_prices,
                        significance=self.coint_threshold
                    )

                    if not is_cointegrated:
                        continue

                    correlation = asset1_prices.pct_change().corr(asset2_prices.pct_change())
                    spread = self.calculator.calculate_spread(asset1_prices, asset2_prices)
                    half_life = self.calculator.calculate_half_life(spread)

                    if (correlation >= self.min_correlation and
                            self.min_half_life <= half_life <= self.max_half_life):
                        valid_pairs.append((asset1, asset2))

                    if len(valid_pairs) >= self.max_pairs:
                        break

                except Exception as e:
                    logger.warning(f"Error analyzing pair {asset1}-{asset2}: {str(e)}")
                    continue

        logger.info(f"Found {len(valid_pairs)} valid pairs")
        return valid_pairs

    def _calculate_returns(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy returns and performance metrics.

        Processes trading signals and price data to compute:
        - Individual pair returns
        - Portfolio returns
        - Transaction costs
        - Performance metrics (Sharpe ratio, drawdown)

        Args:
            signals: DataFrame with trading signals
            prices: DataFrame with price data

        Returns:
            DataFrame containing returns and performance metrics

        Note:
            Handles position adjustments based on pair statistics
            and includes transaction cost calculations
        """
        returns = pd.DataFrame(index=signals.index)
        returns['portfolio_return'] = 0.0

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                asset1_prices = prices[prices['Symbol'] == asset1].set_index('Date')['Adj_Close']
                asset2_prices = prices[prices['Symbol'] == asset2].set_index('Date')['Adj_Close']

                rets1 = asset1_prices.pct_change()
                rets2 = asset2_prices.pct_change()

                common_dates = signals.index.intersection(rets1.index).intersection(rets2.index)
                if len(common_dates) == 0:
                    continue

                signals_aligned = signals.loc[common_dates, pair]
                rets1_aligned = rets1.loc[common_dates]
                rets2_aligned = rets2.loc[common_dates]

                pair_signals = signals_aligned.shift(1).fillna(0)
                stats = self.positions.get(pair)

                if stats is not None and stats.position_sizes:
                    pair_returns = pair_signals * (
                            stats.position_sizes['asset1'] * rets1_aligned +
                            stats.position_sizes['asset2'] * rets2_aligned
                    )
                else:
                    hedge_ratio = self.calculator.calculate_hedge_ratio(
                        asset1_prices[common_dates],
                        asset2_prices[common_dates]
                    )
                    pair_returns = pair_signals * (rets1_aligned - hedge_ratio * rets2_aligned)

                trades = pair_signals.diff().fillna(0) != 0
                transaction_costs = trades * self.transaction_cost_pct

                returns[f'{asset1}_{asset2}_return'] = pair_returns - transaction_costs
                returns.loc[common_dates, 'portfolio_return'] += pair_returns - transaction_costs

            except Exception as e:
                logger.error(f"Error calculating returns for pair {pair}: {str(e)}")
                continue

        returns['cumulative_return'] = (1 + returns['portfolio_return']).cumprod()
        returns['drawdown'] = self._calculate_drawdown(returns['cumulative_return'])
        returns['rolling_sharpe'] = self._calculate_rolling_sharpe(returns['portfolio_return'], 252)
        returns['rolling_volatility'] = returns['portfolio_return'].rolling(window=252).std() * np.sqrt(252)

        return returns

    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series for portfolio returns.

        Computes running maximum drawdown as percentage decline
        from previous peak value.

        Args:
            cumulative_returns: Series of cumulative strategy returns

        Returns:
            Series containing drawdown values
        """
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling Sharpe ratio over specified window.

        Computes annualized Sharpe ratio using rolling window of returns.
        Assumes risk-free rate of zero for simplicity.

        Args:
            returns: Series of strategy returns
            window: Rolling window size in days

        Returns:
            Series containing rolling Sharpe ratio values

        Note:
            Annualizes using sqrt(252) assuming daily returns
        """
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = np.sqrt(252) * (rolling_mean / rolling_std)
        return rolling_sharpe


def load_and_preprocess_data(data_dir: str, symbols: List[str]) -> pd.DataFrame:
    """
    Load and preprocess price data for analysis.

    Loads CSV files for specified symbols and combines into a single
    DataFrame with required formatting and cleaning.

    Args:
        data_dir: Directory containing price data files
        symbols: List of symbols to load

    Returns:
        DataFrame with preprocessed price data

    Raises:
        ValueError: If required columns are missing
        Exception: If error occurs during loading
    """
    all_data = []

    for symbol in symbols:
        try:
            file_path = os.path.join(data_dir, f"{symbol}.csv")
            df = pd.read_csv(file_path)
            df['Symbol'] = symbol
            all_data.append(df)
        except Exception as e:
            logger.error(f"Error loading {symbol}: {str(e)}")
            raise

    combined_df = pd.concat(all_data, ignore_index=True)
    required_columns = ['Date', 'Symbol', 'Adj_Close', 'High', 'Low', 'Open', 'Close', 'Volume']
    missing_columns = set(required_columns) - set(combined_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.sort_values(['Date', 'Symbol']).reset_index(drop=True)

    logger.info(f"Successfully loaded data for {len(symbols)} symbols")
    return combined_df


def main():
    """
    Test and analyze the enhanced statistical pairs trading strategy.

    Performs comprehensive testing and analysis including:
    1. Data validation and preprocessing
    2. Strategy parameter optimization
    3. Trading simulation
    4. Performance analysis
    5. Risk metrics calculation
    6. Trade analysis

    Returns:
        Dict containing test results and analysis

    Raises:
        ValueError: If data validation fails
        Exception: If error occurs during testing
    """
    try:
        logger.info("Starting enhanced pairs trading strategy test")

        symbols = ['AAPL', 'MSFT']
        data_dir = os.path.join(DATA_DIR.replace(r'\config', ''), "raw")
        prices = load_and_preprocess_data(data_dir, symbols)

        if prices.empty:
            raise ValueError("No valid price data loaded")

        prices = prices.ffill().bfill()
        logger.info(f"Successfully loaded data for {len(symbols)} symbols")

        for symbol in symbols:
            symbol_data = prices[prices['Symbol'] == symbol]
            missing_pct = symbol_data['Adj_Close'].isnull().mean() * 100
            logger.info(f"{symbol} missing data: {missing_pct:.2f}%")

        strategy = EnhancedStatPairsStrategy(
            lookback_window=252,
            zscore_entry=2.0,
            zscore_exit=0.5,
            min_half_life=3,
            max_half_life=126,
            max_spread_vol=0.12,
            min_correlation=0.45,
            coint_threshold=0.05,
            max_pairs=5,
            position_size=0.1,
            stop_loss=0.10,
            max_drawdown=0.2,
            close_on_signal_flip=True,
            signal_exit_threshold=0.4,
            confirmation_periods=3,
            close_on_regime_change=True
        )

        strategy.pairs = [('AAPL', 'MSFT')]
        logger.info(f"Using fixed pair: {strategy.pairs}")
        if not strategy.pairs:
            strategy.pairs = [('AAPL', 'MSFT')]
        logger.info(f"Testing with pairs: {strategy.pairs}")

        logger.info("Generating trading signals...")
        try:
            signals = strategy.generate_signals(prices)
            signal_analysis = analyze_signals(signals)
            logger.info("\nSignal Analysis:")
            logger.info(f"Total signals generated: {signal_analysis['total_signals']}")
            logger.info(f"Long signals: {signal_analysis['long_signals']}")
            logger.info(f"Short signals: {signal_analysis['short_signals']}")
        except Exception as e:
            logger.error(f"Error in signal generation: {str(e)}")
            raise

        logger.info("Calculating returns...")
        equity_curve = strategy._calculate_returns(signals, prices)

        portfolio_returns = equity_curve['portfolio_return']
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        risk_metrics = {
            'sortino_ratio': calculate_sortino_ratio(portfolio_returns),
            'calmar_ratio': -annual_return / max_drawdown if max_drawdown != 0 else np.inf,
            'avg_drawdown': drawdowns.mean(),
            'drawdown_std': drawdowns.std(),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }

        logger.info("\nStrategy Performance:")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Return: {annual_return:.2%}")
        logger.info(f"Annualized Volatility: {volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
        logger.info(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}")

        print("\nStrategy Performance:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annual_return:.2%}")
        print(f"Annualized Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")

        if strategy.trades:
            trades_df = pd.DataFrame(strategy.trades)

            exits = trades_df[trades_df['Action'] == 'EXIT']
            print("\nExit Analysis:")
            if not exits.empty:
                exit_reasons = exits['ExitReason'].value_counts()

                print("\nExit Distribution:")
                for reason, count in exit_reasons.items():
                    print(f"{reason}: {count} exits ({count / len(exits):.1%})")

                pnl_by_reason = exits.groupby('ExitReason')['PnL'].agg(['mean', 'count', 'sum'])
                print("\nPnL by Exit Reason:")
                print(pnl_by_reason)

        print("\nPairs Analysis:")
        pairs_metrics = {}
        for pair in strategy.pairs:
            stats = strategy.positions.get(pair)
            if stats:
                pair_name = f"{pair[0]}/{pair[1]}"
                pairs_metrics[pair_name] = {
                    "hedge_ratio": stats.hedge_ratio,
                    "half_life": stats.half_life,
                    "correlation": stats.correlation,
                    "spread_vol": stats.spread_vol,
                    "avg_position_size": np.mean([abs(v) for v in stats.position_sizes.values()])
                }

                print(f"\n{pair_name}:")
                print(f"Hedge Ratio: {stats.hedge_ratio:.4f}")
                print(f"Half-Life: {stats.half_life:.1f} days")
                print(f"Correlation: {stats.correlation:.4f}")
                print(f"Spread Volatility: {stats.spread_vol:.4f}")

        exit_reasons = pd.Series()
        pnl_by_reason = pd.Series()

        return {
            'strategy': strategy,
            'signals': signals,
            'signal_analysis': signal_analysis,
            'prices': prices,
            'pairs': strategy.pairs,
            'pairs_metrics': pairs_metrics,
            'equity_curve': equity_curve,
            'performance': {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'risk_metrics': risk_metrics
            },
            'trade_analysis': {
                'total_trades': len(strategy.trades),
                'exit_distribution': exit_reasons.to_dict() if strategy.trades else {},
                'pnl_by_reason': pnl_by_reason.to_dict() if strategy.trades else {}
            }
        }

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return None


def analyze_signals(signals: pd.DataFrame) -> Dict:
    """
    Analyze trading signal characteristics.

    Args:
        signals: DataFrame of trading signals

    Returns:
        Dict containing signal statistics
    """
    signal_counts = {
        'total_signals': np.array((signals != 0)).sum().sum(),
        'long_signals': np.array((signals > 0)).sum().sum(),
        'short_signals': np.array((signals < 0)).sum().sum()
    }
    return signal_counts


def calculate_sortino_ratio(returns: pd.Series) -> float:
    """
    Calculate Sortino ratio using negative returns standard deviation.

    Args:
        returns: Series of strategy returns

    Returns:
        float: Sortino ratio value
    """
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return np.inf
    return (returns.mean() * 252) / downside_std


if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\nEnhanced Statistical Pairs Trading analysis completed successfully!")
    else:
        print("\nAnalysis failed. Check logs for details.")