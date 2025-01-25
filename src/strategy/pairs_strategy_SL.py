"""
Enhanced Statistical Pairs Trading Strategy

Features:
1. Multi-period cointegration voting
2. Composite scoring system
3. Regime detection and adaptation
4. Enhanced signal generation using statistical indicators
5. Clear position sizing for each leg
6. Maintains BaseStrategy compatibility
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from plotly.subplots import make_subplots
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from pathlib import Path
import json
from sklearn.cluster import KMeans

from src.strategy.base import BaseStrategy
from src.models.statistical import StatisticalModel
from config.logging_config import logger
from config.settings import DATA_DIR

@dataclass
class RegimeMetrics:
    """Track market regime characteristics"""
    volatility: float
    correlation: float
    trend_strength: float
    regime_type: str
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp] = None

@dataclass
class PairStats:
    """Enhanced pair statistics with multi-period metrics"""
    hedge_ratio: float
    half_life: float
    coint_pvalue: float
    spread_zscore: float
    spread_vol: float
    correlation: float
    last_update: pd.Timestamp
    cointegration_votes: Dict[int, bool] = field(default_factory=dict)
    composite_score: float = 0.0
    regime_metrics: Optional[RegimeMetrics] = None
    lookback_data: Dict[str, pd.Series] = field(default_factory=dict)
    position_sizes: Dict[str, float] = field(default_factory=dict)

class MarketRegimeDetector:
    """Detect and analyze market regimes"""

    def __init__(self, n_regimes: int = 3, window: int = 63):
        self.n_regimes = n_regimes
        self.window = window
        self.kmeans = KMeans(n_clusters=n_regimes)

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime detection features"""
        features = pd.DataFrame(index=data['Date'].unique())

        returns_by_symbol = {}
        for symbol in data['Symbol'].unique():
            mask = data['Symbol'] == symbol
            symbol_data = data[mask].sort_values('Date')
            returns_by_symbol[symbol] = symbol_data['Adj_Close'].pct_change()

        returns_matrix = pd.DataFrame(returns_by_symbol, index=features.index).fillna(0)

        symbol_vols = pd.DataFrame()
        for symbol in returns_matrix.columns:
            symbol_vols[symbol] = returns_matrix[symbol].rolling(window=self.window).std()
        features['volatility'] = symbol_vols.mean(axis=1)

        def calculate_average_correlation(x):
            if x.isnull().all().all():
                return 0
            corr_matrix = x.corr()
            upper_tri = np.triu(corr_matrix.values, k=1)
            valid_corrs = upper_tri[upper_tri != 0]
            return np.mean(valid_corrs) if len(valid_corrs) > 0 else 0

        correlations = []
        for i in range(len(returns_matrix)):
            if i >= self.window:
                window_data = returns_matrix.iloc[i - self.window:i]
                correlations.append(calculate_average_correlation(window_data))
            else:
                correlations.append(0)
        features['avg_correlation'] = correlations

        def hurst(ts):
            """Calculate Hurst exponent for time series"""
            if len(ts) < 20:
                return np.nan
            lags = range(2, min(len(ts) // 2, 20))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]

        symbol_trends = pd.DataFrame()
        for symbol in returns_matrix.columns:
            trends = []
            symbol_returns = returns_matrix[symbol]
            for i in range(len(symbol_returns)):
                if i >= self.window:
                    window_data = symbol_returns.iloc[i - self.window:i]
                    trends.append(hurst(window_data.values))
                else:
                    trends.append(np.nan)
            symbol_trends[symbol] = trends

        features['trend_strength'] = symbol_trends.mean(axis=1)

        return features.ffill().bfill()

    def detect_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Detect market regime using clustering"""
        standardized = (features - features.mean()) / features.std()
        standardized = standardized.dropna()

        if len(standardized) == 0:
            return 'Unknown'

        regimes = self.kmeans.fit_predict(standardized)
        return regimes[-1]

    def classify_regime(self, regime_idx: int, features: pd.DataFrame) -> str:
        """Classify regime type based on characteristics"""
        regime_features = features.iloc[regime_idx]

        if regime_features['volatility'] > regime_features['volatility'].quantile(0.7):
            if regime_features['avg_correlation'] > regime_features['avg_correlation'].quantile(0.7):
                return 'Crisis'
            return 'High Volatility'

        if regime_features['trend_strength'] > regime_features['trend_strength'].quantile(0.7):
            return 'Trending'

        return 'Mean Reverting'

class EnhancedStatPairsStrategy(BaseStrategy):
    """Enhanced statistical pairs trading strategy"""

    def __init__(
            self,
            lookback_window: int = 252,
            zscore_entry: float = 2.0,
            zscore_exit: float = 0.0,
            min_half_life: int = 5,
            max_half_life: int = 126,
            max_spread_vol: float = 0.1,
            min_correlation: float = 0.5,
            coint_threshold: float = 0.05,
            max_pairs: int = 10,
            position_size: float = 0.1,
            stop_loss: float = 0.02,
            max_drawdown: float = 0.2,
            cointegration_windows: List[int] = None,
            min_votes: int = None,
            regime_adaptation: bool = True,

            close_on_signal_flip: bool = True,
            signal_exit_threshold: float = 0.3,
            confirmation_periods: int = 2,
            close_on_regime_change: bool = True
    ):
        """Initialize strategy with enhanced parameters"""
        super().__init__(
            name="EnhancedStatPairs",
            max_position_size=position_size,
            max_portfolio_exposure=1.0,
            transaction_cost_pct=0.001
        )

        self.lookback_window = lookback_window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.max_spread_vol = max_spread_vol
        self.min_correlation = min_correlation
        self.coint_threshold = coint_threshold
        self.max_pairs = max_pairs
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.max_drawdown = max_drawdown

        self.cointegration_windows = cointegration_windows or [63, 126, 252]
        self.min_votes = min_votes or (len(self.cointegration_windows) // 2 + 1)
        self.regime_adaptation = regime_adaptation

        self.calculator = StatisticalModel()
        self.regime_detector = MarketRegimeDetector()
        self.pairs: List[Tuple[str, str]] = []
        self.positions: Dict[Tuple[str, str], PairStats] = {}
        self.trades: List[Dict] = []

        self.current_regime: Optional[RegimeMetrics] = None


        self.close_on_signal_flip = close_on_signal_flip
        self.signal_exit_threshold = signal_exit_threshold
        self.confirmation_periods = confirmation_periods
        self.close_on_regime_change = close_on_regime_change

        self.signal_history: Dict[Tuple[str, str], pd.DataFrame] = {}

    def test_cointegration_multiple_periods(
            self,
            price1: pd.Series,
            price2: pd.Series) -> Dict[int, bool]:
        """Test cointegration across multiple lookback periods"""
        votes = {}

        for window in self.cointegration_windows:
            if len(price1) < window or len(price2) < window:
                continue

            try:
                _, pvalue, _ = coint(
                    price1.iloc[-window:],
                    price2.iloc[-window:],
                    maxlag=min(window // 10, 20),
                    trend='c'
                )
                votes[window] = pvalue < self.coint_threshold
            except:
                votes[window] = False

        return votes

    def plot_pair_analysis(self, prices: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
        """Plot comprehensive pair analysis with regime information"""
        if not self.pairs:
            logger.warning("No pairs available for plotting")
            return

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                stats = self.positions.get(pair)
                if not stats:
                    continue

                returns1 = prices[asset1].pct_change()
                returns2 = prices[asset2].pct_change()
                spread = prices[asset1] - stats.hedge_ratio * prices[asset2]

                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        'Asset Prices',
                        'Returns Correlation',
                        'Spread Z-Score',
                        'Rolling Volatility',
                        'Regime Analysis',
                        'Position Sizes'
                    ),
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1
                )

                fig.add_trace(
                    go.Scatter(x=prices.index, y=prices[asset1], name=asset1),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=prices.index, y=prices[asset2], name=asset2),
                    row=1, col=1
                )

                rolling_corr = returns1.rolling(63).corr(returns2)
                fig.add_trace(
                    go.Scatter(x=prices.index, y=rolling_corr, name='Rolling Correlation'),
                    row=1, col=2
                )

                zscore = self.calculator.calculate_spread_zscore(spread)
                fig.add_trace(
                    go.Scatter(x=prices.index, y=zscore, name='Z-Score'),
                    row=2, col=1
                )

                fig.add_hline(y=self.zscore_entry, line_dash="dash", row=2, col=1)
                fig.add_hline(y=-self.zscore_entry, line_dash="dash", row=2, col=1)

                roll_vol = returns1.rolling(21).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(x=prices.index, y=roll_vol, name=f'{asset1} Volatility'),
                    row=2, col=2
                )
                roll_vol2 = returns2.rolling(21).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(x=prices.index, y=roll_vol2, name=f'{asset2} Volatility'),
                    row=2, col=2
                )

                if stats.regime_metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=[stats.regime_metrics.start_date],
                            y=[stats.regime_metrics.volatility],
                            mode='markers+text',
                            text=[stats.regime_metrics.regime_type],
                            textposition="top center",
                            name='Current Regime'
                        ),
                        row=3, col=1
                    )

                if stats.position_sizes:
                    fig.add_trace(
                        go.Bar(
                            x=[asset1, asset2],
                            y=[stats.position_sizes['asset1'], stats.position_sizes['asset2']],
                            name='Position Sizes'
                        ),
                        row=3, col=2
                    )

                fig.update_layout(
                    height=1200,
                    title_text=f"Pair Analysis: {asset1}/{asset2}",
                    showlegend=True
                )

                if output_dir:
                    fig.write_html(output_dir / f"pair_analysis_{asset1}_{asset2}.html")
                else:
                    fig.show()

            except Exception as e:
                logger.error(f"Error plotting pair analysis for {pair}: {str(e)}")

    def calculate_composite_score(
            self,
            pair: Tuple[str, str],
            stats: PairStats,
            prices: pd.DataFrame) -> float:
        """Calculate composite score combining multiple metrics"""
        try:
            coint_score = sum(stats.cointegration_votes.values()) / len(stats.cointegration_votes)

            asset1, asset2 = pair
            asset1_data = prices[prices['Symbol'] == asset1].sort_values('Date')
            asset2_data = prices[prices['Symbol'] == asset2].sort_values('Date')

            returns1 = asset1_data['Adj_Close'].pct_change()
            returns2 = asset2_data['Adj_Close'].pct_change()

            rolling_corr = pd.Series(index=returns1.index)
            for i in range(63, len(returns1)):
                rolling_corr.iloc[i] = returns1.iloc[i - 63:i].corr(returns2.iloc[i - 63:i])

            corr_stability = 1 - rolling_corr.std()

            hl_score = 1 - abs(stats.half_life - np.mean([self.min_half_life, self.max_half_life])) / (
                        self.max_half_life - self.min_half_life)
            vol_score = 1 - (stats.spread_vol / self.max_spread_vol)

            weights = {
                'cointegration': 0.4,
                'correlation': 0.2,
                'half_life': 0.2,
                'volatility': 0.2
            }

            composite_score = (
                    weights['cointegration'] * coint_score +
                    weights['correlation'] * corr_stability +
                    weights['half_life'] * hl_score +
                    weights['volatility'] * vol_score
            )

            return float(composite_score)

        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0.0

    def calculate_position_sizes_v2(
            self,
            pair: Tuple[str, str],
            signal: float,
            prices: pd.DataFrame) -> Dict[str, float]:
        """Calculate actual position sizes for each leg"""
        try:
            asset1, asset2 = pair
            stats = self.positions.get(pair)

            if not stats:
                return {'asset1': 0.0, 'asset2': 0.0}

            latest_prices = prices[prices['Date'] == prices['Date'].max()]
            price1 = latest_prices[latest_prices['Symbol'] == asset1]['Adj_Close'].iloc[0]
            price2 = latest_prices[latest_prices['Symbol'] == asset2]['Adj_Close'].iloc[0]

            pair_value = price1 + stats.hedge_ratio * price2
            base_position = self.position_size * signal

            if self.regime_adaptation and stats.regime_metrics:
                regime = stats.regime_metrics.regime_type
                if regime == 'Crisis':
                    base_position *= 0.5
                elif regime == 'High Volatility':
                    base_position *= 0.75
                elif regime == 'Trending':
                    base_position *= 0.9

            notional_value = pair_value * base_position
            shares1 = np.floor(notional_value / price1)
            shares2 = np.floor(notional_value * stats.hedge_ratio / price2)

            return {
                'asset1': shares1,
                'asset2': -shares2 * stats.hedge_ratio
            }

        except Exception as e:
            logger.error(f"Error calculating position sizes: {str(e)}")
            return {'asset1': 0.0, 'asset2': 0.0}

    def generate_signals(self, prices: pd.DataFrame) -> Union[pd.DataFrame, Dict[Tuple[str, str], pd.Series]]:
        """Generate trading signals for pairs"""
        dates = prices['Date'].unique()
        signals = pd.DataFrame(index=dates, columns=self.pairs)

        if not self.pairs:
            self.pairs = self.find_pairs(prices)

        if self.regime_adaptation:
            regime_features = self.regime_detector.calculate_features(prices)
            current_regime_type = self.regime_detector.classify_regime(-1, regime_features)

            self.current_regime = RegimeMetrics(
                volatility=regime_features['volatility'].iloc[-1],
                correlation=regime_features['avg_correlation'].iloc[-1],
                trend_strength=regime_features['trend_strength'].iloc[-1],
                regime_type=current_regime_type,
                start_date=prices['Date'].max()
            )

        for date in dates:
            date_mask = prices['Date'] <= date
            historical_data = prices[date_mask]

            for pair in self.pairs:
                try:
                    asset1, asset2 = pair

                    asset1_data = historical_data[historical_data['Symbol'] == asset1]['Adj_Close']
                    asset2_data = historical_data[historical_data['Symbol'] == asset2]['Adj_Close']

                    votes = self.test_cointegration_multiple_periods(
                        asset1_data,
                        asset2_data
                    )

                    if sum(votes.values()) < self.min_votes:
                        signals.loc[date, pair] = 0
                        continue

                    hedge_ratio = self.calculator.calculate_hedge_ratio(
                        asset1_data,
                        asset2_data
                    )

                    spread = asset1_data - hedge_ratio * asset2_data

                    asset1_returns = asset1_data.pct_change()
                    asset2_returns = asset2_data.pct_change()
                    correlation = asset1_returns.corr(asset2_returns)

                    stats = PairStats(
                        hedge_ratio=hedge_ratio,
                        half_life=self.calculator.calculate_half_life(spread),
                        coint_pvalue=min(v for v in votes.values()),
                        spread_zscore=self.calculator.calculate_spread_zscore(spread),
                        spread_vol=spread.std(),
                        correlation=correlation,
                        last_update=date,
                        cointegration_votes=votes,
                        regime_metrics=self.current_regime if self.regime_adaptation else None,
                        lookback_data={'spread': spread}
                    )

                    stats.composite_score = self.calculate_composite_score(pair, stats, historical_data)
                    self.positions[pair] = stats

                    signal = self.generate_enhanced_signals(spread, stats)

                    if signal != 0:
                        position_sizes = self.calculate_position_sizes_v2(pair, signal, historical_data)
                        stats.position_sizes = position_sizes
                        signals.loc[date, pair] = signal
                    else:
                        signals.loc[date, pair] = signals.loc[date, pair].shift(1).fillna(0)

                except Exception as e:
                    logger.error(f"Error generating signals for pair {pair} at {date}: {str(e)}")
                    signals.loc[date, pair] = 0

        return signals

    def generate_enhanced_signals(
            self,
            spread: pd.Series,
            stats: PairStats) -> float:
        """Generate trading signals using multiple indicators"""
        try:
            mr_signal = self.calculator.mean_reversion_signal(
                spread,
                window=21,
                z_threshold=self.zscore_entry
            ).iloc[-1]

            bb_signal = self.calculator.bollinger_band_signal(
                spread,
                window=21,
                num_std_dev=2.0
            ).iloc[-1]

            rsi_signal = self.calculator.rsi_signal(
                spread,
                window=14,
                lower_threshold=30,
                upper_threshold=70
            ).iloc[-1]

            weights = {'mr': 0.4, 'bb': 0.4, 'rsi': 0.2}

            if stats.regime_metrics:
                regime = stats.regime_metrics.regime_type
                if regime == 'Mean Reverting':
                    weights = {'mr': 0.5, 'bb': 0.3, 'rsi': 0.2}
                elif regime == 'Trending':
                    weights = {'mr': 0.3, 'bb': 0.5, 'rsi': 0.2}
                elif regime == 'High Volatility':
                    weights = {'mr': 0.4, 'bb': 0.4, 'rsi': 0.2}

            combined_signal = (
                weights['mr'] * mr_signal +
                weights['bb'] * bb_signal +
                weights['rsi'] * rsi_signal
            )

            if abs(combined_signal) < 0.5:
                return 0.0
            return np.sign(combined_signal)

        except Exception as e:
            logger.error(f"Error generating enhanced signals: {str(e)}")
            return 0.0

    def _calculate_returns(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns with enhanced metrics"""
        returns = pd.DataFrame(index=signals.index)
        returns['portfolio_return'] = 0.0

        for pair in self.pairs:
            try:
                asset1, asset2 = pair

                # Get returns for each asset
                asset1_data = prices[prices['Symbol'] == asset1].sort_values('Date')
                asset2_data = prices[prices['Symbol'] == asset2].sort_values('Date')

                rets1 = asset1_data['Adj_Close'].pct_change()
                rets2 = asset2_data['Adj_Close'].pct_change()

                stats = self.positions.get(pair)
                if not stats:
                    continue

                pair_signals = signals[pair].shift(1)
                pair_returns = pair_signals * (rets1 - stats.hedge_ratio * rets2)

                if stats.position_sizes:
                    adj_returns = (
                            stats.position_sizes['asset1'] * rets1 +
                            stats.position_sizes['asset2'] * rets2
                    )
                    pair_returns = pair_signals * adj_returns

                trades = pair_signals.diff().fillna(0) != 0
                transaction_costs = trades * self.transaction_cost_pct

                net_returns = pair_returns - transaction_costs
                returns[f'{asset1}_{asset2}_return'] = net_returns

                position_size = self.calculate_position_sizes_v2(pair, stats, prices)
                returns['portfolio_return'] += net_returns * position_size

            except Exception as e:
                logger.error(f"Error calculating returns for pair {pair}: {str(e)}")
                continue

        returns['cumulative_return'] = (1 + returns['portfolio_return']).cumprod()
        returns['drawdown'] = self._calculate_drawdown(returns['cumulative_return'])

        window = min(252, len(returns) // 4)
        returns['rolling_sharpe'] = self._calculate_rolling_sharpe(returns['portfolio_return'], window)
        returns['rolling_volatility'] = returns['portfolio_return'].rolling(window).std() * np.sqrt(252)

        return returns

    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = np.sqrt(252) * (rolling_mean / rolling_std)
        return rolling_sharpe

    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations for optimization"""
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]

    def optimize_parameters(self, prices: pd.DataFrame, param_grid: Dict) -> Dict:
        """Optimize strategy parameters using grid search"""
        best_sharpe = -np.inf
        best_params = {}

        for params in self._generate_param_combinations(param_grid):
            try:
                for param, value in params.items():
                    setattr(self, param, value)

                signals = self.generate_signals(prices)
                returns = self._calculate_returns(signals, prices)

                sharpe = self._calculate_sharpe_ratio(returns['portfolio_return'])

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params.copy()

            except Exception as e:
                logger.error(f"Error in parameter optimization: {str(e)}")
                continue

        return best_params

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return -np.inf

        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)

        return annualized_return / annualized_vol if annualized_vol > 0 else -np.inf

    def plot_strategy_analysis(self, prices: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
        """Plot comprehensive strategy analysis including exit behaviors, regimes, and price data"""
        if not self.trades:
            logger.warning("No trades available for analysis")
            return

        trades_df = pd.DataFrame(self.trades)

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Price Evolution and Trades',
                'Returns Distribution',
                'Exit Types Distribution',
                'Signal Strength vs. Time',
                'Regime Changes and Positions',
                'Rolling Volatility',
                'Performance by Exit Type',
                'Regime Transition Matrix'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            row_heights=[0.3, 0.23, 0.23, 0.23]
        )

        traded_pairs = set((trade['Pair'].split('/')[0], trade['Pair'].split('/')[1])
                           for trade in self.trades if 'Pair' in trade)

        for asset1, asset2 in traded_pairs:
            norm_price1 = prices[asset1] / prices[asset1].iloc[0]
            norm_price2 = prices[asset2] / prices[asset2].iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=norm_price1,
                    name=f'{asset1} Price',
                    line=dict(width=1)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=norm_price2,
                    name=f'{asset2} Price',
                    line=dict(width=1)
                ),
                row=1, col=1
            )

        entry_trades = trades_df[trades_df['Action'] == 'ENTRY']
        exit_trades = trades_df[trades_df['Action'] == 'EXIT']

        for trades, color, name in [(entry_trades, 'green', 'Entries'),
                                    (exit_trades, 'red', 'Exits')]:
            if not trades.empty and 'Date' in trades.columns:
                fig.add_trace(
                    go.Scatter(
                        x=trades['Date'],
                        y=[1] * len(trades),
                        mode='markers',
                        marker=dict(color=color, size=8),
                        name=name,
                        text=trades['Pair'],
                        hovertemplate='%{text}<br>Date: %{x}'
                    ),
                    row=1, col=1
                )

        for pair in traded_pairs:
            asset1, asset2 = pair
            pair_returns = pd.Series(0.0, index=prices.index)
            pair_trades = trades_df[trades_df['Pair'] == f"{asset1}/{asset2}"]

            if not pair_trades.empty:
                returns = prices[asset1].pct_change() - prices[asset2].pct_change()
                mask = pd.Series(False, index=prices.index)

                for _, trade in pair_trades.iterrows():
                    if 'Date' in trade:
                        mask[trade['Date']] = True

                pair_returns[mask] = returns[mask]

                fig.add_trace(
                    go.Histogram(
                        x=pair_returns[pair_returns != 0],
                        name=f"{asset1}/{asset2}",
                        opacity=0.7,
                        nbinsx=30
                    ),
                    row=1, col=2
                )

        exit_reasons = trades_df[trades_df['Action'] == 'EXIT']['Reason'].value_counts()
        fig.add_trace(
            go.Bar(x=exit_reasons.index, y=exit_reasons.values, name='Exit Types'),
            row=2, col=1
        )

        for pair in self.pairs:
            if pair in self.signal_history:
                history = self.signal_history[pair]
                fig.add_trace(
                    go.Scatter(
                        x=history['timestamp'],
                        y=history['signal_strength'],
                        name=f"{pair[0]}/{pair[1]} Signal",
                        mode='lines'
                    ),
                    row=2, col=2
                )

        fig.add_hline(
            y=self.signal_exit_threshold,
            line_dash="dash",
            row=2, col=2,
            name='Exit Threshold'
        )

        lookback = 21
        for asset in prices.columns:
            vol = prices[asset].pct_change().rolling(lookback).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=vol,
                    name=f'{asset} Volatility',
                    line=dict(width=1)
                ),
                row=3, col=2
            )

        exit_performance = exit_trades.groupby('Reason')['PnL'].agg(['mean', 'count'])
        fig.add_trace(
            go.Bar(
                x=exit_performance.index,
                y=exit_performance['mean'],
                name='Avg PnL by Exit Type'
            ),
            row=4, col=1
        )

        if 'regime_type' in exit_trades.columns:
            regime_transitions = []
            for i in range(1, len(exit_trades)):
                prev_regime = exit_trades.iloc[i - 1]['regime_type']
                curr_regime = exit_trades.iloc[i]['regime_type']
                regime_transitions.append((prev_regime, curr_regime))

            if regime_transitions:
                transition_df = pd.DataFrame(regime_transitions, columns=['from', 'to'])
                transition_matrix = pd.crosstab(
                    transition_df['from'],
                    transition_df['to'],
                    normalize='index'
                )

                fig.add_trace(
                    go.Heatmap(
                        z=transition_matrix.values,
                        x=transition_matrix.columns,
                        y=transition_matrix.index,
                        colorscale='RdYlBu',
                        name='Regime Transitions'
                    ),
                    row=4, col=2
                )

        fig.update_layout(
            height=1600,
            title_text="Strategy Analysis - Exit Behaviors and Regimes",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
        fig.update_xaxes(title_text="Return", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Annual Volatility", row=3, col=2)
        fig.update_yaxes(title_text="Average PnL", row=4, col=1)

        if output_dir:
            fig.write_html(output_dir / "strategy_analysis.html")
        else:
            fig.show()

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Return default parameters for the enhanced strategy"""
        return {
            "lookback_window": 252,
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "min_half_life": 5,
            "max_half_life": 126,
            "max_spread_vol": 0.1,
            "min_correlation": 0.5,
            "coint_threshold": 0.05,
            "max_pairs": 10,
            "position_size": 0.1,
            "stop_loss": 0.02,
            "max_drawdown": 0.2,
            "cointegration_windows": [63, 126, 252],
            "min_votes": 2,
            "regime_adaptation": True,
            "close_on_signal_flip": True,
            "signal_exit_threshold": 0.3,
            "confirmation_periods": 2,
            "close_on_regime_change": True
        }

    def save_state(self, path: str) -> None:
        """Save enhanced strategy state and parameters"""
        state_path = Path(path)
        state_path.mkdir(parents=True, exist_ok=True)

        state = {
            'parameters': {
                'lookback_window': self.lookback_window,
                'zscore_entry': self.zscore_entry,
                'zscore_exit': self.zscore_exit,
                'min_half_life': self.min_half_life,
                'max_half_life': self.max_half_life,
                'max_spread_vol': self.max_spread_vol,
                'min_correlation': self.min_correlation,
                'coint_threshold': self.coint_threshold,
                'cointegration_windows': self.cointegration_windows,
                'min_votes': self.min_votes,
                'regime_adaptation': self.regime_adaptation
            },
            'pairs': self.pairs,
            'trades': self.trades,
            'current_regime': self.current_regime.__dict__ if self.current_regime else None
        }

        with open(state_path / "strategy_state.json", 'w') as f:
            json.dump(state, f, indent=4, default=str)

        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(state_path / "trade_history.csv", index=False)

    def load_state(self, path: str) -> None:
        """Load enhanced strategy state and parameters"""
        state_path = Path(path)

        with open(state_path / "strategy_state.json", 'r') as f:
            state = json.load(f)

        for param, value in state['parameters'].items():
            setattr(self, param, value)

        self.pairs = state['pairs']
        self.trades = state['trades']

        if state['current_regime']:
            self.current_regime = RegimeMetrics(**state['current_regime'])

        trade_history_path = state_path / "trade_history.csv"
        if trade_history_path.exists():
            self.trades = pd.read_csv(trade_history_path).to_dict('records')

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find cointegrated pairs with multiple testing correction"""
        valid_pairs = []
        pair_scores = []
        all_pvalues = []
        all_pairs = []

        symbols = prices['Symbol'].unique()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                asset1, asset2 = symbols[i], symbols[j]

                try:
                    asset1_prices = prices[prices['Symbol'] == asset1].sort_values('Date')['Adj_Close']
                    asset2_prices = prices[prices['Symbol'] == asset2].sort_values('Date')['Adj_Close']

                    votes = self.test_cointegration_multiple_periods(
                        asset1_prices,
                        asset2_prices
                    )
                    min_pvalue = min(v for v in votes.values())
                    all_pvalues.append(min_pvalue)
                    all_pairs.append((asset1, asset2))

                except Exception as e:
                    logger.warning(f"Error analyzing pair {asset1}-{asset2}: {str(e)}")
                    continue

        reject, adj_pvalues, _, _ = multipletests(all_pvalues, alpha=self.coint_threshold, method='fdr_bh')

        for (asset1, asset2), adj_pval, is_significant in zip(all_pairs, adj_pvalues, reject):
            if not is_significant:
                continue

            try:
                asset1_prices = prices[prices['Symbol'] == asset1].sort_values('Date')['Adj_Close']
                asset2_prices = prices[prices['Symbol'] == asset2].sort_values('Date')['Adj_Close']

                hedge_ratio = self.calculator.calculate_hedge_ratio(
                    asset1_prices,
                    asset2_prices
                )

                spread = asset1_prices - hedge_ratio * asset2_prices

                asset1_returns = asset1_prices.pct_change()
                asset2_returns = asset2_prices.pct_change()
                correlation = asset1_returns.corr(asset2_returns)

                stats = PairStats(
                    hedge_ratio=hedge_ratio,
                    half_life=self.calculator.calculate_half_life(spread),
                    coint_pvalue=adj_pval,
                    spread_zscore=self.calculator.calculate_spread_zscore(spread),
                    spread_vol=spread.std(),
                    correlation=correlation,
                    last_update=prices['Date'].max(),
                    lookback_data={'spread': spread}
                )

                if (stats.correlation >= self.min_correlation and
                        self.min_half_life <= stats.half_life <= self.max_half_life and
                        stats.spread_vol <= self.max_spread_vol):
                    score = self.calculate_composite_score((asset1, asset2), stats, prices)
                    pair_scores.append((asset1, asset2, score))

            except Exception as e:
                logger.warning(f"Error calculating statistics for {asset1}-{asset2}: {str(e)}")
                continue

        if pair_scores:
            pair_scores.sort(key=lambda x: x[2], reverse=True)
            valid_pairs = [(p[0], p[1]) for p in pair_scores[:self.max_pairs]]

        logger.info(f"Found {len(valid_pairs)} valid pairs after multiple testing correction")
        return valid_pairs

def main():
    """Test the enhanced statistical pairs trading strategy"""
    logger.info("Starting enhanced pairs trading strategy test")

    output_dir = Path("pairs_trading_strategy_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    results_dir = output_dir / "results"

    for directory in [models_dir, plots_dir, data_dir, results_dir]:
        directory.mkdir(exist_ok=True)

    try:
        logger.info("Loading test data...")
        selected_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'AMD', 'INTC', 'TSM', 'AVGO'
        ]

        raw_data_dir = Path(DATA_DIR) / "raw"
        prices = pd.DataFrame()

        for symbol in selected_symbols:
            try:
                csv_path = raw_data_dir / f"{symbol}.csv"
                if not csv_path.exists():
                    logger.warning(f"Data file not found for {symbol}")
                    continue

                df = pd.read_csv(csv_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                prices[symbol] = df['Adj_Close']

            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                continue

        if prices.empty:
            raise ValueError("No valid price data loaded")

        prices = prices.ffill().bfill()
        prices.to_csv(data_dir / "price_data.csv")
        logger.info(f"Successfully loaded data for {len(prices.columns)} symbols")

        strategy = EnhancedStatPairsStrategy(
            lookback_window=252,
            zscore_entry=2.0,
            zscore_exit=0.5,
            min_half_life=5,
            max_half_life=126,
            max_spread_vol=0.12,
            min_correlation=0.45,
            coint_threshold=0.05,
            max_pairs=5,
            position_size=0.1,
            stop_loss=0.02,
            max_drawdown=0.2,
            close_on_signal_flip=True,
            signal_exit_threshold=0.3,
            confirmation_periods=2,
            close_on_regime_change=True
        )

        logger.info("Finding cointegrated pairs...")
        pairs = strategy.find_pairs(prices)
        logger.info(f"Found {len(pairs)} valid pairs")

        logger.info("Generating trading signals...")
        signals = strategy.generate_signals(prices)
        signals.to_csv(results_dir / "trading_signals.csv")

        logger.info("Running backtest...")
        equity_curve = strategy._calculate_returns(signals, prices)
        equity_curve.to_csv(results_dir / "equity_curve.csv")

        logger.info("Generating analysis plots...")
        strategy.plot_pair_analysis(prices, plots_dir)
        strategy.plot_strategy_analysis(prices, plots_dir)

        if True:
            logger.info("Optimizing strategy parameters...")
            param_grid = {
                'zscore_entry': [1.5, 2.0, 2.5],
                'zscore_exit': [0.0, 0.5],
                'signal_exit_threshold': [0.2, 0.3, 0.4],
                'confirmation_periods': [2, 3]
            }

            best_params = strategy.optimize_parameters(prices, param_grid)

            with open(results_dir / "optimization_results.json", 'w') as f:
                json.dump(best_params, f, indent=4)

        strategy.save_state(str(models_dir / "final_state"))

        with open(results_dir / "analysis_summary.txt", 'w') as f:
            f.write("Enhanced Statistical Pairs Trading Analysis\n")
            f.write("=======================================\n\n")

            f.write(f"Analysis Period: {prices.index[0].date()} to {prices.index[-1].date()}\n")
            f.write(f"Number of assets analyzed: {len(prices.columns)}\n")
            f.write(f"Number of trading pairs found: {len(strategy.pairs)}\n\n")

            f.write("Trading Pairs:\n")
            for pair in strategy.pairs:
                stats = strategy.positions.get(pair)
                if stats:
                    f.write(f"\n{pair[0]} - {pair[1]}:\n")
                    f.write(f"  Hedge Ratio: {stats.hedge_ratio:.4f}\n")
                    f.write(f"  Half-Life: {stats.half_life:.1f} days\n")
                    f.write(f"  Correlation: {stats.correlation:.4f}\n")
                    f.write(f"  Composite Score: {stats.composite_score:.4f}\n")
                    if stats.regime_metrics:
                        f.write(f"  Current Regime: {stats.regime_metrics.regime_type}\n")

            f.write("\nExit Analysis:\n")
            exit_types = pd.DataFrame(strategy.trades)
            if not exit_types.empty and 'Reason' in exit_types.columns:
                exit_stats = exit_types[exit_types['Action'] == 'EXIT']['Reason'].value_counts()
                for reason, count in exit_stats.items():
                    f.write(f"  {reason}: {count} exits\n")

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return {
            'strategy': strategy,
            'signals': signals,
            'prices': prices,
            'pairs': pairs,
            'equity_curve': equity_curve
        }

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None




if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\nEnhanced Statistical Pairs Trading analysis completed successfully!")

        strategy = results['strategy']
        pairs = results['pairs']
        equity_curve = results['equity_curve']

        print("\nStrategy Performance:")
        total_return = (equity_curve['portfolio_return'] + 1).prod() - 1
        sharpe = np.sqrt(252) * (equity_curve['portfolio_return'].mean() /
                                equity_curve['portfolio_return'].std())

        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {equity_curve['drawdown'].min():.2%}")

        print("\nPairs Analysis:")
        for pair in pairs:
            stats = strategy.positions.get(pair)
            if stats:
                print(f"\n{pair[0]} - {pair[1]}:")
                print(f"Hedge Ratio: {stats.hedge_ratio:.4f}")
                print(f"Half-Life: {stats.half_life:.1f} days")
                print(f"Correlation: {stats.correlation:.4f}")
                print(f"Composite Score: {stats.composite_score:.4f}")
                if stats.regime_metrics:
                    print(f"Current Regime: {stats.regime_metrics.regime_type}")
    else:
        print("\nAnalysis failed. Check logs for details.")

