"""
Enhanced Statistical Pairs Trading Strategy

Features:
1. Proper BaseStrategy integration
2. Efficient statistical calculations with caching
3. Comprehensive risk management
4. No look-ahead bias
5. Optimization support
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import plotly.graph_objects as go
from pathlib import Path
import json

from src.strategy.base import BaseStrategy
from config.logging_config import logger
from config.settings import DATA_DIR


@dataclass
class PairStats:
    """Track pair statistics and risk metrics."""
    hedge_ratio: float
    half_life: float
    coint_pvalue: float
    spread_zscore: float
    spread_vol: float
    correlation: float
    last_update: pd.Timestamp
    lookback_data: Dict[str, pd.Series] = field(default_factory=dict)


class StatisticalCalculator:
    """Efficient statistical calculations with caching."""

    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.pair_stats: Dict[Tuple[str, str], PairStats] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def calculate_hedge_ratio(self,
                            price1: pd.Series,
                            price2: pd.Series,
                            window: int = 252) -> float:
        """Calculate hedge ratio using rolling OLS."""
        if len(price1) < window:
            return 1.0

        try:
            X = add_constant(price2.iloc[-window:])
            y = price1.iloc[-window:]
            model = OLS(y, X).fit()
            return model.params[1]
        except:
            return 1.0

    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life."""
        try:
            lagged_spread = spread.shift(1)
            delta_spread = spread - lagged_spread
            X = add_constant(lagged_spread.iloc[1:])
            model = OLS(delta_spread.iloc[1:], X).fit()
            gamma = model.params[1]
            half_life = -np.log(2) / gamma if gamma < 0 else np.inf
            return half_life
        except:
            return np.inf

    def calculate_spread_zscore(self,
                              spread: pd.Series,
                              window: int = 20) -> float:
        """Calculate z-score of spread."""
        if len(spread) < window:
            return 0.0

        try:
            rolling_mean = spread.rolling(window=window).mean()
            rolling_std = spread.rolling(window=window).std()
            zscore = (spread - rolling_mean) / rolling_std
            return zscore.iloc[-1]
        except:
            return 0.0

    def update_pair_stats(self,
                         pair: Tuple[str, str],
                         price1: pd.Series,
                         price2: pd.Series,
                         force_update: bool = False) -> PairStats:
        """Update statistical metrics for a pair."""
        current_time = price1.index[-1]

        if pair in self.pair_stats and not force_update:
            stats = self.pair_stats[pair]
            if (current_time - stats.last_update) <= pd.Timedelta(days=1):
                self.cache_hits += 1
                return stats

        self.cache_misses += 1

        hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        spread = price1 - hedge_ratio * price2
        half_life = self.calculate_half_life(spread)
        zscore = self.calculate_spread_zscore(spread)
        spread_vol = spread.std()
        correlation = price1.corr(price2)

        _, coint_pvalue, _ = coint(price1, price2)

        stats = PairStats(
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            coint_pvalue=coint_pvalue,
            spread_zscore=zscore,
            spread_vol=spread_vol,
            correlation=correlation,
            last_update=current_time,
            lookback_data={'spread': spread}
        )

        self.pair_stats[pair] = stats

        if len(self.pair_stats) > self.cache_size:
            oldest_pair = min(self.pair_stats.items(), key=lambda x: x[1].last_update)[0]
            del self.pair_stats[oldest_pair]

        return stats


class EnhancedStatPairsStrategy(BaseStrategy):
    """Enhanced statistical pairs trading strategy."""

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
            max_drawdown: float = 0.2
    ):
        """Initialize strategy with enhanced parameters."""
        super().__init__(name="EnhancedStatPairs")

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

        self.calculator = StatisticalCalculator()
        self.pairs: List[Tuple[str, str]] = []
        self.positions: Dict[Tuple[str, str], PairStats] = {}
        self.trades: List[Dict] = []

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find cointegrated pairs meeting criteria."""
        logger.info("Searching for valid trading pairs")
        valid_pairs = []
        total_checked = 0

        for i in range(len(prices.columns)):
            for j in range(i + 1, len(prices.columns)):
                total_checked += 1
                asset1, asset2 = prices.columns[i], prices.columns[j]

                try:
                    stats = self.calculator.update_pair_stats(
                        (asset1, asset2),
                        prices[asset1],
                        prices[asset2],
                        force_update=True
                    )

                    if (stats.correlation >= self.min_correlation and
                        stats.coint_pvalue <= self.coint_threshold and
                        self.min_half_life <= stats.half_life <= self.max_half_life and
                        stats.spread_vol <= self.max_spread_vol):

                        valid_pairs.append((asset1, asset2))
                        logger.debug(f"Found valid pair: {asset1}-{asset2}")

                        if len(valid_pairs) >= self.max_pairs:
                            break
                except Exception as e:
                    logger.warning(f"Error analyzing pair {asset1}-{asset2}: {str(e)}")
                    continue

        logger.info(f"Found {len(valid_pairs)} valid pairs out of {total_checked} checked")
        return valid_pairs

    def calculate_position_size(self,
                              pair: Tuple[str, str],
                              stats: PairStats,
                              prices: pd.DataFrame) -> float:
        """Calculate dynamic position size based on risk metrics."""
        try:
            asset1, asset2 = pair
            price1 = prices[asset1].iloc[-1]
            price2 = prices[asset2].iloc[-1]

            vol_adjustment = 1.0 / (1.0 + stats.spread_vol)

            hl_adjustment = np.exp(-stats.half_life / self.max_half_life)

            base_size = self.position_size * vol_adjustment * hl_adjustment

            pair_value = price1 + stats.hedge_ratio * price2
            position_size = base_size / pair_value

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for pairs."""
        signals = pd.DataFrame(index=prices.index, columns=self.pairs)

        if not self.pairs:
            self.pairs = self.find_pairs(prices)

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                stats = self.calculator.update_pair_stats(
                    pair,
                    prices[asset1],
                    prices[asset2]
                )

                zscore = stats.spread_zscore
                position_size = self.calculate_position_size(pair, stats, prices)

                if abs(zscore) >= self.zscore_entry:
                    signals[pair] = -np.sign(zscore) * position_size
                elif abs(zscore) <= self.zscore_exit and pair in self.positions:
                    signals[pair] = 0
                else:
                    signals[pair] = signals[pair].shift(1).fillna(0)

            except Exception as e:
                logger.error(f"Error generating signals for {pair}: {str(e)}")
                signals[pair] = 0

        return signals

    def update_positions(self, prices: pd.DataFrame) -> None:
        """Update position metrics and check risk limits."""
        current_time = prices.index[-1]

        for pair in list(self.positions.keys()):
            try:
                asset1, asset2 = pair
                stats = self.calculator.update_pair_stats(
                    pair,
                    prices[asset1],
                    prices[asset2]
                )

                if abs(stats.spread_zscore) > self.stop_loss:
                    self._close_position(current_time, pair, "Stop loss")
                    continue

                spread = stats.lookback_data['spread']
                drawdown = (spread.max() - spread.iloc[-1]) / spread.max()
                if drawdown > self.max_drawdown:
                    self._close_position(current_time, pair, "Max drawdown")
                    continue

            except Exception as e:
                logger.error(f"Error updating position for {pair}: {str(e)}")
                self._close_position(current_time, pair, "Error")

    def _close_position(self,
                       timestamp: pd.Timestamp,
                       pair: Tuple[str, str],
                       reason: str = None) -> None:
        """Close a position and record trade details."""
        if pair not in self.positions:
            return

        try:
            stats = self.positions[pair]
            asset1, asset2 = pair

            trade = {
                'timestamp': timestamp,
                'pair': f"{asset1}/{asset2}",
                'action': 'EXIT',
                'reason': reason,
                'hedge_ratio': stats.hedge_ratio,
                'half_life': stats.half_life,
                'spread_zscore': stats.spread_zscore
            }

            self.trades.append(trade)
            del self.positions[pair]

        except Exception as e:
            logger.error(f"Error closing position for {pair}: {str(e)}")

    def optimize_parameters(self,
                          prices: pd.DataFrame,
                          param_grid: Dict) -> Dict:
        """Optimize strategy parameters using grid search."""
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

    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations for optimization."""
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]

    def _calculate_returns(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy returns for all pairs.

        Args:
            signals: DataFrame with trading signals
            prices: DataFrame with asset prices

        Returns:
            DataFrame with returns and metrics
        """
        returns = pd.DataFrame(index=signals.index)
        returns['portfolio_return'] = 0.0

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                if asset1 not in prices.columns or asset2 not in prices.columns:
                    continue

                stats = self.calculator.pair_stats.get(pair)
                if not stats:
                    continue

                rets1 = prices[asset1].pct_change()
                rets2 = prices[asset2].pct_change()

                pair_signals = signals[pair].shift(1)
                pair_returns = pair_signals * (rets1 - stats.hedge_ratio * rets2)

                trades = pair_signals.diff().fillna(0) != 0
                transaction_costs = trades * self.transaction_cost_pct

                net_returns = pair_returns - transaction_costs
                returns[f'{asset1}_{asset2}_return'] = net_returns

                position_size = self.calculate_position_size(pair, stats, prices)
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
        """Calculate drawdown series."""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = np.sqrt(252) * (rolling_mean / rolling_std)
        return rolling_sharpe

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return -np.inf

        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)

        return annualized_return / annualized_vol if annualized_vol > 0 else -np.inf

    def save_state(self, path: str) -> None:
        """Save strategy state and parameters."""
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
                'coint_threshold': self.coint_threshold
            },
            'pairs': self.pairs,
            'trades': self.trades,
            'calculator_stats': {
                'cache_hits': self.calculator.cache_hits,
                'cache_misses': self.calculator.cache_misses
            }
        }

        with open(state_path / "strategy_state.json", 'w') as f:
            json.dump(state, f, indent=4)

        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(state_path / "trade_history.csv", index=False)

    def load_state(self, path: str) -> None:
        """Load strategy state and parameters."""
        state_path = Path(path)

        with open(state_path / "strategy_state.json", 'r') as f:
            state = json.load(f)

        for param, value in state['parameters'].items():
            setattr(self, param, value)

        self.pairs = state['pairs']
        self.trades = state['trades']

        trade_history_path = state_path / "trade_history.csv"
        if trade_history_path.exists():
            self.trades = pd.read_csv(trade_history_path).to_dict('records')

    def plot_pairs_analysis(self, prices: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
        """Plot comprehensive pair analysis."""
        if not self.pairs:
            logger.warning("No pairs available for plotting")
            return

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                stats = self.calculator.update_pair_stats(
                    pair,
                    prices[asset1],
                    prices[asset2]
                )

                fig = self._create_pair_plot(
                    prices[asset1],
                    prices[asset2],
                    stats
                )

                if output_dir:
                    fig.write_html(output_dir / f"pair_analysis_{asset1}_{asset2}.html")
                else:
                    fig.show()

            except Exception as e:
                logger.error(f"Error plotting pair analysis for {pair}: {str(e)}")

    def _create_pair_plot(self,
                         price1: pd.Series,
                         price2: pd.Series,
                         stats: PairStats) -> go.Figure:
        """Create detailed plot for pair analysis."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=price1.index,
            y=price1,
            name=price1.name,
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=price2.index,
            y=price2 * stats.hedge_ratio,
            name=f"{price2.name} (Adjusted)",
            line=dict(color='red')
        ))

        spread = stats.lookback_data['spread']
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=spread,
            name='Spread',
            line=dict(color='green'),
            yaxis='y2'
        ))

        fig.update_layout(
            title=f"Pair Analysis: {price1.name} vs {price2.name}",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Spread", overlaying='y', side='right'),
            height=800,
            showlegend=True
        )

        return fig

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """
        Return a dictionary of default parameters suitable for the
        EnhancedStatPairsStrategy.
        """
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
            "max_drawdown": 0.2
        }

def main():
    """Test the enhanced statistical pairs trading strategy."""
    output_dir = Path("pairs_trading_strategy_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    results_dir = output_dir / "results"

    for directory in [models_dir, plots_dir, data_dir, results_dir]:
        directory.mkdir(exist_ok=True)

    try:
        logger.info("Starting statistical pairs trading analysis...")

        raw_data_dir = Path(DATA_DIR.replace(r'\config', '')) / "raw"
        selected_symbols = [
            'AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC',
            'QCOM', 'AVGO', 'ASML', 'AMAT', 'MU'
        ]

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
                prices[symbol] = df['Adj Close']

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
            zscore_exit=0.0,
            min_half_life=5,
            max_half_life=126,
            max_spread_vol=0.1,
            min_correlation=0.5,
            coint_threshold=0.05,
            max_pairs=5,
            position_size=0.1,
            stop_loss=0.02,
            max_drawdown=0.2
        )

        logger.info("Finding cointegrated pairs...")
        pairs = strategy.find_pairs(prices)

        logger.info("Generating trading signals...")
        signals = strategy.generate_signals(prices)
        signals.to_csv(results_dir / "trading_signals.csv")

        logger.info("Creating pair analysis plots...")
        strategy.plot_pairs_analysis(prices, plots_dir)

        if True:
            logger.info("Optimizing strategy parameters...")
            param_grid = {
                'zscore_entry': [1.5, 2.0, 2.5],
                'zscore_exit': [0.0, 0.5],
                'min_half_life': [5, 10],
                'max_half_life': [100, 126]
            }

            best_params = strategy.optimize_parameters(prices, param_grid)

            with open(results_dir / "optimization_results.json", 'w') as f:
                json.dump(best_params, f, indent=4)

        strategy.save_state(str(models_dir / "final_state"))

        with open(results_dir / "analysis_summary.txt", 'w') as f:
            f.write("Statistical Pairs Trading Analysis\n")
            f.write("================================\n\n")

            f.write(f"Analysis Period: {prices.index[0].date()} to {prices.index[-1].date()}\n")
            f.write(f"Number of assets analyzed: {len(prices.columns)}\n")
            f.write(f"Number of trading pairs found: {len(strategy.pairs)}\n\n")

            f.write("Trading Pairs:\n")
            for pair in strategy.pairs:
                stats = strategy.calculator.pair_stats.get(pair)
                if stats:
                    f.write(f"\n{pair[0]} - {pair[1]}:\n")
                    f.write(f"  Hedge Ratio: {stats.hedge_ratio:.4f}\n")
                    f.write(f"  Half-Life: {stats.half_life:.1f} days\n")
                    f.write(f"  Correlation: {stats.correlation:.4f}\n")
                    f.write(f"  Cointegration p-value: {stats.coint_pvalue:.4f}\n")

            f.write("\nCalculator Statistics:\n")
            f.write(f"  Cache Hits: {strategy.calculator.cache_hits}\n")
            f.write(f"  Cache Misses: {strategy.calculator.cache_misses}\n")

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return {
            'strategy': strategy,
            'signals': signals,
            'prices': prices,
            'pairs': pairs
        }

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\nStatistical Pairs Trading analysis completed successfully!")

        strategy = results['strategy']
        pairs = results['pairs']

        print("\nPairs Analysis:")
        for pair in pairs:
            stats = strategy.calculator.pair_stats.get(pair)
            if stats:
                print(f"\n{pair[0]} - {pair[1]}:")
                print(f"Hedge Ratio: {stats.hedge_ratio:.4f}")
                print(f"Half-Life: {stats.half_life:.1f} days")
                print(f"Correlation: {stats.correlation:.4f}")
                print(f"Cointegration p-value: {stats.coint_pvalue:.4f}")
    else:
        print("\nAnalysis failed. Check logs for details.")