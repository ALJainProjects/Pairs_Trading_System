"""
Enhanced Basic Pairs Trading Strategy Module

Implements a pairs trading strategy with full integration into the backtesting system,
proper risk management, and optimization support.
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
from dataclasses import dataclass, field
from statsmodels.regression.linear_model import OLS
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import DATA_DIR
from src.strategy.base import BaseStrategy
from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.analysis.cointegration import determine_cointegration
from config.logging_config import logger


@dataclass
class PairPosition:
    """Track position details for a pair."""
    asset1: str
    asset2: str
    quantity: float
    entry_price1: float
    entry_price2: float
    entry_date: pd.Timestamp
    hedge_ratio: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_spread: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown: float = 0.0
    transaction_costs: float = 0.0
    trades: List[Dict] = field(default_factory=list)

    def update_metrics(self, current_price1: float, current_price2: float) -> None:
        """Update position metrics."""
        current_spread = current_price1 - self.hedge_ratio * current_price2
        spread_pnl = self.quantity * (current_spread - self.entry_spread)

        self.peak_pnl = max(self.peak_pnl, spread_pnl)
        self.current_drawdown = (self.peak_pnl - spread_pnl) / abs(self.peak_pnl) if self.peak_pnl != 0 else 0


class PairsTrader(BaseStrategy):
    """Enhanced pairs trading strategy with full backtesting integration."""

    def __init__(
            self,
            correlation_threshold: float = 0.8,
            lookback_period: int = 20,
            entry_threshold: float = 1.5,
            exit_threshold: float = 0.5,
            max_position_size: float = 0.1,
            stop_loss: float = 0.02,
            take_profit: float = 0.04,
            max_drawdown: float = 0.2,
            max_pairs: int = 10,
            transaction_cost: float = 0.001
    ):
        """Initialize the strategy."""
        super().__init__(
            name="PairsTrader",
            max_position_size=max_position_size
        )

        self.correlation_threshold = correlation_threshold
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.max_pairs = max_pairs
        self.transaction_cost = transaction_cost

        self.correlation_analyzer = None
        self.pairs: List[Tuple[str, str]] = []
        self.positions: Dict[Tuple[str, str], PairPosition] = {}
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}

        self.validate_parameters()
        self._portfolio_value = 0.0

    @property.fget
    def get_current_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self._portfolio_value

    @property.fset
    def set_current_portfolio_value(self, value: float) -> None:
        """Set current portfolio value."""
        self._portfolio_value = value

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if not 0 < self.correlation_threshold < 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if self.lookback_period < 1:
            raise ValueError("lookback_period must be positive")
        if self.entry_threshold < self.exit_threshold:
            raise ValueError("entry_threshold must be greater than exit_threshold")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        if not 0 < self.stop_loss < 1:
            raise ValueError("stop_loss must be between 0 and 1")
        if not 0 < self.take_profit < 1:
            raise ValueError("take_profit must be between 0 and 1")

    def reset(self) -> None:
        """Reset strategy state between optimization runs."""
        self.clear_state()
        self.correlation_analyzer = None
        self.pairs = []
        self.positions = {}
        self.hedge_ratios = {}

    @property
    def optimization_parameters(self) -> Dict:
        """Define parameters for optimization."""
        return {
            'correlation_threshold': (0.5, 0.9),
            'lookback_period': (10, 50),
            'entry_threshold': (1.0, 3.0),
            'exit_threshold': (0.2, 1.0),
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.08)
        }

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Find suitable trading pairs based on correlation and cointegration.

        Args:
            prices: Price data for all assets

        Returns:
            List of pairs (tuples of asset names)
        """
        returns = prices.pct_change().dropna()

        try:
            self.correlation_analyzer = CorrelationAnalyzer(returns)

            corr_pairs = self.correlation_analyzer.get_highly_correlated_pairs(
                correlation_type='pearson',
                threshold=self.correlation_threshold
            )

            cointegrated_pairs = []
            for _, row in corr_pairs.iterrows():
                asset1, asset2 = row['asset1'], row['asset2']

                if len(cointegrated_pairs) >= self.max_pairs:
                    break

                try:
                    is_coint, _, p_value, _ = determine_cointegration(
                        prices[asset1],
                        prices[asset2]
                    )

                    if is_coint:
                        hedge_ratio = self.calculate_hedge_ratio(
                            prices[asset1],
                            prices[asset2]
                        )

                        cointegrated_pairs.append((asset1, asset2))
                        self.hedge_ratios[(asset1, asset2)] = hedge_ratio

                        logger.info(f"Found cointegrated pair: {asset1}-{asset2} "
                                  f"(p-value: {p_value:.4f}, hedge ratio: {hedge_ratio:.4f})")

                except Exception as e:
                    logger.warning(f"Error testing cointegration for {asset1}-{asset2}: {str(e)}")
                    continue

            return cointegrated_pairs

        except Exception as e:
            logger.error(f"Error in find_pairs: {str(e)}")
            return []

    def calculate_hedge_ratio(self, series1: pd.Series, series2: pd.Series, lookback: Optional[int] = None) -> float:
        """
        Calculate hedge ratio between two price series.

        Args:
            series1: First price series
            series2: Second price series
            lookback: Optional lookback period for calculation

        Returns:
            float: Calculated hedge ratio
        """
        try:
            if lookback:
                series1 = series1[-lookback:]
                series2 = series2[-lookback:]

            model = OLS(series1, series2).fit()
            return model.params[0]
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {str(e)}")
            return 1.0

    def clear_state(self) -> None:
        """Clear strategy state between runs instead of using reset."""
        self.correlation_analyzer = None
        self.pairs = []
        self.positions = {}
        self.hedge_ratios = {}
        self._portfolio_value = 0.0

    def calculate_spread(self, prices: pd.DataFrame, pair: Tuple[str, str]) -> pd.Series:
        """
        Calculate the spread between two assets.

        Args:
            prices: Price data
            pair: Asset pair tuple

        Returns:
            Series containing the spread
        """
        asset1, asset2 = pair
        hedge_ratio = self.hedge_ratios.get(pair, 1.0)
        return prices[asset1] - hedge_ratio * prices[asset2]

    def calculate_position_size(self,
                              pair: Tuple[str, str],
                              prices: pd.DataFrame,
                              portfolio_value: float) -> float:
        """
        Calculate position size based on spread volatility and portfolio constraints.

        Args:
            pair: Asset pair
            prices: Price data
            portfolio_value: Current portfolio value

        Returns:
            float: Position size in units
        """
        asset1, asset2 = pair
        price1 = prices[asset1].iloc[-1]
        price2 = prices[asset2].iloc[-1]

        spread = self.calculate_spread(prices.iloc[-self.lookback_period:], pair)
        spread_vol = spread.std()

        vol_adjustment = 1.0 / (1.0 + spread_vol) if spread_vol > 0 else 1.0

        max_trade_value = portfolio_value * self.max_position_size
        pair_value = price1 + (self.hedge_ratios.get(pair, 1.0) * price2)

        position_size = (max_trade_value / pair_value) * vol_adjustment

        position_size *= (1 - self.transaction_cost)

        return position_size

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for pairs."""
        if not self.pairs:
            self.pairs = self.find_pairs(prices)

        signals = pd.DataFrame(index=prices.index)

        if hasattr(self, 'positions') and self.positions:
            portfolio_value = sum(
                position.quantity * (
                        current_prices[position.asset1] +
                        position.hedge_ratio * current_prices[position.asset2]
                )
                for position, current_prices in zip(
                    self.positions.values(),
                    prices.itertuples()
                )
            )
            self._portfolio_value = portfolio_value

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                if asset1 not in prices.columns or asset2 not in prices.columns:
                    continue

                spread = self.calculate_spread(prices, pair)

                rolling_mean = spread.rolling(window=self.lookback_period).mean()
                rolling_std = spread.rolling(window=self.lookback_period).std()
                zscore = (spread - rolling_mean) / rolling_std

                pair_signals = pd.Series(0, index=prices.index)

                pair_signals.loc[zscore < -self.entry_threshold] = 1
                pair_signals.loc[zscore > self.entry_threshold] = -1

                pair_signals.loc[zscore > -self.exit_threshold] &= 0
                pair_signals.loc[zscore < self.exit_threshold] &= 0

                if not pair_signals.empty and pair_signals.iloc[-1] != 0:
                    position_size = self.calculate_position_size(
                        pair,
                        prices,
                        self._portfolio_value
                    )
                    pair_signals *= position_size

                signals[pair] = pair_signals

            except Exception as e:
                logger.error(f"Error generating signals for {pair}: {str(e)}")
                continue

        return signals

    def update_positions(self,
                        current_prices: pd.Series,
                        timestamp: pd.Timestamp) -> None:
        """
        Update position metrics and check risk limits.

        Args:
            current_prices: Current asset prices
            timestamp: Current timestamp
        """
        for pair, position in list(self.positions.items()):
            asset1, asset2 = pair

            if asset1 not in current_prices or asset2 not in current_prices:
                self._close_position(timestamp, pair)
                continue

            current_price1 = current_prices[asset1]
            current_price2 = current_prices[asset2]

            position.update_metrics(current_price1, current_price2)

            if position.current_drawdown > self.stop_loss:
                logger.info(f"Stop loss triggered for {pair}")
                self._close_position(timestamp, pair)
                continue

            current_spread = current_price1 - position.hedge_ratio * current_price2
            profit_pct = abs(current_spread - position.entry_spread) / abs(position.entry_spread)

            if profit_pct > self.take_profit:
                logger.info(f"Take profit triggered for {pair}")
                self._close_position(timestamp, pair)

    def _close_position(self,
                        date: pd.Timestamp,
                        pair: Tuple[str, str],
                        reason: str = None) -> None:
        """Close a position and record the trade."""
        if pair in self.positions:
            position = self.positions[pair]

            trade_record = {
                'pair': f"{position.asset1}/{position.asset2}",
                'entry_date': position.entry_date,
                'exit_date': date,
                'entry_spread': position.entry_spread,
                'quantity': position.quantity,
                'pnl': position.peak_pnl,
                'max_drawdown': position.current_drawdown,
                'transaction_costs': position.transaction_costs
            }

            self.trades.append(trade_record)
            del self.positions[pair]

    def get_portfolio_stats(self) -> Dict:
        """Calculate portfolio statistics."""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)
        stats = {
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'total_pnl': trades_df['pnl'].sum(),
            'total_costs': trades_df['transaction_costs'].sum(),
            'max_drawdown': trades_df['max_drawdown'].max(),
            'avg_trade_duration': (trades_df['exit_date'] - trades_df['entry_date']).mean().days,
            'pairs_traded': len(trades_df['pair'].unique())
        }

        if len(trades_df) > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
            stats['avg_profit_per_trade'] = stats['total_pnl'] / stats['total_trades']

        return stats


def main():
    """Test the pairs trading strategy using local CSV data and generate outputs."""
    output_dir = Path("pairs_trading_strategy_basic_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    results_dir = output_dir / "results"

    for directory in [plots_dir, data_dir, results_dir]:
        directory.mkdir(exist_ok=True)

    try:
        logger.info("Reading data from local CSV files...")
        raw_data = DATA_DIR.replace(r'\config', '')
        raw_data_dir = Path(f"{raw_data}/raw")

        selected_symbols = [
            'AAPL',
            'MSFT',
            'NVDA',
            'AMD',
            'INTC',
            'QCOM',
            'AVGO',
            'ASML',
            'AMAT',
            'MU'
        ]

        logger.info(f"Selected stocks for analysis: {', '.join(selected_symbols)}")

        prices = pd.DataFrame()
        for symbol in selected_symbols:
            try:
                csv_path = raw_data_dir / f"{symbol}.csv"
                if not csv_path.exists():
                    logger.warning(f"Data file not found for {symbol}")
                    continue

                df = pd.read_csv(csv_path)
                if 'Date' not in df.columns or 'Adj_Close' not in df.columns:
                    logger.warning(f"Required columns missing in {symbol}")
                    continue

                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                prices[symbol] = df['Adj_Close']
                logger.debug(f"Successfully loaded data for {symbol}")
            except Exception as e:
                logger.error(f"Error reading data for {symbol}: {str(e)}")
                continue

        if prices.empty:
            raise ValueError("No valid price data loaded")

        prices = prices.ffill().bfill()
        logger.info(f"Successfully loaded data for {len(prices.columns)} symbols")

        strategy = PairsTrader(
            correlation_threshold=0.7,
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            max_position_size=0.1,
            stop_loss=0.02,
            take_profit=0.04,
            max_drawdown=0.2,
            max_pairs=5
        )

        logger.info("Finding trading pairs...")
        pairs = strategy.find_pairs(prices)
        if not pairs:
            logger.warning("No valid trading pairs found")
            return None

        pair_analysis = []
        for pair in pairs:
            asset1, asset2 = pair
            correlation = prices[asset1].corr(prices[asset2])
            spread = strategy.calculate_spread(prices, pair)
            hedge_ratio = strategy.hedge_ratios[pair]

            analysis = {
                'pair': f"{asset1}/{asset2}",
                'correlation': correlation,
                'hedge_ratio': hedge_ratio,
                'spread_mean': spread.mean(),
                'spread_std': spread.std()
            }
            pair_analysis.append(analysis)

        pd.DataFrame(pair_analysis).to_csv(results_dir / "pair_analysis.csv", index=False)

        logger.info("Generating trading signals...")
        signals = strategy.generate_signals(prices)
        signals.to_csv(results_dir / "trading_signals.csv")

        logger.info("Creating visualizations...")
        for pair in pairs:
            asset1, asset2 = pair
            spread = strategy.calculate_spread(prices, pair)
            pair_signals = signals[pair]

            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    f'Price Movement: {asset1} vs {asset2}',
                    'Normalized Prices',
                    'Spread and Signals'
                ],
                vertical_spacing=0.1
            )

            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[asset1], name=asset1),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[asset2], name=asset2),
                row=1, col=1
            )

            norm_price1 = prices[asset1] / prices[asset1].iloc[0]
            norm_price2 = prices[asset2] / prices[asset2].iloc[0]

            fig.add_trace(
                go.Scatter(x=prices.index, y=norm_price1, name=f"{asset1} (norm)"),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=prices.index, y=norm_price2, name=f"{asset2} (norm)"),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=spread.index, y=spread, name='Spread'),
                row=3, col=1
            )

            mean_spread = spread.rolling(window=strategy.lookback_period).mean()
            std_spread = spread.rolling(window=strategy.lookback_period).std()

            fig.add_trace(
                go.Scatter(
                    x=spread.index,
                    y=mean_spread + strategy.entry_threshold * std_spread,
                    name='Upper Entry',
                    line=dict(dash='dash', color='red')
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=spread.index,
                    y=mean_spread - strategy.entry_threshold * std_spread,
                    name='Lower Entry',
                    line=dict(dash='dash', color='red')
                ),
                row=3, col=1
            )

            long_signals = pair_signals > 0
            short_signals = pair_signals < 0

            if long_signals.any():
                fig.add_trace(
                    go.Scatter(
                        x=spread.index[long_signals],
                        y=spread[long_signals],
                        mode='markers',
                        marker=dict(color='green', size=10),
                        name='Long Signal'
                    ),
                    row=3, col=1
                )

            if short_signals.any():
                fig.add_trace(
                    go.Scatter(
                        x=spread.index[short_signals],
                        y=spread[short_signals],
                        mode='markers',
                        marker=dict(color='red', size=10),
                        name='Short Signal'
                    ),
                    row=3, col=1
                )

            fig.update_layout(
                height=1200,
                title_text=f"Pair Analysis: {asset1}/{asset2}",
                showlegend=True
            )

            fig.write_html(plots_dir / f"pair_analysis_{asset1}_{asset2}.html")

        strategy_config = {
            'correlation_threshold': strategy.correlation_threshold,
            'lookback_period': strategy.lookback_period,
            'entry_threshold': strategy.entry_threshold,
            'exit_threshold': strategy.exit_threshold,
            'max_position_size': strategy.max_position_size,
            'stop_loss': strategy.stop_loss,
            'take_profit': strategy.take_profit,
            'max_drawdown': strategy.max_drawdown,
            'max_pairs': strategy.max_pairs
        }

        with open(results_dir / "strategy_config.json", 'w') as f:
            json.dump(strategy_config, f, indent=4)

        with open(results_dir / "analysis_summary.txt", 'w') as f:
            f.write("Pairs Trading Strategy Analysis Summary\n")
            f.write("=====================================\n\n")

            f.write(f"Analysis Period: {prices.index[0].date()} to {prices.index[-1].date()}\n")
            f.write(f"Number of assets analyzed: {len(prices.columns)}\n")
            f.write(f"Number of trading pairs found: {len(pairs)}\n\n")

            f.write("Trading Pairs:\n")
            for pair_info in pair_analysis:
                f.write(f"\n{pair_info['pair']}:\n")
                f.write(f"  Correlation: {pair_info['correlation']:.4f}\n")
                f.write(f"  Hedge Ratio: {pair_info['hedge_ratio']:.4f}\n")
                f.write(f"  Spread Mean: {pair_info['spread_mean']:.4f}\n")
                f.write(f"  Spread Std: {pair_info['spread_std']:.4f}\n")

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return {
            'pairs': pairs,
            'signals': signals,
            'prices': prices,
            'strategy': strategy,
            'pair_analysis': pair_analysis
        }

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\nPairs trading analysis completed successfully!")
    else:
        print("\nAnalysis failed. Check logs for details.")