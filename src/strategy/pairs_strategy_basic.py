"""
Basic Pairs Trading Strategy Module

This module contains a basic implementation of pairs trading strategy.
"""

from typing import List, Tuple
import pandas as pd
from statsmodels.regression.linear_model import OLS
from src.analysis.cointegration import calculate_half_life
from config.logging_config import logger

from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.analysis.cointegration import determine_cointegration


class PairsTrader:
    """Basic pairs trading strategy implementation."""

    def __init__(self,
                correlation_threshold: float = 0.8,
                lookback_period: int = 20,
                entry_threshold: float = 1.5,
                exit_threshold: float = 0.5):
        """
        Initialize the pairs trading strategy.

        Args:
            correlation_threshold: Minimum correlation coefficient to consider a pair
            lookback_period: Period for calculating statistics
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
        """
        self.correlation_threshold = correlation_threshold
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        self.correlation_analyzer = None

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Find pairs suitable for trading based on correlation and cointegration.

        Args:
            prices: DataFrame of asset prices

        Returns:
            List of pairs (tuples of asset names)
        """
        returns = prices.pct_change().dropna()

        self.correlation_analyzer = CorrelationAnalyzer(returns)

        corr_pairs = self.correlation_analyzer.get_highly_correlated_pairs(
            correlation_type='pearson',
            threshold=self.correlation_threshold
        )

        cointegrated_pairs = []
        for _, row in corr_pairs.iterrows():
            asset1, asset2 = row['asset1'], row['asset2']
            is_coint, _, _, _ = determine_cointegration(
                prices[asset1],
                prices[asset2]
            )
            if is_coint:
                cointegrated_pairs.append((asset1, asset2))

        return cointegrated_pairs

    def calculate_spread(self, prices: pd.DataFrame, pair: Tuple[str, str]) -> pd.Series:
        """
        Calculate the spread between two assets.

        Args:
            prices: DataFrame of asset prices
            pair: Tuple of asset names

        Returns:
            Series containing the spread
        """
        asset1, asset2 = pair

        model = OLS(prices[asset1], prices[asset2]).fit()
        hedge_ratio = model.params[0]

        spread = prices[asset1] - hedge_ratio * prices[asset2]
        return spread

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for pairs.

        Args:
            prices: DataFrame of asset prices

        Returns:
            DataFrame containing signals for each pair, where:
            - index is dates
            - columns are tuples of (asset1, asset2)
            - values are the signal values (-1, 0, 1)
        """
        pairs = self.find_pairs(prices)

        signals = pd.DataFrame(index=prices.index)

        for pair in pairs:
            asset1, asset2 = pair
            spread = self.calculate_spread(prices, pair)

            rolling_mean = spread.rolling(window=self.lookback_period).mean()
            rolling_std = spread.rolling(window=self.lookback_period).std()
            zscore = (spread - rolling_mean) / rolling_std

            pair_signals = pd.Series(0, index=prices.index)

            pair_signals.loc[zscore < -self.entry_threshold] = 1
            pair_signals.loc[zscore > -self.exit_threshold] &= 0

            pair_signals.loc[zscore > self.entry_threshold] = -1
            pair_signals.loc[zscore < self.exit_threshold] &= 0

            signals[pair] = pair_signals

        return signals


def main():
    """Example usage of basic pairs trading strategy."""
    import yfinance as yf
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    try:
        logger.info("Starting pairs trading analysis...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)

        symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM']
        logger.info(f"Downloading data for {len(symbols)} symbols...")

        prices = pd.DataFrame()
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date)
                prices[symbol] = data['Close']
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")

        prices = prices.dropna(axis=1)
        logger.info(f"Successfully downloaded data for {len(prices.columns)} symbols")

        trader = PairsTrader(
            correlation_threshold=0.8,
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5
        )

        logger.info("Finding suitable pairs...")
        pairs = trader.find_pairs(prices)
        logger.info(f"Found {len(pairs)} cointegrated pairs")

        if not pairs:
            logger.warning("No suitable pairs found. Try adjusting parameters.")
            return None

        logger.info("Generating trading signals...")
        signals = trader.generate_signals(prices)

        results = []
        for pair in pairs:
            asset1, asset2 = pair
            spread = trader.calculate_spread(prices, pair)
            pair_signals = signals[pair]

            mean_spread = spread.mean()
            std_spread = spread.std()
            half_life = calculate_half_life(spread)

            results.append({
                'pair': f"{asset1}/{asset2}",
                'correlation': prices[asset1].corr(prices[asset2]),
                'mean_spread': mean_spread,
                'std_spread': std_spread,
                'half_life': half_life,
                'num_signals': (pair_signals != 0).sum()
            })

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=[f'Price Movement: {asset1} vs {asset2}',
                                                'Spread and Signals'])

            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[asset1], name=asset1),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[asset2], name=asset2),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=spread.index, y=spread, name='Spread'),
                row=2, col=1
            )

            fig.add_hline(y=mean_spread + trader.entry_threshold * std_spread,
                          line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=mean_spread - trader.entry_threshold * std_spread,
                          line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=mean_spread + trader.exit_threshold * std_spread,
                          line_dash="dot", line_color="gray", row=2, col=1)
            fig.add_hline(y=mean_spread - trader.exit_threshold * std_spread,
                          line_dash="dot", line_color="gray", row=2, col=1)

            long_signals = pair_signals == 1
            short_signals = pair_signals == -1

            if long_signals.any():
                fig.add_trace(
                    go.Scatter(x=spread.index[long_signals],
                               y=spread[long_signals],
                               mode='markers',
                               marker=dict(color='green', size=10),
                               name='Long Signal'),
                    row=2, col=1
                )

            if short_signals.any():
                fig.add_trace(
                    go.Scatter(x=spread.index[short_signals],
                               y=spread[short_signals],
                               mode='markers',
                               marker=dict(color='red', size=10),
                               name='Short Signal'),
                    row=2, col=1
                )

            fig.update_layout(height=800, title_text=f"Pair Analysis: {asset1}/{asset2}")
            fig.show()

        results_df = pd.DataFrame(results)
        print("\nPairs Trading Analysis Results:")
        print(results_df.to_string(index=False))

        return {
            'pairs': pairs,
            'signals': signals,
            'prices': prices,
            'results': results_df,
            'trader': trader
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