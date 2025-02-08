"""
Statistical Models Module

Enhancements:
 1. More robust cointegration testing using Engle-Granger and optional Johansen.
 2. Spread calculation with optional ratio-based spread.
 3. Extended docstrings for usage in a pair trading workflow.
 4. Additional trading signals including Moving Average Crossover, Bollinger Bands, and RSI
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from config.logging_config import logger
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import plotly.graph_objects as go
from plotly.subplots import make_subplots



class StatisticalModel:
    """
    The StatisticalModel provides core statistical methods relevant to pair selection and
    signal generation in pair trading. Methods included:
      - Engle-Granger cointegration test
      - Optional Johansen cointegration test (if installed)
      - Calculating a spread between two assets via regression
      - Mean reversion signal generation
      - Moving Average Crossover signals
      - Bollinger Bands signals
      - RSI-based signals
    """

    def __init__(self):
        logger.info("Initializing StatisticalModel.")
        self.market_features = {}

    def update_features(self, data: pd.DataFrame):
        self.market_features = self.calculate_microstructure_features(data)

    def calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday volatility and liquidity features from OHLCV data.

        Args:
            data: DataFrame with OHLCV data
        Returns:
            DataFrame with additional features
        """
        features = pd.DataFrame(index=data.index)

        features['parkinson_vol'] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) *
            (np.log(data['High'] / data['Low']) ** 2)
        )

        features['relative_volume'] = (
                data['Volume'] / data['Volume'].rolling(20).mean()
        )

        features['amihud_ratio'] = (
                                           np.abs(data['Adj_Close'].pct_change()) /
                                           (data['Volume'] * data['Adj_Close'])
                                   ) * 1e6

        features['trading_range'] = (
                (data['High'] - data['Low']) / data['Open']
        )

        return features

    def cointegration_test(self, asset1: pd.Series, asset2: pd.Series, significance: float = 0.05) -> bool:
        """
        Perform the Engle-Granger cointegration test between two price series.

        Args:
            asset1 (pd.Series): Price series of the first asset.
            asset2 (pd.Series): Price series of the second asset.
            significance (float): Significance threshold for cointegration (default=0.05).

        Returns:
            bool: True if cointegrated, False otherwise.
        """
        logger.info("Performing Engle-Granger cointegration test.")

        if len(asset1) != len(asset2):
            raise ValueError("Series must have the same length for Engle-Granger test.")
        if asset1.isnull().any() or asset2.isnull().any():
            raise ValueError("Series contain NaN values. Clean your data first.")

        score, pvalue, _ = coint(asset1, asset2)
        logger.debug(f"EG cointegration p-value: {pvalue:.4f}")
        return pvalue < significance

    def johansen_test(self, df_prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> bool:
        """
        (Optional) Perform Johansen cointegration test on multiple price series.

        Args:
            df_prices (pd.DataFrame): Price DataFrame with columns as different assets.
            det_order (int): Deterministic trend order (0=none, 1=constant, etc.).
            k_ar_diff (int): Lagged differences used in test.

        Returns:
            bool: True if test indicates at least one cointegration relationship, else False.
        """
        logger.info("Performing Johansen cointegration test on multiple series.")
        result = coint_johansen(df_prices, det_order, k_ar_diff)

        trace_stat = result.lr1
        crit_vals = result.cvt[:, 1]
        coint_count = sum(np.array(trace_stat) > np.array(crit_vals))
        logger.debug(f"Johansen coint relationships found: {coint_count}")
        return coint_count > 0

    def calculate_spread(self, asset1: pd.Series, asset2: pd.Series, use_ratio: bool = False) -> pd.Series:
        """
        Calculate the spread between two assets using OLS or a ratio approach.

        Args:
            asset1 (pd.Series): First asset's price series.
            asset2 (pd.Series): Second asset's price series.
            use_ratio (bool): If True, use ratio-based spread (asset1 / asset2). Else OLS regression.

        Returns:
            pd.Series: Spread series over time.
        """
        logger.info(f"Calculating spread (use_ratio={use_ratio}).")
        if use_ratio:
            spread = asset1 / asset2
        else:
            X = add_constant(asset2)
            model = OLS(asset1, X).fit()
            alpha = model.params["const"]
            beta = model.params[asset2.name]
            spread = asset1 - (beta * asset2 + alpha)
        logger.debug("Spread calculation completed.")
        return spread

    def mean_reversion_signal(self, spread: pd.Series, window: int = 20, z_threshold: float = 2.0) -> pd.Series:
        """
        Generate mean reversion trading signals based on z-score of the spread.

        Args:
            spread (pd.Series): Spread series (e.g., from calculate_spread).
            window (int): Rolling window for mean/std.
            z_threshold (float): +/- threshold to trigger signals.

        Returns:
            pd.Series: A series of signals [1, -1, 0].
        """
        logger.info("Generating mean reversion signals from spread.")
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()
        zscore = (spread - roll_mean) / (roll_std + 1e-12)

        signals = pd.Series(index=spread.index, data=0, dtype=float)
        signals[zscore > z_threshold] = -1.0
        signals[zscore < -z_threshold] = 1.0
        return signals

    def moving_average_crossover_signal(self, spread: pd.Series, short_window: int = 10, long_window: int = 50) -> pd.Series:
        """
        Generate trading signals based on moving average crossovers.

        Args:
            spread (pd.Series): Spread series.
            short_window (int): Window size for the short-term moving average.
            long_window (int): Window size for the long-term moving average.

        Returns:
            pd.Series: A series of signals [1, -1, 0].
        """
        logger.info("Generating moving average crossover signals.")
        short_ma = spread.rolling(short_window).mean()
        long_ma = spread.rolling(long_window).mean()

        signals = pd.Series(index=spread.index, data=0, dtype=float)
        signals[short_ma > long_ma] = 1.0
        signals[short_ma < long_ma] = -1.0
        return signals

    def bollinger_band_signal(self, spread: pd.Series, window: int = 20, num_std_dev: float = 2.0) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands.

        Args:
            spread (pd.Series): Spread series.
            window (int): Rolling window size for the mean and standard deviation.
            num_std_dev (float): Number of standard deviations for the bands.

        Returns:
            pd.Series: A series of signals [1, -1, 0].
        """
        logger.info("Generating Bollinger Bands signals.")
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()

        upper_band = roll_mean + (num_std_dev * roll_std)
        lower_band = roll_mean - (num_std_dev * roll_std)

        signals = pd.Series(index=spread.index, data=0, dtype=float)
        signals[spread < lower_band] = 1.0
        signals[spread > upper_band] = -1.0
        return signals

    def rsi_signal(self, spread: pd.Series, window: int = 14,
                  lower_threshold: float = 30, upper_threshold: float = 70) -> pd.Series:
        """
        Generate trading signals based on the RSI of the spread.

        Args:
            spread (pd.Series): Spread series.
            window (int): Window size for RSI calculation.
            lower_threshold (float): RSI level to trigger a long signal.
            upper_threshold (float): RSI level to trigger a short signal.

        Returns:
            pd.Series: A series of signals [1, -1, 0].
        """
        logger.info("Generating RSI-based signals.")
        delta = spread.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(index=spread.index, data=0, dtype=float)
        signals[rsi < lower_threshold] = 1.0
        signals[rsi > upper_threshold] = -1.0
        return signals

    def combine_signals(self, signal_series_list: list, weights: list = None) -> pd.Series:
        """
        Combine multiple trading signals using weighted average.

        Args:
            signal_series_list (list): List of signal series to combine.
            weights (list): List of weights for each signal series. If None, equal weights are used.

        Returns:
            pd.Series: Combined trading signals [-1, 0, 1].
        """
        logger.info("Combining multiple trading signals.")
        if weights is None:
            weights = [1/len(signal_series_list)] * len(signal_series_list)

        if len(weights) != len(signal_series_list):
            raise ValueError("Number of weights must match number of signal series.")

        combined = pd.Series(0, index=signal_series_list[0].index)
        for signal, weight in zip(signal_series_list, weights):
            combined += signal * weight

        final_signals = pd.Series(index=combined.index, data=0, dtype=float)
        final_signals[combined > 0.5] = 1.0
        final_signals[combined < -0.5] = -1.0

        return final_signals

    def calculate_spread_zscore(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate z-score of spread based on rolling window.

        Args:
            spread (pd.Series): Spread series to calculate z-score for
            window (int): Rolling window size for mean/std calculation

        Returns:
            pd.Series: Z-score series
        """
        if len(spread) < window:
            return pd.Series(0, index=spread.index)

        try:
            rolling_mean = spread.rolling(window=window).mean()
            rolling_std = spread.rolling(window=window).std()

            zscore = (spread - rolling_mean) / (rolling_std + 1e-8)
            zscore = zscore.replace([np.inf, -np.inf], 0)

            return zscore.fillna(0)

        except Exception as e:
            logger.error(f"Error calculating zscore: {str(e)}")
            return pd.Series(0, index=spread.index)

    def calculate_hedge_ratio(self, asset1: pd.Series, asset2: pd.Series,
                              window: int = 63) -> float:
        """
        Calculate hedge ratio using linear regression with aligned indices

        Args:
            asset1: Price series of first asset
            asset2: Price series of second asset
            window: Rolling window size

        Returns:
            float: Calculated hedge ratio
        """
        if len(asset1) < window or len(asset2) < window:
            return 1.0

        try:
            if not isinstance(asset1.index, pd.DatetimeIndex):
                asset1 = pd.Series(asset1.values, index=pd.date_range(end=pd.Timestamp.now(),
                                                                      periods=len(asset1), freq='D'))
            if not isinstance(asset2.index, pd.DatetimeIndex):
                asset2 = pd.Series(asset2.values, index=pd.date_range(end=pd.Timestamp.now(),
                                                                      periods=len(asset2), freq='D'))

            df = pd.DataFrame({'asset1': asset1, 'asset2': asset2})
            df = df.dropna()

            if len(df) < window:
                return 1.0

            y = df['asset1'].iloc[-window:]
            X = add_constant(df['asset2'].iloc[-window:])

            model = OLS(y, X).fit()
            hedge_ratio = model.params[1]

            if not np.isfinite(hedge_ratio) or abs(hedge_ratio) > 10:
                logger.warning(f"Invalid hedge ratio {hedge_ratio}, using 1.0")
                return 1.0

            return hedge_ratio

        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {str(e)}")
            return 1.0

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.

        Args:
            spread (pd.Series): Spread series

        Returns:
            float: Half-life in periods
        """
        try:
            lag_spread = spread.shift(1)
            delta_spread = spread - lag_spread
            lag_spread = lag_spread[1:]
            delta_spread = delta_spread[1:]

            X = add_constant(lag_spread)
            model = OLS(delta_spread, X).fit()
            gamma = model.params[1]

            half_life = -np.log(2) / gamma if gamma < 0 else np.inf

            if not np.isfinite(half_life) or half_life < 0:
                logger.warning(f"Invalid half-life {half_life}, using inf")
                return np.inf

            return half_life

        except Exception as e:
            logger.error(f"Error calculating half-life: {str(e)}")
            return np.inf

    def calculate_cointegration_score(self, asset1: pd.Series, asset2: pd.Series,
                                      window: int = 252) -> Tuple[float, float]:
        """
        Calculate cointegration test statistics using rolling window.

        Args:
            asset1 (pd.Series): First asset prices
            asset2 (pd.Series): Second asset prices
            window (int): Rolling window size

        Returns:
            Tuple[float, float]: Cointegration test statistic and p-value
        """
        try:
            if len(asset1) < window or len(asset2) < window:
                return 0.0, 1.0

            y = asset1[-window:]
            x = asset2[-window:]

            hedge_ratio = self.calculate_hedge_ratio(y, x)
            spread = y - hedge_ratio * x

            adf_stat, pvalue, _ = adfuller(spread, maxlag=int(np.sqrt(len(spread))))

            return float(adf_stat), float(pvalue)

        except Exception as e:
            logger.error(f"Error in cointegration test: {str(e)}")
            return 0.0, 1.0

    def generate_enhanced_signals(self, spread: pd.Series,
                                  asset1: pd.DataFrame,
                                  asset2: pd.DataFrame) -> float:
        """
        Generate trading signals incorporating OHLCV-based features.
        """
        zscore = self.calculate_spread_zscore(spread)

        vol_ratio = (
                            asset1['Volume'] / asset1['Volume'].rolling(20).mean()
                    ) / (
                            asset2['Volume'] / asset2['Volume'].rolling(20).mean()
                    )

        range_ratio = (
                              (asset1['High'] - asset1['Low']) / asset1['Open']
                      ) / (
                              (asset2['High'] - asset2['Low']) / asset2['Open']
                      )

        signal = zscore * np.sign(1 - range_ratio)

        signal *= np.minimum(1, vol_ratio)

        return signal


def analyze_strategy_performance(positions):
    """
    Calculate key performance metrics for the pairs trading strategy.

    Args:
        positions: DataFrame containing strategy positions and returns

    Returns:
        dict containing performance metrics
    """
    returns = positions['strategy_returns']

    total_return = positions['cumulative_returns'].iloc[-1]
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1

    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * returns.mean() / daily_vol

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    max_drawdown = drawdowns.min()

    avg_exposure = positions['total_exposure'].mean()
    max_exposure = positions['total_exposure'].max()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_exposure': avg_exposure,
        'max_exposure': max_exposure
    }

def main():
    """Test the StatisticalModel with AAPL and MSFT data."""
    output_dir = "models_data/statistical_model_v2/"
    os.makedirs(output_dir, exist_ok=True)

    stock1 = pd.read_csv(r"C:\Users\arnav\Downloads\pairs_trading_system\data\raw\AAPL.csv")
    stock2 = pd.read_csv(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw\MSFT.csv')

    stock1['Date'] = pd.to_datetime(stock1['Date'])
    stock2['Date'] = pd.to_datetime(stock2['Date'])
    stock1.set_index('Date', inplace=True)
    stock2.set_index('Date', inplace=True)

    model = StatisticalModel()

    print("\nTesting cointegration...")
    is_cointegrated = model.cointegration_test(
        stock1['Close'],
        stock2['Close']
    )
    print(f"Series are cointegrated: {is_cointegrated}")

    print("\nTesting Johansen cointegration...")
    df_prices = pd.DataFrame({
        'AAPL': stock1['Close'],
        'MSFT': stock2['Close']
    })
    has_johansen_coint = model.johansen_test(df_prices)
    print(f"Johansen shows cointegration: {has_johansen_coint}")

    print("\nCalculating spreads...")
    ols_spread = model.calculate_spread(
        stock1['Close'],
        stock2['Close'],
        use_ratio=False
    )
    ratio_spread = model.calculate_spread(
        stock1['Close'],
        stock2['Close'],
        use_ratio=True
    )

    print("\nGenerating trading signals using multiple methods...")
    signals_ols = model.mean_reversion_signal(ols_spread)
    signals_ma = model.moving_average_crossover_signal(ols_spread)
    signals_bb = model.bollinger_band_signal(ols_spread)
    signals_rsi = model.rsi_signal(ols_spread)

    combined_signals = model.combine_signals([
        signals_ols,
        signals_ma,
        signals_bb,
        signals_rsi
    ])

    results = pd.DataFrame({
        'AAPL': stock1['Close'],
        'MSFT': stock2['Close'],
        'OLS_Spread': ols_spread,
        'Ratio_Spread': ratio_spread,
        'Mean_Reversion_Signals': signals_ols,
        'MA_Crossover_Signals': signals_ma,
        'Bollinger_Signals': signals_bb,
        'RSI_Signals': signals_rsi,
        'Combined_Signals': combined_signals
    })

    results.to_csv(os.path.join(output_dir, 'pair_trading_results.csv'))

    def calculate_pair_returns(asset1_prices, asset2_prices, signals, hedge_ratio=1.0):
        """
        Calculate returns for a market-neutral pairs trading strategy.
        Takes simultaneous long/short positions in both assets.

        Args:
            asset1_prices: Series of prices for first asset
            asset2_prices: Series of prices for second asset
            signals: Series of trading signals (-1, 0, or 1)
            hedge_ratio: Ratio for hedging positions (default 1.0)

        Returns:
            DataFrame containing:
            - Individual position returns
            - Combined strategy returns
            - Position sizes
        """
        asset1_returns = asset1_prices.pct_change()
        asset2_returns = asset2_prices.pct_change()

        positions = pd.DataFrame(index=signals.index)

        positions['asset1_pos'] = signals.shift(1)
        positions['asset2_pos'] = -signals.shift(1) * hedge_ratio

        positions['asset1_returns'] = positions['asset1_pos'] * asset1_returns
        positions['asset2_returns'] = positions['asset2_pos'] * asset2_returns

        positions['strategy_returns'] = positions['asset1_returns'] + positions['asset2_returns']

        positions['cumulative_returns'] = positions['strategy_returns'].cumsum()

        initial_capital = 100000
        positions['asset1_value'] = initial_capital * positions['asset1_pos']
        positions['asset2_value'] = initial_capital * positions['asset2_pos']

        positions['total_exposure'] = abs(positions['asset1_value']) + abs(positions['asset2_value'])
        positions['net_exposure'] = positions['asset1_value'] + positions['asset2_value']

        return positions

    strategy_returns = pd.DataFrame({
        'Mean_Reversion': calculate_pair_returns(stock1['Close'], stock2['Close'], signals_ols)['strategy_returns'],
        'MA_Crossover': calculate_pair_returns(stock1['Close'], stock2['Close'], signals_ma)['strategy_returns'],
        'Bollinger': calculate_pair_returns(stock1['Close'], stock2['Close'], signals_bb)['strategy_returns'],
        'RSI': calculate_pair_returns(stock1['Close'], stock2['Close'], signals_rsi)['strategy_returns'],
        'Combined': calculate_pair_returns(stock1['Close'], stock2['Close'], combined_signals)['strategy_returns']
    })

    with open(os.path.join(output_dir, 'performance.txt'), 'w') as f:
        f.write("Strategy Performance:\n\n")
        for strategy in strategy_returns.columns:
            final_return = strategy_returns[strategy].iloc[-1]
            sharpe = np.sqrt(252) * (strategy_returns[strategy].diff().mean() /
                                   strategy_returns[strategy].diff().std())
            f.write(f"{strategy} Strategy:\n")
            f.write(f"Final Return: {final_return:.4f}\n")
            f.write(f"Sharpe Ratio: {sharpe:.4f}\n\n")

    fig1 = make_subplots(rows=2, cols=1, subplot_titles=("Asset Prices", "Combined Signals"))

    fig1.add_trace(go.Scatter(x=results.index, y=results['AAPL'], name='AAPL'), row=1, col=1)
    fig1.add_trace(go.Scatter(x=results.index, y=results['MSFT'], name='MSFT'), row=1, col=1)
    fig1.add_trace(go.Scatter(x=results.index, y=combined_signals, name='Combined Signals'), row=2, col=1)

    long_positions = combined_signals == 1
    short_positions = combined_signals == -1

    fig1.add_trace(go.Scatter(
        x=results.index[long_positions],
        y=results.loc[long_positions, 'AAPL'],
        mode='markers',
        name='Long AAPL',
        marker=dict(symbol='triangle-up', size=10, color='green')
    ), row=1, col=1)

    fig1.add_trace(go.Scatter(
        x=results.index[short_positions],
        y=results.loc[short_positions, 'AAPL'],
        mode='markers',
        name='Short AAPL',
        marker=dict(symbol='triangle-down', size=10, color='red')
    ), row=1, col=1)

    fig1.update_layout(height=800, title_text="Pairs Trading Analysis")
    fig1.write_html(os.path.join(output_dir, 'trading_signals.html'))

    fig2 = make_subplots(rows=2, cols=1, subplot_titles=("OLS Spread", "Signals Comparison"))

    fig2.add_trace(go.Scatter(x=results.index, y=ols_spread, name='OLS Spread'), row=1, col=1)

    roll_mean = ols_spread.rolling(window=20).mean()
    roll_std = ols_spread.rolling(window=20).std()
    upper_band = roll_mean + (2 * roll_std)
    lower_band = roll_mean - (2 * roll_std)

    fig2.add_trace(go.Scatter(x=results.index, y=upper_band, name='Upper BB',
                             line=dict(dash='dash')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=results.index, y=lower_band, name='Lower BB',
                             line=dict(dash='dash')), row=1, col=1)

    fig2.add_trace(go.Scatter(x=results.index, y=signals_ols, name='Mean Reversion'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=results.index, y=signals_ma, name='MA Crossover'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=results.index, y=signals_bb, name='Bollinger'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=results.index, y=signals_rsi, name='RSI'), row=2, col=1)

    fig2.update_layout(height=800, title_text="Spread Analysis and Signal Comparison")
    fig2.write_html(os.path.join(output_dir, 'spread_analysis.html'))

    fig3 = go.Figure()
    for strategy in strategy_returns.columns:
        fig3.add_trace(go.Scatter(x=results.index,
                                 y=strategy_returns[strategy],
                                 name=strategy))

    fig3.update_layout(title="Cumulative Strategy Returns",
                      xaxis_title="Date",
                      yaxis_title="Cumulative Return",
                      height=600)
    fig3.write_html(os.path.join(output_dir, 'strategy_returns.html'))

    print(f"\nResults and visualizations saved to {output_dir}")

    strategy_positions = {}
    for signal_type in ['Mean_Reversion', 'MA_Crossover', 'Bollinger', 'RSI', 'Combined']:
        signals = results[f'{signal_type}_Signals'] if signal_type != 'Combined' else combined_signals
        strategy_positions[signal_type] = calculate_pair_returns(
            stock1['Close'],
            stock2['Close'],
            signals
        )

    with open(os.path.join(output_dir, 'enhanced_performance.txt'), 'w') as f:
        f.write("Enhanced Strategy Performance:\n\n")
        for strategy, positions in strategy_positions.items():
            metrics = analyze_strategy_performance(positions)
            f.write(f"{strategy} Strategy:\n")
            f.write(f"Total Return: {metrics['total_return']:.4f}\n")
            f.write(f"Annual Return: {metrics['annual_return']:.4f}\n")
            f.write(f"Annual Volatility: {metrics['annual_volatility']:.4f}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.4f}\n")
            f.write(f"Average Exposure: ${metrics['avg_exposure']:,.2f}\n")
            f.write(f"Maximum Exposure: ${metrics['max_exposure']:,.2f}\n\n")

    fig4 = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Asset Positions",
            "Strategy Returns",
            "Position Exposures"
        )
    )

    combined_pos = strategy_positions['Combined']

    fig4.add_trace(
        go.Scatter(
            x=combined_pos.index,
            y=combined_pos['asset1_value'],
            name='AAPL Position'
        ),
        row=1,
        col=1
    )
    fig4.add_trace(
        go.Scatter(
            x=combined_pos.index,
            y=combined_pos['asset2_value'],
            name='MSFT Position'
        ),
        row=1,
        col=1
    )

    fig4.add_trace(
        go.Scatter(
            x=combined_pos.index,
            y=combined_pos['cumulative_returns'],
            name='Strategy Returns'
        ),
        row=2,
        col=1
    )

    fig4.add_trace(
        go.Scatter(
            x=combined_pos.index,
            y=combined_pos['total_exposure'],
            name='Total Exposure'
        ),
        row=3,
        col=1
    )
    fig4.add_trace(
        go.Scatter(
            x=combined_pos.index,
            y=combined_pos['net_exposure'],
            name='Net Exposure'
        ),
        row=3,
        col=1
    )

    fig4.update_layout(
        height=1200,
        title_text="Enhanced Pairs Trading Analysis"
    )
    fig4.write_html(os.path.join(output_dir, 'enhanced_trading_analysis.html'))

    return results, model, strategy_positions

if __name__ == "__main__":
    results, model, strategy_returns = main()
