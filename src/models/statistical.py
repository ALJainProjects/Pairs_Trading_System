"""
Statistical Models Module

Enhancements:
 1. More robust cointegration testing using Engle-Granger and optional Johansen.
 2. Spread calculation with optional ratio-based spread.
 3. Extended docstrings for usage in a pair trading workflow.
"""
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from config.logging_config import logger
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import MODEL_DIR


class StatisticalModel:
    """
    The StatisticalModel provides core statistical methods relevant to pair selection and
    signal generation in pair trading. Methods included:
      - Engle-Granger cointegration test
      - Optional Johansen cointegration test (if installed)
      - Calculating a spread between two assets via regression
      - Mean reversion signal generation
    """

    def __init__(self):
        logger.info("Initializing StatisticalModel.")

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


def main():
    """Test the StatisticalModel with AAPL and MSFT data."""
    output_dir = f"{MODEL_DIR}/statistical_model"
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

    print("\nGenerating trading signals...")
    signals_ols = model.mean_reversion_signal(
        ols_spread,
        window=20,
        z_threshold=2.0
    )
    signals_ratio = model.mean_reversion_signal(
        ratio_spread,
        window=20,
        z_threshold=2.0
    )

    results = pd.DataFrame({
        'AAPL': stock1['Close'],
        'MSFT': stock2['Close'],
        'OLS_Spread': ols_spread,
        'Ratio_Spread': ratio_spread,
        'OLS_Signals': signals_ols,
        'Ratio_Signals': signals_ratio
    })

    results.to_csv(os.path.join(output_dir, 'pair_trading_results.csv'))

    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write("Summary Statistics:\n\n")
        f.write("OLS Spread Statistics:\n")
        f.write(ols_spread.describe().to_string())
        f.write("\n\nRatio Spread Statistics:\n")
        f.write(ratio_spread.describe().to_string())
        f.write("\n\nSignal Counts:\n")
        f.write("\nOLS Signals:\n")
        f.write(signals_ols.value_counts().to_string())
        f.write("\n\nRatio Signals:\n")
        f.write(signals_ratio.value_counts().to_string())

    def calculate_pair_returns(asset1_prices, asset2_prices, signals):
        """Calculate returns for a pairs trading strategy."""
        asset1_returns = asset1_prices.pct_change()
        asset2_returns = asset2_prices.pct_change()

        strategy_returns = signals.shift(1) * (asset1_returns - asset2_returns)
        return strategy_returns.cumsum()

    ols_performance = calculate_pair_returns(stock1['Close'], stock2['Close'], signals_ols)
    ratio_performance = calculate_pair_returns(stock1['Close'], stock2['Close'], signals_ratio)

    with open(os.path.join(output_dir, 'performance.txt'), 'w') as f:
        f.write("Strategy Performance:\n")
        f.write(f"OLS Strategy Final Return: {ols_performance.iloc[-1]:.4f}\n")
        f.write(f"Ratio Strategy Final Return: {ratio_performance.iloc[-1]:.4f}\n")

    fig1 = make_subplots(rows=1, cols=1)
    fig1.add_trace(go.Scatter(x=results.index, y=results['AAPL'], name='AAPL'))
    fig1.add_trace(go.Scatter(x=results.index, y=results['MSFT'], name='MSFT'))
    fig1.update_layout(title="Price Series", height=600)
    fig1.write_html(os.path.join(output_dir, 'price_series.html'))

    fig1_signals = make_subplots(rows=1, cols=1)

    fig1_signals.add_trace(go.Scatter(
        x=results.index,
        y=results['AAPL'],
        name='AAPL',
        line=dict(color='blue')
    ))

    fig1_signals.add_trace(go.Scatter(
        x=results.index,
        y=results['MSFT'],
        name='MSFT',
        line=dict(color='purple')
    ))

    long_aapl = signals_ols == 1
    short_aapl = signals_ols == -1

    fig1_signals.add_trace(go.Scatter(
        x=results.index[long_aapl],
        y=results.loc[long_aapl, 'AAPL'],
        mode='markers',
        name='Long AAPL',
        marker=dict(
            symbol='triangle-up',
            size=12,
            color='green',
            line=dict(width=1)
        )
    ))

    fig1_signals.add_trace(go.Scatter(
        x=results.index[short_aapl],
        y=results.loc[short_aapl, 'AAPL'],
        mode='markers',
        name='Short AAPL',
        marker=dict(
            symbol='triangle-down',
            size=12,
            color='red',
            line=dict(width=1)
        )
    ))

    fig1_signals.add_trace(go.Scatter(
        x=results.index[long_aapl],
        y=results.loc[long_aapl, 'MSFT'],
        mode='markers',
        name='Short MSFT',
        marker=dict(
            symbol='triangle-down',
            size=12,
            color='red',
            line=dict(width=1)
        )
    ))

    fig1_signals.add_trace(go.Scatter(
        x=results.index[short_aapl],
        y=results.loc[short_aapl, 'MSFT'],
        mode='markers',
        name='Long MSFT',
        marker=dict(
            symbol='triangle-up',
            size=12,
            color='green',
            line=dict(width=1)
        )
    ))

    fig1_signals.update_layout(
        title="Pairs Trading Signals",
        height=800,
        yaxis_title="Price",
        xaxis_title="Date",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig1_signals.write_html(os.path.join(output_dir, 'price_series_with_signals.html'))

    fig2 = make_subplots(rows=1, cols=1)
    fig2.add_trace(go.Scatter(x=results.index, y=results['OLS_Spread'], name='OLS Spread'))
    fig2.add_trace(go.Scatter(x=results.index, y=results['Ratio_Spread'], name='Ratio Spread'))
    fig2.update_layout(title="Spreads", height=600)
    fig2.write_html(os.path.join(output_dir, 'spreads.html'))

    fig3 = make_subplots(rows=1, cols=1)
    fig3.add_trace(go.Scatter(x=results.index, y=signals_ols, name='OLS Signals'))
    fig3.add_trace(go.Scatter(x=results.index, y=signals_ratio, name='Ratio Signals'))
    fig3.update_layout(title="Trading Signals", height=600)
    fig3.write_html(os.path.join(output_dir, 'signals.html'))

    fig4 = make_subplots(rows=1, cols=1)
    fig4.add_trace(go.Scatter(x=results.index, y=ols_performance, name='OLS Strategy'))
    fig4.add_trace(go.Scatter(x=results.index, y=ratio_performance, name='Ratio Strategy'))
    fig4.update_layout(title="Cumulative Returns", height=600)
    fig4.write_html(os.path.join(output_dir, 'performance.html'))

    print(f"\nResults saved to {output_dir}")
    return results, model

if __name__ == "__main__":
    results, model = main()
