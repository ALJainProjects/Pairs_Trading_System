"""
Cointegration Analysis Module

This module provides functionality for:
1. Loading and preprocessing stock price data
2. Testing for cointegration across multiple time periods
3. Visualizing cointegrated pairs
4. Generating detailed analysis reports
"""
from typing import Optional

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from config.logging_config import logger
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LOOKBACK_PERIODS = {
    '7D': 7,
    '1M': 21,
    '3M': 63,
    '6M': 126,
    '12M': 252,
    '24M': 504
}


def determine_cointegration(series1: pd.Series, series2: pd.Series, significance_level: float = 0.05) -> tuple:
    """
    Test for cointegration between two series.

    Args:
        series1 (pd.Series): First price series
        series2 (pd.Series): Second price series
        significance_level (float): Significance level for cointegration test

    Returns:
        tuple: (is_cointegrated, score, p_value, critical_values)
    """
    logger.info("Testing cointegration between two series.")
    score, p_value, critical_values = coint(series1, series2)
    logger.debug(f"Cointegration test p-value: {p_value:.4f}")
    return p_value < significance_level, score, p_value, critical_values


def check_integration_order(series: pd.Series) -> bool:
    """
    Check if a series is I(1) by testing:
    1. Non-stationary in levels
    2. Stationary in first differences
    """
    adf_level = adfuller(series, regression='c')[1]
    if adf_level < 0.05:
        return False

    adf_diff = adfuller(series.diff().dropna(), regression='c')[1]
    return adf_diff < 0.05

def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion for a spread series with stationarity check.
    """
    adf_result = adfuller(spread)
    if adf_result[1] > 0.05:
        return np.nan

    lag_spread = spread.shift(1)
    delta_spread = spread - lag_spread
    lag_spread = lag_spread[1:]
    delta_spread = delta_spread[1:]

    beta = np.polyfit(lag_spread, delta_spread, 1)[0]
    half_life = -np.log(2) / beta if beta < 0 else np.nan
    return half_life

def find_cointegrated_pairs(prices: pd.DataFrame,
                            lookback_period: int,
                            significance_level: float = 0.05,
                            max_pairs: Optional[int] = None,
                            min_half_life: int = 5,
                            max_half_life: int = 126) -> list:
    """
    Find cointegrated pairs with improved statistical checks and efficiency.

    New Parameters:
        min_correlation (float): Minimum correlation threshold for pairs
    """
    recent_prices = prices.iloc[-lookback_period:]
    n = recent_prices.shape[1]
    cointegrated_pairs = []
    total_pairs = (n * (n - 1)) // 2
    completed_pairs = 0
    failed_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            s1 = recent_prices.iloc[:, i]
            s2 = recent_prices.iloc[:, j]
            s1 = s1.ffill().bfill()
            s2 = s2.ffill().bfill()

            if not (check_integration_order(s1) and check_integration_order(s2)):
                completed_pairs += 1
                failed_pairs += 1
                continue

            score, p_value, critical_values = coint(s1, s2)
            if p_value >= significance_level:
                completed_pairs += 1
                failed_pairs += 1
                continue

            beta, alpha = np.polyfit(s2, s1, 1)
            spread = s1 - (alpha + beta * s2)

            half_life = calculate_half_life(spread)
            if np.isnan(half_life) or half_life < min_half_life or half_life > max_half_life:
                completed_pairs += 1
                failed_pairs += 1
                continue

            pair_info = {
                'stock1': recent_prices.columns[i],
                'stock2': recent_prices.columns[j],
                'p_value': p_value,
                'score': score,
                'critical_values': critical_values,
                'half_life': half_life,
                'beta': beta,
                'alpha': alpha
            }
            cointegrated_pairs.append(pair_info)

            completed_pairs += 1
            if completed_pairs % 100 == 0:
                logger.info(f"Progress: {completed_pairs}/{total_pairs} pairs tested")
                logger.info(f"Failed Pairs: {failed_pairs}/{total_pairs} of pairs tested")

    cointegrated_pairs = sorted(cointegrated_pairs, key=lambda x: x['p_value'])
    if max_pairs is not None and max_pairs < len(cointegrated_pairs):
        cointegrated_pairs = cointegrated_pairs[:max_pairs]

    return cointegrated_pairs

def plot_cointegrated_pair(prices: pd.DataFrame, stock1: str, stock2: str, lookback_period: int) -> go.Figure:
    """
    Create visualization of a cointegrated pair.

    Args:
        prices (pd.DataFrame): Price data for all assets
        stock1 (str): First stock ticker
        stock2 (str): Second stock ticker
        lookback_period (int): Number of days to look back

    Returns:
        go.Figure: Plotly figure object
    """
    recent_prices = prices.iloc[-lookback_period:]
    s1 = recent_prices[stock1]
    s2 = recent_prices[stock2]

    normalized_s1 = (s1 - s1.mean()) / s1.std()
    normalized_s2 = (s2 - s2.mean()) / s2.std()
    ratio = s1 / s2

    beta = np.polyfit(s2, s1, 1)[0]
    spread = s1 - beta * s2
    half_life = calculate_half_life(spread)

    rolling_window = min(60, lookback_period // 4)
    rolling_corr = s1.rolling(window=rolling_window).corr(s2)

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=(
            'Raw Prices',
            'Normalized Prices',
            'Price Spread',
            'Price Ratio',
            f'{rolling_window}-day Rolling Correlation'
        ),
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    fig.add_trace(go.Scatter(x=s1.index, y=s1, name=stock1), row=1, col=1)
    fig.add_trace(go.Scatter(x=s2.index, y=s2, name=stock2), row=1, col=1)

    fig.add_trace(go.Scatter(x=s1.index, y=normalized_s1, name=f'{stock1} (norm)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=s2.index, y=normalized_s2, name=f'{stock2} (norm)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread'), row=3, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name='Price Ratio'), row=4, col=1)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='Rolling Corr'), row=5, col=1)

    fig.update_layout(
        height=1500,
        width=1000,
        showlegend=True,
        title=f'Cointegration Analysis: {stock1} vs {stock2}\n' +
              f'Lookback: {lookback_period} days, Half-life: {half_life:.1f} days'
    )

    return fig


def load_nasdaq100_data(
        data_dir: str = r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw') -> pd.DataFrame:
    """
    Load price data for NASDAQ 100 stocks from CSV files.

    Args:
        data_dir (str): Directory containing the CSV files

    Returns:
        pd.DataFrame: DataFrame with dates as index and tickers as columns
    """
    logger.info(f"Loading price data from {data_dir}")

    csv_files = [f for f in os.listdir(data_dir)
                 if f.endswith('.csv') and f != 'combined_prices.csv']

    prices_dict = {}
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    for file in csv_files:
        try:
            ticker = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, file))

            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Skipping {file}: Missing required columns")
                continue

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            prices_dict[ticker] = pd.to_numeric(df['Close'], errors='coerce')
            logger.info(f"Loaded {ticker} data: {len(df)} rows")

        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")

    if not prices_dict:
        raise ValueError("No valid price data loaded")

    prices_df = pd.DataFrame(prices_dict)
    prices_df = prices_df.ffill().bfill()
    prices_df = prices_df.dropna(axis=1)

    logger.info(f"Loaded {len(prices_df.columns)} stocks with {len(prices_df)} data points each")
    return prices_df


def create_summary_report(all_pairs_results: dict, output_dir: str):
    """
    Create a summary report comparing results across different periods.

    Args:
        all_pairs_results (dict): Dictionary of results for each period
        output_dir (str): Directory to save the report
    """
    summary_data = []

    for period, pairs in all_pairs_results.items():
        period_summary = {
            'period': period,
            'total_pairs': len(pairs),
            'avg_p_value': np.mean([p['p_value'] for p in pairs]) if pairs else np.nan,
            'avg_half_life': np.mean([p['half_life'] for p in pairs]) if pairs else np.nan,
            'unique_stocks': len(set([p['stock1'] for p in pairs] + [p['stock2'] for p in pairs])) if pairs else 0
        }
        summary_data.append(period_summary)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'period_comparison_summary.csv'), index=False)

    with open(os.path.join(output_dir, 'period_comparison_details.txt'), 'w') as f:
        f.write("Comparison of Top Pairs Across Different Periods\n")
        f.write("============================================\n\n")

        for period, pairs in all_pairs_results.items():
            f.write(f"\nTop 10 Pairs for {period} Period:\n")
            f.write("-" * 40 + "\n")
            for pair in pairs[:10]:
                f.write(f"{pair['stock1']} - {pair['stock2']}: ")
                f.write(f"p-value = {pair['p_value']:.4f}, ")
                f.write(f"half-life = {pair['half_life']:.1f} days\n")

def dynamic_cointegration_with_proportion(prices: pd.DataFrame, stock1: str, stock2: str,
                                          window_size: int, significance_level: float = 0.05) -> tuple:
    """
    Perform rolling-window cointegration tests and calculate the proportion of significant p-values.

    Args:
        prices (pd.DataFrame): Price data for all assets.
        stock1 (str): First stock ticker.
        stock2 (str): Second stock ticker.
        window_size (int): Size of the rolling window.
        significance_level (float): Significance level for cointegration test.

    Returns:
        tuple: (results_df, proportion_significant)
            - results_df: DataFrame with p-values, scores, and half-lives for each window.
            - proportion_significant: Proportion of windows with p-value < significance_level.
    """
    results = []
    significant_count = 0
    total_windows = len(prices) - window_size

    for start in range(total_windows):
        end = start + window_size
        window_prices = prices.iloc[start:end]

        s1 = window_prices[stock1]
        s2 = window_prices[stock2]

        # Perform cointegration test
        try:
            score, p_value, critical_values = coint(s1, s2)
            spread = s1 - s2 * np.polyfit(s2, s1, 1)[0]
            half_life = calculate_half_life(spread)
        except Exception as e:
            logger.warning(f"Cointegration failed for window {start}-{end}: {e}")
            p_value, score, half_life = np.nan, np.nan, np.nan

        if not np.isnan(p_value) and p_value < significance_level:
            significant_count += 1

        results.append({
            'start_date': window_prices.index[0],
            'end_date': window_prices.index[-1],
            'p_value': p_value,
            'score': score,
            'half_life': half_life
        })

    results_df = pd.DataFrame(results)
    proportion_significant = significant_count / total_windows if total_windows > 0 else 0
    return results_df, proportion_significant


if __name__ == "__main__":
    logger.info("Loading price data...")
    prices_df = load_nasdaq100_data()
    logger.info(f"Loaded price data for {len(prices_df.columns)} stocks")

    base_output_dir = f"cointegration_test"
    os.makedirs(base_output_dir, exist_ok=True)

    all_pairs_results = {}

    for period_name, lookback_days in LOOKBACK_PERIODS.items():
        logger.info(f"Processing {period_name} period ({lookback_days} days)")

        period_dir = os.path.join(base_output_dir, f"period_{period_name}")
        os.makedirs(period_dir, exist_ok=True)

        cointegrated_pairs = find_cointegrated_pairs(prices_df, lookback_days)
        all_pairs_results[period_name] = cointegrated_pairs

        results_df = pd.DataFrame(cointegrated_pairs)
        results_df.to_csv(os.path.join(period_dir, 'cointegrated_pairs.csv'), index=False)

        plots_dir = os.path.join(period_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        logger.info(f"Plotting top 10 pairs for {period_name} period...")
        for i, pair in enumerate(cointegrated_pairs[:10]):
            stock1, stock2 = pair['stock1'], pair['stock2']
            fig = plot_cointegrated_pair(prices_df, stock1, stock2, lookback_days)
            plot_filename = f'pair_{i + 1}_{stock1}_{stock2}_analysis.html'
            fig.write_html(os.path.join(plots_dir, plot_filename))

        print(f"\nPeriod: {period_name} ({lookback_days} days)")
        print("-" * 40)
        print(f"Total pairs found: {len(cointegrated_pairs)}")
        if cointegrated_pairs:
            print(f"Average p-value: {np.mean([p['p_value'] for p in cointegrated_pairs]):.4f}")
            print(f"Average half-life: {np.mean([p['half_life'] for p in cointegrated_pairs]):.1f} days")
            print("\nTop 10 Most Significant Pairs:")
            for pair in cointegrated_pairs[:10]:
                print(f"{pair['stock1']} - {pair['stock2']}: ")
                print(f"    p-value = {pair['p_value']:.4f}, half-life = {pair['half_life']:.1f} days")

    create_summary_report(all_pairs_results, base_output_dir)
    logger.info("Analysis complete. Results saved in: " + base_output_dir)

