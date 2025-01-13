"""
Cointegration Analysis Module

This module provides functionality for:
1. Loading and preprocessing stock price data
2. Testing for cointegration across multiple time periods
3. Visualizing cointegrated pairs
4. Generating detailed analysis reports
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from config.logging_config import logger
import os
from glob import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Configuration
LOOKBACK_PERIODS = {
    '7D': 7,  # ~1 week
    '1M': 21,  # ~1 month
    '3M': 63,  # ~3 months
    '6M': 126,  # ~6 months
    '12M': 252  # ~12 months
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


def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion for a spread series.

    Args:
        spread (pd.Series): Price spread series

    Returns:
        float: Half-life in periods
    """
    lag_spread = spread.shift(1)
    delta_spread = spread - lag_spread
    lag_spread = lag_spread[1:]
    delta_spread = delta_spread[1:]

    # Regression: delta_spread = alpha + beta * lag_spread + epsilon
    beta = np.polyfit(lag_spread, delta_spread, 1)[0]
    half_life = -np.log(2) / beta if beta < 0 else np.nan
    return half_life


def find_cointegrated_pairs(prices: pd.DataFrame, lookback_period: int, significance_level: float = 0.05) -> list:
    """
    Find all pairs of assets that are cointegrated.

    Args:
        prices (pd.DataFrame): Price data for all assets
        lookback_period (int): Number of days to look back
        significance_level (float): Significance level for cointegration test

    Returns:
        list: List of dictionaries containing pair information
    """
    logger.info(f"Searching for cointegrated pairs with {lookback_period} days lookback.")

    # Use only the most recent lookback_period days
    recent_prices = prices.iloc[-lookback_period:]

    n = recent_prices.shape[1]
    cointegrated_pairs = []

    total_pairs = (n * (n - 1)) // 2
    completed_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            s1 = recent_prices.iloc[:, i]
            s2 = recent_prices.iloc[:, j]

            # Calculate spread and half-life
            spread = s1 - s2 * np.polyfit(s2, s1, 1)[0]
            half_life = calculate_half_life(spread)

            score, p_value, critical_values = coint(s1, s2)

            if p_value < significance_level:
                pair_info = {
                    'stock1': recent_prices.columns[i],
                    'stock2': recent_prices.columns[j],
                    'p_value': p_value,
                    'score': score,
                    'critical_values': critical_values,
                    'half_life': half_life
                }
                cointegrated_pairs.append(pair_info)

            completed_pairs += 1
            if completed_pairs % 100 == 0:
                logger.info(f"Progress: {completed_pairs}/{total_pairs} pairs tested")

    cointegrated_pairs = sorted(cointegrated_pairs, key=lambda x: x['p_value'])
    logger.info(f"Total cointegrated pairs found: {len(cointegrated_pairs)}")
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
    # Get the price series
    recent_prices = prices.iloc[-lookback_period:]
    s1 = recent_prices[stock1]
    s2 = recent_prices[stock2]

    # Calculate metrics
    normalized_s1 = (s1 - s1.mean()) / s1.std()
    normalized_s2 = (s2 - s2.mean()) / s2.std()
    ratio = s1 / s2

    # Calculate spread and half-life
    beta = np.polyfit(s2, s1, 1)[0]
    spread = s1 - beta * s2
    half_life = calculate_half_life(spread)

    # Calculate rolling metrics
    rolling_window = min(60, lookback_period // 4)
    rolling_corr = s1.rolling(window=rolling_window).corr(s2)

    # Create subplots
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

    # Add traces
    fig.add_trace(go.Scatter(x=s1.index, y=s1, name=stock1), row=1, col=1)
    fig.add_trace(go.Scatter(x=s2.index, y=s2, name=stock2), row=1, col=1)

    fig.add_trace(go.Scatter(x=s1.index, y=normalized_s1, name=f'{stock1} (norm)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=s2.index, y=normalized_s2, name=f'{stock2} (norm)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread'), row=3, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name='Price Ratio'), row=4, col=1)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='Rolling Corr'), row=5, col=1)

    # Update layout
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

    # Get CSV files
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

    # Create detailed comparison
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


if __name__ == "__main__":
    # Load price data
    logger.info("Loading price data...")
    prices_df = load_nasdaq100_data()
    logger.info(f"Loaded price data for {len(prices_df.columns)} stocks")

    # Create timestamp for output directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"cointegration_test"
    os.makedirs(base_output_dir, exist_ok=True)

    # Store results for all periods
    all_pairs_results = {}

    # Process each lookback period
    for period_name, lookback_days in LOOKBACK_PERIODS.items():
        logger.info(f"Processing {period_name} period ({lookback_days} days)")

        # Create period-specific directory
        period_dir = os.path.join(base_output_dir, f"period_{period_name}")
        os.makedirs(period_dir, exist_ok=True)

        # Find cointegrated pairs for this period
        cointegrated_pairs = find_cointegrated_pairs(prices_df, lookback_days)
        all_pairs_results[period_name] = cointegrated_pairs

        # Save results to CSV
        results_df = pd.DataFrame(cointegrated_pairs)
        results_df.to_csv(os.path.join(period_dir, 'cointegrated_pairs.csv'), index=False)

        # Create plots directory for this period
        plots_dir = os.path.join(period_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot top 10 pairs for this period
        logger.info(f"Plotting top 10 pairs for {period_name} period...")
        for i, pair in enumerate(cointegrated_pairs[:10]):
            stock1, stock2 = pair['stock1'], pair['stock2']
            fig = plot_cointegrated_pair(prices_df, stock1, stock2, lookback_days)
            plot_filename = f'pair_{i + 1}_{stock1}_{stock2}_analysis.html'
            fig.write_html(os.path.join(plots_dir, plot_filename))

        # Generate period-specific summary
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

    # Create overall summary comparing different periods
    create_summary_report(all_pairs_results, base_output_dir)
    logger.info("Analysis complete. Results saved in: " + base_output_dir)