import pandas as pd
import numpy as np
from config.logging_config import logger  # Assuming this is configured
from typing import Optional  # For optional benchmark returns


def calculate_all_metrics(equity_curve: pd.Series,
                          risk_free_rate: float = 0.02,
                          benchmark_returns: Optional[pd.Series] = None) -> dict:
    """
    Calculates a comprehensive set of performance metrics for an equity curve.

    Args:
        equity_curve (pd.Series): A time series of portfolio equity values.
        risk_free_rate (float): The annualized risk-free rate (e.g., 0.02 for 2%).
        benchmark_returns (Optional[pd.Series]): Daily percentage returns of the benchmark.
                                                 Required for Alpha and Beta calculation.

    Returns:
        dict: A dictionary containing various performance metrics.
    """
    if equity_curve.empty:
        logger.warning("Equity curve is empty. Cannot calculate metrics.")
        return {metric: 0.0 for metric in [
            'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Omega Ratio',
            'Beta', 'Alpha'
        ]}

    # Ensure equity curve starts at a positive value for return calculations
    if equity_curve.iloc[0] <= 0:
        logger.warning(
            "Equity curve starts at zero or negative value. Return calculations may be invalid. Returning 0.0 for returns."
        )
        return {metric: 0.0 for metric in [
            'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Omega Ratio',
            'Beta', 'Alpha'
        ]}

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.empty:
        logger.warning(
            "Daily returns series is empty after pct_change/dropna. Insufficient data. Returning 0.0 for metrics."
        )
        return {metric: 0.0 for metric in [
            'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Omega Ratio',
            'Beta', 'Alpha'
        ]}

    # Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualized Return
    # Ensure there's at least one day difference for calculation
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0.0

    # Annualized Volatility
    annualized_volatility = daily_returns.std() * np.sqrt(252)  # Assuming 252 trading days
    # Handle case where volatility is zero (e.g., flat returns)
    if annualized_volatility == 0:
        annualized_volatility = 1e-9  # Small non-zero value to avoid division by zero in ratios

    # Max Drawdown
    max_dd = calculate_max_drawdown(equity_curve)

    metrics_dict = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': calculate_sharpe_ratio(daily_returns, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(daily_returns, risk_free_rate),
        'Max Drawdown': max_dd,
        'Calmar Ratio': calculate_calmar_ratio(annualized_return, max_dd),
        'Omega Ratio': calculate_omega_ratio(daily_returns, risk_free_rate)
    }

    # Calculate Beta and Alpha only if benchmark returns are provided
    if benchmark_returns is not None:
        # Align portfolio and benchmark returns
        common_index = daily_returns.index.intersection(benchmark_returns.index)
        aligned_portfolio_returns = daily_returns[common_index]
        aligned_benchmark_returns = benchmark_returns[common_index]

        if not aligned_portfolio_returns.empty and not aligned_benchmark_returns.empty:
            beta = calculate_beta(aligned_portfolio_returns, aligned_benchmark_returns)
            alpha = calculate_alpha(aligned_portfolio_returns, aligned_benchmark_returns, risk_free_rate, beta)
            metrics_dict['Beta'] = beta
            metrics_dict['Alpha'] = alpha
        else:
            logger.warning("Could not align portfolio and benchmark returns. Alpha and Beta set to 0.0.")
            metrics_dict['Beta'] = 0.0
            metrics_dict['Alpha'] = 0.0
    else:
        metrics_dict['Beta'] = 0.0
        metrics_dict['Alpha'] = 0.0  # Default if no benchmark is provided

    return metrics_dict


def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the annualized Sharpe Ratio.
    """
    daily_risk_free_rate = risk_free_rate / 252  # Assuming 252 trading days
    excess_returns = daily_returns - daily_risk_free_rate

    std_dev = excess_returns.std()  # Standard deviation of excess returns
    if std_dev == 0: return 0.0

    sharpe = (excess_returns.mean() / std_dev) * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the annualized Sortino Ratio.
    """
    daily_risk_free_rate = risk_free_rate / 252

    # Only consider returns that are below the risk-free rate (downside deviation)
    downside_returns = daily_returns[daily_returns < daily_risk_free_rate]

    downside_std = downside_returns.std()
    if downside_std == 0:
        return np.inf if (daily_returns.mean() - daily_risk_free_rate) > 0 else 0.0

    sortino_ratio = (daily_returns.mean() - daily_risk_free_rate) / downside_std * np.sqrt(252)
    return sortino_ratio


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the maximum drawdown of an equity curve.
    Returns the maximum drawdown as a positive decimal (e.g., 0.25 for 25% drawdown).
    """
    if equity_curve.empty or equity_curve.iloc[0] <= 0:
        return 0.0

    # Calculate the running maximum equity
    running_max = equity_curve.expanding().max()
    # Calculate the drawdown from the running maximum
    drawdown = (equity_curve - running_max) / running_max

    # Max drawdown is the minimum (most negative) value in the drawdown series, take absolute
    max_dd = abs(drawdown.min())

    # Handle case where drawdown is NaN (e.g., constant equity curve where min is not found)
    return max_dd if pd.notna(max_dd) else 0.0


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calculates the Calmar Ratio.
    """
    if max_drawdown <= 1e-9:  # Handle very small or zero drawdown
        return np.inf  # If no drawdown or negligible, ratio is infinite for positive return
    return annualized_return / max_drawdown


def calculate_omega_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the Omega Ratio.
    """
    daily_risk_free_rate = risk_free_rate / 252
    returns_less_target = daily_returns - daily_risk_free_rate

    # Sum of positive excess returns (gains)
    gain = returns_less_target[returns_less_target > 0].sum()
    # Sum of absolute negative excess returns (losses)
    loss = abs(returns_less_target[returns_less_target < 0].sum())

    if loss <= 1e-9:  # Handle very small or zero loss
        return np.inf  # If no losses or negligible, ratio is infinite
    return gain / loss


def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculates the Beta of a portfolio relative to a benchmark.

    Args:
        portfolio_returns (pd.Series): Daily percentage returns of the portfolio.
        benchmark_returns (pd.Series): Daily percentage returns of the benchmark.

    Returns:
        float: The calculated Beta. Returns 0.0 if benchmark variance is zero or data is insufficient.
    """
    # Ensure returns are aligned (though calculate_all_metrics handles it too)
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    aligned_portfolio_returns = portfolio_returns[common_index]
    aligned_benchmark_returns = benchmark_returns[common_index]

    if aligned_benchmark_returns.empty or aligned_portfolio_returns.empty:
        logger.warning("Insufficient aligned data for Beta calculation. Returning 0.0.")
        return 0.0

    # Calculate covariance between portfolio and benchmark returns
    covariance = aligned_portfolio_returns.cov(aligned_benchmark_returns)
    # Calculate variance of benchmark returns
    benchmark_variance = aligned_benchmark_returns.var()

    if benchmark_variance == 0:
        logger.warning("Benchmark returns variance is zero. Cannot calculate Beta. Returning 0.0.")
        return 0.0

    beta = covariance / benchmark_variance
    return beta


def calculate_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                    risk_free_rate: float, beta: Optional[float] = None) -> float:
    """
    Calculates Jensen's Alpha.

    Args:
        portfolio_returns (pd.Series): Daily percentage returns of the portfolio.
        benchmark_returns (pd.Series): Daily percentage returns of the benchmark.
        risk_free_rate (float): Annualized risk-free rate.
        beta (Optional[float]): Pre-calculated Beta. If None, it will be calculated.

    Returns:
        float: The calculated Jensen's Alpha. Returns 0.0 if calculations are not possible.
    """
    # Align returns
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    aligned_portfolio_returns = portfolio_returns[common_index]
    aligned_benchmark_returns = benchmark_returns[common_index]

    if aligned_portfolio_returns.empty or aligned_benchmark_returns.empty:
        logger.warning("Insufficient aligned data for Alpha calculation. Returning 0.0.")
        return 0.0

    # If beta is not provided, calculate it
    if beta is None:
        beta = calculate_beta(aligned_portfolio_returns, aligned_benchmark_returns)
        if beta == 0.0:  # If beta calculation failed or was zero
            logger.warning("Beta could not be calculated or is zero, affecting Alpha. Returning 0.0 for Alpha.")
            return 0.0

    # Calculate average daily portfolio and benchmark returns
    avg_portfolio_return_daily = aligned_portfolio_returns.mean()
    avg_benchmark_return_daily = aligned_benchmark_returns.mean()

    # Annualize the daily risk-free rate
    daily_risk_free_rate = risk_free_rate / 252  # Assuming 252 trading days

    # Calculate expected portfolio return based on CAPM
    expected_portfolio_return_daily = daily_risk_free_rate + beta * (avg_benchmark_return_daily - daily_risk_free_rate)

    # Calculate Alpha (daily)
    alpha_daily = avg_portfolio_return_daily - expected_portfolio_return_daily

    # Annualize Alpha (important for comparability)
    # Annualized alpha is often approximated as daily alpha * 252, but strictly,
    # (1+daily_alpha)^252 - 1. For small alpha, multiplication is a good approximation.
    # Given other metrics are annualized by multiplication, let's keep it consistent.
    alpha_annualized = alpha_daily * 252  # Simpler approximation

    return alpha_annualized


# Example Usage (for testing the functions directly)
if __name__ == "__main__":
    from datetime import date, timedelta

    # Generate sample equity curve and benchmark returns
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')  # Business days
    np.random.seed(42)

    # Simulate equity curve with some growth and volatility
    initial_equity = 100000
    equity_returns = np.random.normal(0.0005, 0.005, len(dates))  # Mean 0.05% daily, 0.5% std dev
    equity_curve = initial_equity * (1 + equity_returns).cumprod()
    equity_curve = pd.Series(equity_curve, index=dates, name='Portfolio Equity')

    # Simulate benchmark returns (e.g., S&P 500)
    benchmark_daily_returns = np.random.normal(0.0003, 0.004, len(dates))
    # Introduce some correlation for beta calculation
    benchmark_daily_returns += equity_returns * 0.5  # Correlate with portfolio returns
    benchmark_returns_series = pd.Series(benchmark_daily_returns, index=dates, name='Benchmark Returns')

    risk_free = 0.02  # 2% annualized

    print("--- Individual Metric Calculations ---")

    daily_port_returns = equity_curve.pct_change().dropna()

    sharpe = calculate_sharpe_ratio(daily_port_returns, risk_free)
    print(f"Sharpe Ratio: {sharpe:.4f}")

    sortino = calculate_sortino_ratio(daily_port_returns, risk_free)
    print(f"Sortino Ratio: {sortino:.4f}")

    max_dd = calculate_max_drawdown(equity_curve)
    print(f"Max Drawdown: {max_dd:.4f} ({max_dd:.2%})")

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days_for_annual = (equity_curve.index[-1] - equity_curve.index[0]).days
    annual_ret = (1 + total_ret) ** (365.25 / days_for_annual) - 1 if days_for_annual > 0 else 0.0

    calmar = calculate_calmar_ratio(annual_ret, max_dd)
    print(f"Calmar Ratio: {calmar:.4f}")

    omega = calculate_omega_ratio(daily_port_returns, risk_free)
    print(f"Omega Ratio: {omega:.4f}")

    beta_val = calculate_beta(daily_port_returns, benchmark_returns_series)
    print(f"Beta: {beta_val:.4f}")

    alpha_val = calculate_alpha(daily_port_returns, benchmark_returns_series, risk_free, beta_val)
    print(f"Alpha: {alpha_val:.4f}")

    print("\n--- Comprehensive Metrics (calculate_all_metrics) ---")
    all_metrics = calculate_all_metrics(equity_curve, risk_free, benchmark_returns=benchmark_returns_series)
    for metric, value in all_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Test edge cases
    print("\n--- Edge Case: Empty Equity Curve ---")
    empty_equity = pd.Series(dtype=float)
    print(calculate_all_metrics(empty_equity))

    print("\n--- Edge Case: Flat Equity Curve ---")
    flat_equity = pd.Series([100, 100, 100, 100, 100], index=pd.to_datetime(
        ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))
    flat_benchmark = pd.Series([0.0001] * 4, index=flat_equity.index[1:])  # Tiny non-zero for benchmark variance
    print(calculate_all_metrics(flat_equity, benchmark_returns=flat_benchmark))

    print("\n--- Edge Case: Equity Curve Starts at Zero ---")
    zero_start_equity = pd.Series([0, 100, 105, 103],
                                  index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']))
    print(calculate_all_metrics(zero_start_equity))