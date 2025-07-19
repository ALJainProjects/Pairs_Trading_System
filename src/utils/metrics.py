import pandas as pd
import numpy as np
from config.logging_config import logger  # Assuming this is configured


def calculate_all_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """
    Calculates a comprehensive set of performance metrics for an equity curve.

    Args:
        equity_curve (pd.Series): A time series of portfolio equity values.
        risk_free_rate (float): The annualized risk-free rate (e.g., 0.02 for 2%).

    Returns:
        dict: A dictionary containing various performance metrics.
    """
    if equity_curve.empty:
        logger.warning("Equity curve is empty. Cannot calculate metrics.")
        return {metric: 0.0 for metric in [
            'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Omega Ratio'
        ]}

    # Ensure equity curve starts at a positive value for return calculations
    if equity_curve.iloc[0] <= 0:
        logger.warning(
            "Equity curve starts at zero or negative value. Return calculations may be invalid. Returning 0.0 for returns.")
        # Attempt to proceed with NaN/0.0 values where division by zero might occur.
        # Set a default positive value for calculations if starting at zero, but be cautious.
        # For simplicity, if starting at <=0, most return-based metrics are invalid.
        return {metric: 0.0 for metric in [
            'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Omega Ratio'
        ]}

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.empty:
        logger.warning(
            "Daily returns series is empty after pct_change/dropna. Insufficient data. Returning 0.0 for metrics.")
        return {metric: 0.0 for metric in [
            'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Omega Ratio'
        ]}

    # Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualized Return
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0.0

    # Annualized Volatility
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    # Handle case where volatility is zero (e.g., flat returns)
    if annualized_volatility == 0:
        annualized_volatility = 1e-9  # Small non-zero value to avoid division by zero in ratios

    # Max Drawdown
    max_dd = calculate_max_drawdown(equity_curve)  # This function now returns a positive value

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': calculate_sharpe_ratio(daily_returns, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(daily_returns, risk_free_rate),
        'Max Drawdown': max_dd,
        'Calmar Ratio': calculate_calmar_ratio(annualized_return, max_dd),
        'Omega Ratio': calculate_omega_ratio(daily_returns, risk_free_rate)
    }


def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the annualized Sharpe Ratio.

    Args:
        daily_returns (pd.Series): Daily percentage returns of the portfolio.
        risk_free_rate (float): Annualized risk-free rate.

    Returns:
        float: The annualized Sharpe Ratio. Returns 0.0 if standard deviation is zero.
    """
    daily_risk_free_rate = risk_free_rate / 252
    excess_returns = daily_returns - daily_risk_free_rate

    std_dev = daily_returns.std()
    if std_dev == 0: return 0.0  # Avoid division by zero

    sharpe = (excess_returns.mean() / std_dev) * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the annualized Sortino Ratio.

    Args:
        daily_returns (pd.Series): Daily percentage returns of the portfolio.
        risk_free_rate (float): Annualized risk-free rate.

    Returns:
        float: The annualized Sortino Ratio. Returns np.inf if downside standard deviation is zero and mean is positive.
    """
    daily_risk_free_rate = risk_free_rate / 252
    downside_returns = daily_returns[daily_returns < daily_risk_free_rate]

    downside_std = downside_returns.std()
    if downside_std == 0:  # Avoid division by zero
        # If no downside volatility, Sortino is infinite if returns are above target, else 0.
        return np.inf if daily_returns.mean() > daily_risk_free_rate else 0.0

    sortino_ratio = (daily_returns.mean() - daily_risk_free_rate) / downside_std * np.sqrt(252)
    return sortino_ratio


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the maximum drawdown of an equity curve.

    Args:
        equity_curve (pd.Series): A time series of portfolio equity values.

    Returns:
        float: The maximum drawdown as a positive decimal (e.g., 0.25 for 25% drawdown). Returns 0.0 if empty or constant.
    """
    if equity_curve.empty or equity_curve.iloc[0] <= 0:  # Handle empty or non-positive starting equity
        return 0.0

    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    # Max drawdown is the minimum (most negative) value in the drawdown series, take absolute
    max_dd = abs(drawdown.min())

    # Handle case where drawdown is NaN (e.g., constant equity curve)
    return max_dd if pd.notna(max_dd) else 0.0


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calculates the Calmar Ratio.

    Args:
        annualized_return (float): The annualized return of the portfolio.
        max_drawdown (float): The maximum drawdown (as a positive decimal).

    Returns:
        float: The Calmar Ratio. Returns np.inf if max_drawdown is zero.
    """
    if max_drawdown == 0: return np.inf  # Avoid division by zero
    return annualized_return / max_drawdown


def calculate_omega_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the Omega Ratio.

    Args:
        daily_returns (pd.Series): Daily percentage returns of the portfolio.
        risk_free_rate (float): Annualized risk-free rate.

    Returns:
        float: The Omega Ratio. Returns np.inf if total loss is zero.
    """
    daily_risk_free_rate = risk_free_rate / 252
    returns_less_target = daily_returns - daily_risk_free_rate

    # Sum of positive excess returns (gains)
    gain = returns_less_target[returns_less_target > 0].sum()
    # Sum of absolute negative excess returns (losses)
    loss = abs(returns_less_target[returns_less_target < 0].sum())

    if loss == 0: return np.inf  # Avoid division by zero
    return gain / loss