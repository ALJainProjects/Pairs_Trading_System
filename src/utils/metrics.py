import pandas as pd
import numpy as np
from config.logging_config import logger

def calculate_sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sharpe Ratio of the portfolio.

    Args:
        equity_curve (pd.Series): Time series of portfolio equity over time.
        risk_free_rate (float): Annual risk-free rate (expressed as a decimal, e.g. 0.02 for 2%).

    Returns:
        float: Annualized Sharpe Ratio. Returns 0.0 if std is 0.
    """
    logger.info("Calculating Sharpe Ratio.")
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) == 0:
        logger.warning("No daily returns to calculate Sharpe Ratio. Returning 0.0.")
        return 0.0
    ret_std = daily_returns.std()
    if ret_std == 0:
        logger.warning("Standard deviation is 0. Returning 0.0 for Sharpe Ratio.")
        return 0.0

    # Subtract daily risk-free portion
    excess = daily_returns - (risk_free_rate / 252.0)
    sr = excess.mean() / ret_std * np.sqrt(252)
    return sr

def calculate_sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sortino Ratio of the portfolio.

    Args:
        equity_curve (pd.Series): Portfolio equity time series.
        risk_free_rate (float): Annual risk-free rate as a decimal.

    Returns:
        float: Annualized Sortino Ratio. Returns NaN if negative std is 0.
    """
    logger.info("Calculating Sortino Ratio.")
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) == 0:
        logger.warning("No daily returns to calculate Sortino Ratio. Returning NaN.")
        return np.nan

    negative = daily_returns[daily_returns < 0]
    neg_std = negative.std()
    if neg_std == 0:
        logger.warning("No negative returns or zero negative std => returning NaN for Sortino.")
        return np.nan

    excess = daily_returns - (risk_free_rate / 252.0)
    sortino = excess.mean() / (neg_std * np.sqrt(252))
    return sortino

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum drawdown in the equity curve.

    Args:
        equity_curve (pd.Series): Time series of portfolio equity.

    Returns:
        float: The max drawdown as a negative fraction (e.g. -0.30 for 30% drawdown).
               If the equity never drops below peak, returns 0.0.
    """
    logger.info("Calculating Max Drawdown.")
    if equity_curve.empty:
        logger.warning("Equity curve is empty. Returning 0.0 for max drawdown.")
        return 0.0
    cummax = equity_curve.cummax()
    drawdowns = (equity_curve - cummax) / cummax
    # Typically drawdowns are negative, so max drawdown is the minimum. If no drawdown, min is 0 or above.
    mdd = drawdowns.min()
    return float(mdd if mdd < 0 else 0.0)

def calculate_calmar_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Calmar Ratio = annualized return / absolute max drawdown.

    Args:
        equity_curve (pd.Series): Portfolio equity series.
        risk_free_rate (float): Unused here, but kept for consistent signature.

    Returns:
        float: Calmar ratio, or NaN if drawdown is 0 or equity_curve is too short.
    """
    logger.info("Calculating Calmar Ratio.")
    mdd = calculate_max_drawdown(equity_curve)  # negative or 0
    if mdd == 0:
        logger.warning("No drawdown => returning NaN for Calmar.")
        return np.nan

    # Estimate annualized return via simple start/end approach
    if len(equity_curve) < 2:
        logger.warning("Equity curve too short for Calmar. Returning NaN.")
        return np.nan

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days <= 0:
        logger.warning("Non-positive time span => returning NaN.")
        return np.nan

    years = days / 365.0
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    return annualized_return / abs(mdd)

