import pandas as pd
import numpy as np
from config.logging_config import logger


def calculate_all_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """Calculates a comprehensive set of performance metrics."""
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.empty:
        return {metric: 0.0 for metric in [
            'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Omega Ratio'
        ]}

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0.0

    annualized_volatility = daily_returns.std() * np.sqrt(252)

    max_dd = calculate_max_drawdown(equity_curve)

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
    """Calculates the annualized Sharpe Ratio."""
    if daily_returns.std() == 0: return 0.0
    excess_returns = daily_returns - (risk_free_rate / 252)
    return (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)


def calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """Calculates the annualized Sortino Ratio."""
    target_return = risk_free_rate / 252
    downside_returns = daily_returns[daily_returns < target_return]
    downside_std = downside_returns.std()
    if downside_std == 0: return np.inf if daily_returns.mean() > target_return else 0.0

    expected_return = daily_returns.mean()
    sortino_ratio = (expected_return - target_return) / downside_std * np.sqrt(252)
    return sortino_ratio


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates the maximum drawdown."""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calculates the Calmar Ratio."""
    if max_drawdown == 0: return np.inf
    return annualized_return / abs(max_drawdown)


def calculate_omega_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    """Calculates the Omega Ratio."""
    target_return = risk_free_rate / 252
    returns_less_target = daily_returns - target_return

    gain = returns_less_target[returns_less_target > 0].sum()
    loss = abs(returns_less_target[returns_less_target < 0].sum())

    if loss == 0: return np.inf
    return gain / loss