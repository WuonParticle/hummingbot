import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the Sharpe ratio for a given series of returns.

    Args:
        returns (pd.Series): A pandas Series of returns.
        risk_free_rate (float): The risk-free rate, defaults to 0.

    Returns:
        float: The calculated Sharpe ratio.
    """
    if returns.empty or len(returns) <= 1:
        return 0.0

    excess_returns = returns - risk_free_rate
    std_dev = returns.std()

    if std_dev == 0:
        # Avoid division by zero; if std is 0, returns are constant.
        # If mean return > risk_free_rate, Sharpe is infinite (or very large).
        # If mean return <= risk_free_rate, Sharpe is 0 or negative.
        return np.inf if excess_returns.mean() > 0 else 0.0

    return excess_returns.mean() / std_dev


def sortino_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
    """
    Calculates the Sortino ratio for a given series of returns.

    Args:
        returns (pd.Series): A pandas Series of returns.
        target_return (float): The target return, defaults to 0 (risk-free rate).

    Returns:
        float: The calculated Sortino ratio.
    """
    if returns.empty:
        return 0.0

    # Calculate the average return relative to the target
    average_return = returns.mean()
    target_adjusted_average_return = average_return - target_return

    # Calculate downside deviation (standard deviation of returns below the target return)
    downside_returns = returns[returns < target_return]
    if downside_returns.empty:
        # If no returns are below target, downside deviation is 0.
        # If average return > target, Sortino is infinite.
        # Otherwise, it implies average return <= target with no downside, so 0.
        return np.inf if target_adjusted_average_return > 0 else 0.0

    # Calculate the square differences from the target return for downside returns
    squared_diffs = np.square(downside_returns - target_return)
    # Calculate the mean of these squared differences (downside variance)
    downside_variance = np.mean(squared_diffs)
    # Calculate the square root (downside deviation)
    downside_deviation = np.sqrt(downside_variance)

    if downside_deviation == 0:
        # Avoid division by zero. Similar logic as above.
        return np.inf if target_adjusted_average_return > 0 else 0.0

    sortino = target_adjusted_average_return / downside_deviation
    return sortino
