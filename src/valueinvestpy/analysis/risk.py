import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional
from ..data_fetch import fetch_stock_data

def calculate_var_monte_carlo(
    initial_investment: float, 
    returns: pd.Series, 
    confidence_level: float = 0.99, 
    horizon_days: int = 1, 
    num_simulations: int = 10000
) -> float:
    """
    Calculates the Value at Risk (VaR) using Monte Carlo simulation.

    This method simulates future investment values based on historical returns
    and calculates VaR from the distribution of simulated outcomes.

    Args:
        initial_investment (float): The total value of the investment.
        returns (pd.Series): A pandas Series of historical returns (e.g., daily log returns).
        confidence_level (float, optional): The confidence level for the VaR calculation. 
                                           Defaults to 0.99 (99%).
        horizon_days (int, optional): The time horizon in days. Defaults to 1.
        num_simulations (int, optional): The number of Monte Carlo simulations to run. 
                                       Defaults to 10000.

    Returns:
        float: The calculated Value at Risk. A positive value represents a potential loss.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("The 'returns' argument must be a pandas Series.")

    # Calculate mean and standard deviation of returns
    mu = np.mean(returns)
    sigma = np.std(returns)

    # Generate random price paths
    rand = np.random.normal(0, 1, [1, num_simulations])
    
    # Simulate the final price using Geometric Brownian Motion
    final_prices = initial_investment * np.exp(
        horizon_days * (mu - 0.5 * sigma ** 2) + sigma * np.sqrt(horizon_days) * rand
    )

    # Find the price at the specified percentile
    percentile = np.percentile(final_prices, (1 - confidence_level) * 100)

    # Calculate VaR as the difference between the initial investment and the percentile price
    var = initial_investment - percentile

    return var

def calculate_var(investment: float,
                  returns: pd.Series,
                  confidence_level: float = 0.99,
                  horizon_days: int = 1) -> float:
    """
    Calculates the Value at Risk (VaR) for a given investment.

    This method assumes that the returns are normally distributed.

    Args:
        investment (float): The total value of the investment.
        returns (pd.Series): A pandas Series of historical returns (e.g., daily log returns).
        confidence_level (float, optional): The confidence level for the VaR calculation. 
                                           Defaults to 0.99 (99%).
        horizon_days (int, optional): The time horizon in days. Defaults to 1.

    Returns:
        float: The calculated Value at Risk. A positive value represents a potential loss.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("The 'returns' argument must be a pandas Series.")

    # Calculate mean and standard deviation of returns
    mu = np.mean(returns)
    sigma = np.std(returns)

    # Calculate VaR using the variance-covariance method
    # norm.ppf gives the z-score for the specified confidence level
    var = investment * (mu * horizon_days - sigma * np.sqrt(horizon_days) * norm.ppf(1 - confidence_level))

    # We return the absolute value since VaR is typically expressed as a positive loss amount
    return abs(var)
