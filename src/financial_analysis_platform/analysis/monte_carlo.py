import numpy as np
import pandas as pd

def monte_carlo_simulation(df: pd.DataFrame, n_simulations: int, n_days: int) -> pd.DataFrame:
    """
    Performs a Monte Carlo simulation for future stock price paths.

    Args:
        df (pd.DataFrame): DataFrame with historical stock data, including a 'Close' column.
        n_simulations (int): The number of simulations to run.
        n_days (int): The number of future days to simulate.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated price paths.
    """
    # Calculate daily log returns
    log_returns = np.log(1 + df['Close'].pct_change().dropna())

    # Calculate drift and volatility
    u = log_returns.mean()
    var = log_returns.var()
    drift = (u - (0.5 * var)).item()
    stdev = log_returns.std().item()

    # Generate random variables
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (n_days, n_simulations)))

    # Create price paths
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = df['Close'].iloc[-1]
    for t in range(1, n_days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    return pd.DataFrame(price_paths)
