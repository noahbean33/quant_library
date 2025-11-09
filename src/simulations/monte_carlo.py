import numpy as np
import pandas as pd

def geometric_brownian_motion(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_sims: int) -> pd.DataFrame:
    """Simulates stock price paths using Geometric Brownian Motion (GBM).

    Args:
        S0 (float): Initial stock price.
        mu (float): Expected return (drift).
        sigma (float): Volatility.
        T (float): Time horizon in years.
        n_steps (int): Number of time steps.
        n_sims (int): Number of simulations (paths) to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated stock price paths.
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    S = np.zeros((n_steps + 1, n_sims))
    S[0] = S0

    for i in range(1, n_steps + 1):
        z = np.random.standard_normal(n_sims)
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    return pd.DataFrame(S, index=t)
