import numpy as np
import pandas as pd

def simulate_gbm(S0: float, mu: float, sigma: float, T: int = 1, N: int = 252, num_simulations: int = 1) -> pd.DataFrame:
    """
    Simulates stock price paths using Geometric Brownian Motion (GBM).

    Args:
        S0 (float): Initial stock price.
        mu (float): Annual drift (expected return).
        sigma (float): Annual volatility.
        T (int, optional): Time horizon in years. Defaults to 1.
        N (int, optional): Number of time steps in the simulation. Defaults to 252 (trading days in a year).
        num_simulations (int, optional): Number of simulation paths to generate. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a simulated price path 
                      and the index represents the time steps.
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)

    # W has shape (N+1, num_simulations)
    W = np.random.standard_normal(size=(N + 1, num_simulations))
    W[0, :] = 0  # Start at 0
    W = np.cumsum(W, axis=0) * np.sqrt(dt)

    # X has shape (N+1, num_simulations)
    X = (mu - 0.5 * sigma ** 2) * t[:, np.newaxis] + sigma * W
    S = S0 * np.exp(X)

    # Create a DataFrame
    sim_df = pd.DataFrame(S, index=t)
    sim_df.index.name = 'Time (Years)'
    sim_df.columns = [f'Simulation_{i+1}' for i in range(num_simulations)]

    return sim_df

def simulate_ornstein_uhlenbeck(x0: float, theta: float, mu: float, sigma: float, T: int = 1, N: int = 252, num_simulations: int = 1) -> pd.DataFrame:
    """
    Simulates paths for the Ornstein-Uhlenbeck mean-reverting process.

    Args:
        x0 (float): Initial value of the process.
        theta (float): The speed of reversion to the mean.
        mu (float): The long-term mean of the process.
        sigma (float): The volatility of the process.
        T (int, optional): Time horizon in years. Defaults to 1.
        N (int, optional): Number of time steps. Defaults to 252.
        num_simulations (int, optional): Number of simulation paths to generate. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a simulated path.
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)

    # Initialize the array for the process paths
    x = np.zeros((N + 1, num_simulations))
    x[0] = x0

    # Generate the paths
    for i in range(1, N + 1):
        dW = np.random.standard_normal(num_simulations) * np.sqrt(dt)
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * dW

    # Create a DataFrame
    sim_df = pd.DataFrame(x, index=t)
    sim_df.index.name = 'Time (Years)'
    sim_df.columns = [f'Simulation_{i+1}' for i in range(num_simulations)]

    return sim_df

def simulate_random_walk(num_steps: int = 1000, num_simulations: int = 1) -> pd.DataFrame:
    """
    Simulates paths for a simple random walk.

    A simple random walk is a stochastic process formed by the cumulative sum 
    of independent, identically distributed random variables that take the 
    value +1 or -1 with equal probability.

    Args:
        num_steps (int, optional): The number of steps in the walk. Defaults to 1000.
        num_simulations (int, optional): The number of simulation paths to generate. 
                                         Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a simulated path.
    """
    # Generate random steps: -1 or 1
    steps = np.random.choice([-1, 1], size=(num_steps, num_simulations))
    
    # Prepend a row of zeros for the initial position
    walks = np.vstack([np.zeros((1, num_simulations)), steps])
    
    # Cumulatively sum the steps to get the paths
    paths = np.cumsum(walks, axis=0)
    
    # Create a DataFrame
    sim_df = pd.DataFrame(paths)
    sim_df.index.name = 'Steps'
    sim_df.columns = [f'Simulation_{i+1}' for i in range(num_simulations)]
    
    return sim_df

def simulate_vasicek_model(r0: float, kappa: float, theta: float, sigma: float, T: int = 1, N: int = 252, num_simulations: int = 1) -> pd.DataFrame:
    """
    Simulates interest rate paths using the Vasicek model.

    The Vasicek model is an application of the Ornstein-Uhlenbeck process
    to model the dynamics of a short-term interest rate.

    Args:
        r0 (float): The initial interest rate.
        kappa (float): The speed of reversion to the mean (theta).
        theta (float): The long-term mean interest rate.
        sigma (float): The volatility of the interest rate.
        T (int, optional): Time horizon in years. Defaults to 1.
        N (int, optional): Number of time steps. Defaults to 252.
        num_simulations (int, optional): Number of simulation paths to generate. 
                                         Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a simulated interest rate path.
    """
    # The Vasicek model is a specific case of the Ornstein-Uhlenbeck process.
    # We can reuse the existing simulation function by mapping the parameters:
    # x0 -> r0 (initial value)
    # theta -> kappa (speed of reversion)
    # mu -> theta (long-term mean)
    return simulate_ornstein_uhlenbeck(
        x0=r0,
        theta=kappa,
        mu=theta,
        sigma=sigma,
        T=T,
        N=N,
        num_simulations=num_simulations
    )

def simulate_wiener_process(T: int = 1, N: int = 252, num_simulations: int = 1) -> pd.DataFrame:
    """
    Simulates paths for the standard Wiener process (Brownian Motion).

    Args:
        T (int, optional): Time horizon in years. Defaults to 1.
        N (int, optional): Number of time steps. Defaults to 252.
        num_simulations (int, optional): Number of simulation paths to generate. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a simulated path.
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)

    # Initialize the array for the process paths, starting at 0
    W = np.zeros((N + 1, num_simulations))

    # Generate the random increments
    random_increments = np.random.standard_normal((N, num_simulations)) * np.sqrt(dt)

    # Cumulatively sum the increments to get the paths
    W[1:, :] = np.cumsum(random_increments, axis=0)

    # Create a DataFrame
    sim_df = pd.DataFrame(W, index=t)
    sim_df.index.name = 'Time (Years)'
    sim_df.columns = [f'Simulation_{i+1}' for i in range(num_simulations)]

    return sim_df


