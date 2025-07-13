import numpy as np

def generate_ornstein_uhlenbeck_process(dt: float = 0.1, theta: float = 1.2, mu: float = 0.5, sigma: float = 0.3, n: int = 10000) -> np.ndarray:
    """
    Generates a time series that follows an Ornstein-Uhlenbeck (OU) process.

    The OU process is a stochastic process that describes the velocity of a massive
    Brownian particle under the influence of friction. It is often used in finance
    to model mean-reverting variables like interest rates or pairs trading spreads.

    The process is defined by the stochastic differential equation:
    dx(t) = theta * (mu - x(t)) * dt + sigma * dW(t)

    where:
        - theta: The speed of mean reversion.
        - mu: The long-term mean of the process.
        - sigma: The volatility or magnitude of the random fluctuations.
        - dW(t): A Wiener process or Brownian motion.

    Args:
        dt (float, optional): The time step increment. Defaults to 0.1.
        theta (float, optional): The speed of mean reversion. Defaults to 1.2.
        mu (float, optional): The long-term mean. Defaults to 0.5.
        sigma (float, optional): The volatility. Defaults to 0.3.
        n (int, optional): The number of time steps to generate. Defaults to 10000.

    Returns:
        np.ndarray: An array representing the generated Ornstein-Uhlenbeck process.
    """
    # Initialize the process array and set the starting point to 0
    x = np.zeros(n)

    # Generate the process step-by-step
    for t in range(1, n):
        drift = theta * (mu - x[t-1]) * dt
        diffusion = sigma * np.random.normal(0, np.sqrt(dt))
        x[t] = x[t-1] + drift + diffusion

    return x
