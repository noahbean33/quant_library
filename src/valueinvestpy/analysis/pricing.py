import numpy as np
from scipy.stats import norm

from ..simulation.stochastic_processes import simulate_gbm

def black_scholes_analytical(S0: float, E: float, T: float, rf: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculates the price of a European option using the analytical Black-Scholes formula.

    Args:
        S0 (float): Initial stock price.
        E (float): Strike price.
        T (float): Time to maturity in years.
        rf (float): Risk-free interest rate (annual).
        sigma (float): Annual volatility of the stock.
        option_type (str, optional): Type of the option, 'call' or 'put'. Defaults to 'call'.

    Returns:
        float: The price of the European option.
    """
    d1 = (np.log(S0 / E) + (rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S0 * norm.cdf(d1) - E * np.exp(-rf * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = E * np.exp(-rf * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    return price

def black_scholes_monte_carlo(S0: float, E: float, T: float, rf: float, sigma: float, num_simulations: int = 10000, option_type: str = 'call') -> float:
    """
    Calculates the price of a European option using Monte Carlo simulation.

    This function leverages the Geometric Brownian Motion (GBM) simulation for the 
    underlying asset price.

    Args:
        S0 (float): Initial stock price.
        E (float): Strike price.
        T (float): Time to maturity in years.
        rf (float): Risk-free interest rate (annual).
        sigma (float): Annual volatility of the stock.
        num_simulations (int, optional): Number of Monte Carlo simulations. Defaults to 10000.
        option_type (str, optional): Type of the option, 'call' or 'put'. Defaults to 'call'.

    Returns:
        float: The estimated price of the European option.
    """
    # Simulate the stock price paths using GBM
    # We only need one time step for a European option (the price at expiry)
    sim_df = simulate_gbm(S0=S0, mu=rf, sigma=sigma, T=T, N=1, num_simulations=num_simulations)
    
    # Get the stock prices at maturity (the last row of the DataFrame)
    S_T = sim_df.iloc[-1]
    
    # Calculate the payoff for each simulation
    if option_type.lower() == 'call':
        payoff = np.maximum(S_T - E, 0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(E - S_T, 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")
        
    # Discount the average payoff to present value
    option_price = np.exp(-rf * T) * np.mean(payoff)
    
    return option_price
