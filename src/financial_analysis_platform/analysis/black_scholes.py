# black_scholes.py

import numpy as np
from scipy import stats
from numpy import log, exp, sqrt

def call_option_price(S: float, E: float, T: float, rf: float, sigma: float) -> float:
    """Calculate the Black-Scholes price for a European call option.

    Args:
        S: Underlying stock price at time t=0.
        E: Strike price of the option.
        T: Time to expiration in years.
        rf: Risk-free interest rate.
        sigma: Volatility of the underlying stock.

    Returns:
        The price of the call option.
    """
    d1 = (log(S / E) + (rf + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = S * stats.norm.cdf(d1) - E * exp(-rf * T) * stats.norm.cdf(d2)
    return price

def put_option_price(S: float, E: float, T: float, rf: float, sigma: float) -> float:
    """Calculate the Black-Scholes price for a European put option.

    Args:
        S: Underlying stock price at time t=0.
        E: Strike price of the option.
        T: Time to expiration in years.
        rf: Risk-free interest rate.
        sigma: Volatility of the underlying stock.

    Returns:
        The price of the put option.
    """
    d1 = (log(S / E) + (rf + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = E * exp(-rf * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return price

class OptionPricing:
    """Class for option pricing using Monte Carlo simulation."""

    def __init__(self, S0: float, E: float, T: float, rf: float, sigma: float, iterations: int):
        """Initialize the OptionPricing class.

        Args:
            S0: Underlying stock price at time t=0.
            E: Strike price of the option.
            T: Time to expiration in years.
            rf: Risk-free interest rate.
            sigma: Volatility of the underlying stock.
            iterations: Number of Monte Carlo iterations.
        """
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self) -> float:
        """Calculate the call option price using Monte Carlo simulation.

        Returns:
            The simulated price of the call option.
        """
        rand = np.random.normal(0, 1, self.iterations)
        stock_price = self.S0 * np.exp(
            (self.rf - 0.5 * self.sigma ** 2) * self.T
            + self.sigma * sqrt(self.T) * rand
        )
        payoffs = np.maximum(stock_price - self.E, 0)
        price = exp(-self.rf * self.T) * np.mean(payoffs)
        return price

    def put_option_simulation(self) -> float:
        """Calculate the put option price using Monte Carlo simulation.

        Returns:
            The simulated price of the put option.
        """
        rand = np.random.normal(0, 1, self.iterations)
        stock_price = self.S0 * np.exp(
            (self.rf - 0.5 * self.sigma ** 2) * self.T
            + self.sigma * sqrt(self.T) * rand
        )
        payoffs = np.maximum(self.E - stock_price, 0)
        price = exp(-self.rf * self.T) * np.mean(payoffs)
        return price

