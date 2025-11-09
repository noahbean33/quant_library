import numpy as np
import pandas as pd
from typing import Optional
from ..data_fetch import fetch_stock_data
from scipy import stats

# Default risk-free rate (e.g., 10-year Treasury yield)
DEFAULT_RISK_FREE_RATE = 0.05
MONTHS_IN_YEAR = 12

def calculate_capm(stock_ticker: str, 
                   market_ticker: str = '^GSPC', 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, 
                   risk_free_rate: float = DEFAULT_RISK_FREE_RATE):
    """
    Calculates the Capital Asset Pricing Model (CAPM) for a given stock.

    Args:
        stock_ticker (str): The ticker symbol of the stock.
        market_ticker (str, optional): The ticker for the market index. Defaults to '^GSPC'.
        start_date (Optional[str], optional): The start date for data (YYYY-MM-DD).
        end_date (Optional[str], optional): The end date for data (YYYY-MM-DD).
        risk_free_rate (float, optional): The annual risk-free rate. Defaults to 0.05.

    Returns:
        dict: A dictionary containing CAPM results ('beta', 'alpha', 'expected_return', 'data'),
              or an empty dictionary if data cannot be fetched.
    """
    # 1. Download stock and market data
    stock_data = fetch_stock_data(stock_ticker, start_date=start_date, end_date=end_date)
    market_data = fetch_stock_data(market_ticker, start_date=start_date, end_date=end_date)

    if stock_data.empty or market_data.empty:
        print("Could not fetch data for stock or market. Cannot calculate CAPM.")
        return {}

    # 2. Resample to monthly and calculate log returns
    data = pd.DataFrame({
        'stock': stock_data['Close'],
        'market': market_data['Close']
    }).resample('ME').last()

    log_returns = np.log(data / data.shift(1)).dropna()

    # 3. Calculate Beta and Alpha using linear regression
    beta, alpha = np.polyfit(log_returns['market'], log_returns['stock'], deg=1)

    # 4. Calculate expected return
    market_return_annual = log_returns['market'].mean() * MONTHS_IN_YEAR
    expected_return = risk_free_rate + beta * (market_return_annual - risk_free_rate)

    return {
        'beta': beta,
        'alpha': alpha,
        'expected_return': expected_return,
        'returns_data': log_returns
    }

def calculate_black_scholes(S: float, E: float, T: float, rf: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculates the price of a European option using the Black-Scholes model.

    Args:
        S (float): Current price of the underlying asset.
        E (float): Strike price of the option.
        T (float): Time to expiration (in years).
        rf (float): Annual risk-free interest rate.
        sigma (float): Volatility of the underlying asset's returns.
        option_type (str, optional): Type of the option, 'call' or 'put'. Defaults to 'call'.

    Returns:
        float: The Black-Scholes price of the option.
    
    Raises:
        ValueError: If option_type is not 'call' or 'put'.
    """
    d1 = (np.log(S / E) + (rf + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * stats.norm.cdf(d1) - E * np.exp(-rf * T) * stats.norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = E * np.exp(-rf * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return price

