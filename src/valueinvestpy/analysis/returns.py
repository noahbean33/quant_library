import pandas as pd
import numpy as np

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculates the logarithmic returns for a given price series.

    Log returns are calculated as ln(P_t / P_{t-1}).

    Args:
        prices (pd.Series): A pandas Series of asset prices, indexed by date.

    Returns:
        pd.Series: A pandas Series of logarithmic returns, with the first 
                   entry being NaN.
    """
    return np.log(prices / prices.shift(1))
