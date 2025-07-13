import pandas as pd

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).

    Args:
        data (pd.Series): A pandas Series of prices.
        window (int): The moving average window size.

    Returns:
        pd.Series: A pandas Series containing the SMA.
    """
    return data.rolling(window=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).

    Args:
        data (pd.Series): A pandas Series of prices.
        window (int): The moving average window size.

    Returns:
        pd.Series: A pandas Series containing the EMA.
    """
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data (pd.Series): A pandas Series of prices.
        window (int): The RSI window size (default is 14).

    Returns:
        pd.Series: A pandas Series containing the RSI.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
