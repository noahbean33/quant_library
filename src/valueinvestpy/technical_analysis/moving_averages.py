import pandas as pd

def sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate the Simple Moving Average (SMA)."""
    return data.rolling(window=window).mean()

def ema(data: pd.Series, window: int) -> pd.Series:
    """Calculate the Exponential Moving Average (EMA)."""
    return data.ewm(span=window, adjust=False).mean()
