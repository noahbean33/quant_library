import pandas as pd
from valueinvestpy.technical_analysis.moving_averages import sma

def bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    middle_band = sma(data, window)
    std_dev = data.rolling(window=window).std()

    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    return pd.DataFrame({
        'Upper': upper_band,
        'Middle': middle_band,
        'Lower': lower_band
    })

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2, price_source: str = 'Close') -> pd.DataFrame:
    """
    Calculates Bollinger Bands for a given dataset.

    Args:
        data (pd.DataFrame): DataFrame with at least 'High', 'Low', and 'Close' columns.
        period (int): The moving average period.
        std_dev (int): The number of standard deviations for the bands.
        price_source (str): The column to use for calculation. Can be 'Close' or 'Typical'.
                           'Typical' is calculated as (High + Low + Close) / 3.

    Returns:
        pd.DataFrame: A DataFrame with 'MiddleBand', 'UpperBand', and 'LowerBand' columns.
    """
    if price_source == 'Typical':
        price = (data['High'] + data['Low'] + data['Close']) / 3
    else:
        price = data['Close']

    middle_band = price.rolling(window=period).mean()
    std = price.rolling(window=period).std()

    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    bands = pd.DataFrame({
        'MiddleBand': middle_band,
        'UpperBand': upper_band,
        'LowerBand': lower_band
    })

    return bands
