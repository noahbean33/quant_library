import numpy as np
import pandas as pd
from valueinvestpy.technical_analysis.oscillators import rsi
from valueinvestpy.technical_analysis.moving_averages import sma

def construct_lagged_return_features(data, lags=2):
    """
    Constructs features based on lagged returns for machine learning models.

    Args:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        lags (int): The number of lagged periods to create as features.

    Returns:
        pd.DataFrame: The original DataFrame with added feature columns.
    """
    # Calculate the lagged adjusted closing prices
    for i in range(0, lags):
        data[f'Lag{i+1}'] = data['Close'].shift(i+1)

    # Calculate the percent change
    data["Today Change"] = data["Close"].pct_change() * 100

    # Calculate the lags in percentage (normalization)
    for i in range(0, lags):
        data[f'Lag{i+1}'] = data[f'Lag{i+1}'].pct_change() * 100

    # Direction - the target variable
    data['Direction'] = np.where(data['Today Change'] > 0, 1, -1)

    return data

def construct_technical_indicator_features(data, ma_period=60, rsi_period=14):
    """
    Constructs features for a machine learning model using technical indicators.

    Args:
        data (pd.DataFrame): DataFrame with 'Open' and 'Close' columns.
        ma_period (int): The period for the simple moving average.
        rsi_period (int): The period for the Relative Strength Index.

    Returns:
        pd.DataFrame: The original DataFrame with added feature columns.
    """
    data['SMA'] = sma(data['Close'], window=ma_period)
    # These are the 2 features
    data['trend'] = (data['Open'] - data['SMA']) * 100
    data['RSI'] = rsi(data['Close'], window=rsi_period) / 100
    # We need the target variables (labels)
    data['direction'] = np.where(data['Close'] - data['Open'] > 0, 1, -1)
    return data
