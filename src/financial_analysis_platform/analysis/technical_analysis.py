# src/financial_analysis_platform/analysis/technical_analysis.py
import pandas as pd
import numpy as np

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Calculates Bollinger Bands for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' price column.
        window (int): The moving average window.
        num_std (int): The number of standard deviations.

    Returns:
        pd.DataFrame: DataFrame with 'MA', 'Upper', and 'Lower' band columns.
    """
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['StdDev'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['MA'] + (df['StdDev'] * num_std)
    df['Lower'] = df['MA'] - (df['StdDev'] * num_std)
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' price column.
        window (int): The RSI window.

    Returns:
        pd.DataFrame: DataFrame with the 'RSI' column.
    """
    delta = df['Close'].diff(1)

    gain = delta.copy()
    loss = delta.copy()

    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD).

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' price column.
        short_window (int): The short-term EMA window.
        long_window (int): The long-term EMA window.
        signal_window (int): The signal line EMA window.

    Returns:
        pd.DataFrame: DataFrame with 'MACD', 'Signal_Line', and 'MACD_Histogram' columns.
    """
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Average True Range (ATR).

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns.
        window (int): The ATR window.

    Returns:
        pd.DataFrame: DataFrame with 'ATR' column.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    true_range.iloc[0] = np.nan  # First TR is undefined

    # Using ewm to calculate the smoothed moving average for ATR
    df['ATR'] = true_range.ewm(alpha=1/window, adjust=False).mean()
    
    return df

def calculate_stochastic_oscillator(df: pd.DataFrame, n: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Calculates the Stochastic Oscillator.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns.
        n (int): The look-back period for the highest high and lowest low.
        d_window (int): The moving average window for %D.

    Returns:
        pd.DataFrame: DataFrame with '%K' and '%D' columns.
    """
    low_n = df['Low'].rolling(window=n).min()
    high_n = df['High'].rolling(window=n).max()

    df['%K'] = (df['Close'] - low_n) / (high_n - low_n) * 100
    df['%D'] = df['%K'].rolling(window=d_window).mean()

    return df
