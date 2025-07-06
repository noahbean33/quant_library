# exploratory_data_analysis.py

# pip install pandas yfinance matplotlib seaborn sktime kats

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Set plotting style
sns.set_theme(context="talk", style="whitegrid",
              palette="colorblind", color_codes=True,
              rc={"figure.figsize": [12, 8]})


def detect_outliers_rolling_stats(df, column, window_size, n_sigmas):
    """
    Identifies outliers using rolling statistics.

    Parameters:
    - df (DataFrame): Input DataFrame
    - column (str): The column to check for outliers
    - window_size (int): The rolling window size
    - n_sigmas (int): The number of standard deviations for the threshold

    Returns:
    - Series: A boolean Series indicating outliers
    """
    df_rolling = df[[column]].rolling(window=window_size).agg(["mean", "std"])
    df_rolling.columns = df_rolling.columns.droplevel()
    df = df.join(df_rolling)
    df["upper"] = df["mean"] + n_sigmas * df["std"]
    df["lower"] = df["mean"] - n_sigmas * df["std"]
    return (df[column] > df["upper"]) | (df[column] < df["lower"])

def detect_outliers_hampel(series, window_length=10):
    """
    Detects outliers using the Hampel filter.

    Parameters:
    - series (Series): The time series to analyze
    - window_length (int): The length of the rolling window

    Returns:
    - Series: A boolean Series indicating outliers
    """
    from sktime.transformations.series.outlier_detection import HampelFilter
    hampel_detector = HampelFilter(window_length=window_length, return_bool=True)
    return hampel_detector.fit_transform(series)

def detect_changepoints(series, detector_type='cusum', **kwargs):
    """
    [Commented out due to kats dependency issues on Python 3.13]
    Detects changepoints in a time series using Kats.
    """
    print("Function 'detect_changepoints' is disabled due to kats installation issues.")
    pass

def detect_trends(series, direction='up', window_size=30, threshold=0.9):
    """
    [Commented out due to kats dependency issues on Python 3.13]
    Detects trends in a time series using the Mann-Kendall test.
    """
    print("Function 'detect_trends' is disabled due to kats installation issues.")
    pass

def get_hurst_exponent(ts, max_lag=20):
    """
    Calculates the Hurst Exponent of a time series.

    Parameters:
    - ts (array-like): The time series
    - max_lag (int): The maximum lag to use

    Returns:
    - float: The Hurst Exponent
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    hurst_exp = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst_exp

def main():
    # Download Tesla data for outlier detection
    tsla_df = yf.download("TSLA", start="2019-01-01", end="2020-12-31", progress=False)
    tsla_df["rtn"] = tsla_df["Adj Close"].pct_change()

    # Test 1: Outlier detection with rolling stats
    print("\nTesting outlier detection with rolling stats...")
    outliers_rolling = detect_outliers_rolling_stats(tsla_df, 'rtn', 21, 3)
    print(f"Found {outliers_rolling.sum()} outliers.")

    # Test 2: Outlier detection with Hampel filter
    print("\nTesting outlier detection with Hampel filter...")
    outliers_hampel = detect_outliers_hampel(tsla_df['Adj Close'])
    print(f"Found {outliers_hampel.sum()} outliers.")

    # Download Apple data for changepoint detection
    aapl_df = yf.download("AAPL", start="2020-01-01", end="2020-12-31", progress=False)

    # Test 3: Changepoint detection (Commented out due to kats installation issues)
    print("\nTesting changepoint detection...")
    detect_changepoints(aapl_df['Adj Close'], detector_type='cusum', change_directions=["increase"])

    # Download NVIDIA data for trend detection
    nvda_df = yf.download("NVDA", start="2020-01-01", end="2020-12-31", progress=False)

    # Test 4: Trend detection (Commented out due to kats installation issues)
    print("\nTesting trend detection...")
    detect_trends(nvda_df['Adj Close'], direction='up')

    # Download S&P 500 data for Hurst exponent
    gspc_df = yf.download("^GSPC", start="2000-01-01", end="2019-12-31", progress=False)

    # Test 5: Hurst Exponent
    print("\nCalculating Hurst Exponent for S&P 500...")
    hurst_exp = get_hurst_exponent(gspc_df["Adj Close"].values, max_lag=100)
    print(f"Hurst Exponent: {hurst_exp:.4f}")

if __name__ == "__main__":
    main()
