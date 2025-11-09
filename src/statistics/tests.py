import pandas as pd
from scipy.stats import linregress
import statsmodels.tsa.stattools as ts

def engle_granger_cointegration_test(series1: pd.Series, series2: pd.Series):
    """
    Tests for cointegration between two time series using the Engle-Granger method.

    This method involves two steps:
    1. Run a linear regression between the two series.
    2. Test the residuals of the regression for stationarity using the
       Augmented Dickey-Fuller (ADF) test.

    If the residuals are stationary, the two series are considered cointegrated.

    Args:
        series1 (pd.Series): The first time series (e.g., stock prices).
        series2 (pd.Series): The second time series (e.g., stock prices).

    Returns:
        tuple: The results of the ADF test on the residuals, which includes:
               - ADF statistic
               - p-value
               - Number of lags used
               - Number of observations
               - Critical values at 1%, 5%, and 10% levels
               - AIC (if available)
    """
    # Perform linear regression to find the hedge ratio (slope)
    # We regress series1 on series2 to find the relationship
    result = linregress(x=series2.values, y=series1.values)
    
    # Calculate the residuals of the regression
    residuals = series1 - result.slope * series2
    
    # Perform the Augmented Dickey-Fuller test on the residuals
    adf_result = ts.adfuller(residuals)
    
    return adf_result
