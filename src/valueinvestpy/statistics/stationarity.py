import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series: pd.Series, signif: float = 0.05) -> dict:
    """Performs the Augmented Dickey-Fuller test for stationarity.

    Args:
        series (pd.Series): The time series to test.
        signif (float, optional): The significance level. Defaults to 0.05.

    Returns:
        dict: A dictionary containing the test statistic, p-value, and whether the series is stationary.
    """
    adf_result = adfuller(series, autolag='AIC')
    p_value = adf_result[1]
    is_stationary = p_value < signif
    return {'Test Statistic': adf_result[0], 'p-value': p_value, 'Stationary': is_stationary}

def kpss_test(series: pd.Series, signif: float = 0.05) -> dict:
    """Performs the KPSS test for stationarity.

    Args:
        series (pd.Series): The time series to test.
        signif (float, optional): The significance level. Defaults to 0.05.

    Returns:
        dict: A dictionary containing the test statistic, p-value, and whether the series is stationary.
    """
    kpss_result = kpss(series, regression='c', nlags="auto")
    p_value = kpss_result[1]
    is_stationary = p_value >= signif
    return {'Test Statistic': kpss_result[0], 'p-value': p_value, 'Stationary': is_stationary}
