# src/financial_analysis_platform/analysis/time_series_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import DecomposeResult, STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from typing import List, Optional, Tuple, Union

def adf_test(x: Union[pd.Series, np.ndarray]) -> pd.Series:
    """Perform the Augmented Dickey-Fuller test for stationarity.

    Null Hypothesis: The time series is not stationary.
    Alternate Hypothesis: The time series is stationary.

    Args:
        x: The time series to be checked for stationarity.

    Returns:
        A pandas Series with the ADF test's results.
    """
    indices = ['Test Statistic', 'p-value', '# of Lags Used', '# of Observations Used']
    adf_test_results = adfuller(x, autolag='AIC')
    results = pd.Series(adf_test_results[0:4], index=indices)
    for key, value in adf_test_results[4].items():
        results[f'Critical Value ({key})'] = value
    return results

def kpss_test(x: Union[pd.Series, np.ndarray], h0_type: str = 'c') -> pd.Series:
    """Perform the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    Null Hypothesis: The time series is stationary.
    Alternate Hypothesis: The time series is not stationary.

    Args:
        x: The time series to be checked for stationarity.
        h0_type: Indicates the null hypothesis of the KPSS test:
            'c': The data is stationary around a constant (default).
            'ct': The data is stationary around a trend.

    Returns:
        A pandas Series with the KPSS test's results.
    """
    indices = ['Test Statistic', 'p-value', '# of Lags']
    kpss_test_results = kpss(x, regression=h0_type, nlags='auto')
    results = pd.Series(kpss_test_results[0:3], index=indices)
    for key, value in kpss_test_results[3].items():
        results[f'Critical Value ({key})'] = value
    return results

def run_stationarity_tests(x: Union[pd.Series, np.ndarray], h0_type: str = 'c') -> pd.DataFrame:
    """Run both ADF and KPSS tests and return their results in a DataFrame.

    Args:
        x: The time series to be checked for stationarity.
        h0_type: The null hypothesis type for the KPSS test ('c' or 'ct').

    Returns:
        A DataFrame containing the results of both tests.
    """
    adf_results = adf_test(x)
    kpss_results = kpss_test(x, h0_type=h0_type)
    results_df = pd.DataFrame({'ADF': adf_results, 'KPSS': kpss_results})
    return results_df

def plot_autocorrelation(x: Union[pd.Series, np.ndarray], n_lags: int = 40, alpha: float = 0.05) -> plt.Figure:
    """Plot the ACF and PACF for a given time series.

    Args:
        x: The time series to be plotted.
        n_lags: The number of lags for the ACF/PACF plots.
        alpha: Significance level for the ACF/PACF plots.

    Returns:
        A matplotlib Figure containing the ACF/PACF plots.
    """
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))
    plot_acf(x, ax=ax[0], lags=n_lags, alpha=alpha)
    ax[0].set_title('Autocorrelation Function (ACF)')
    plot_pacf(x, ax=ax[1], lags=n_lags, alpha=alpha)
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    return fig

def plot_stl_decomposition(
    x: Union[pd.Series, np.ndarray],
    period: int,
    robust: bool = True,
    stl_result_2: Optional[DecomposeResult] = None,
    labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot the STL decomposition of a time series.

    :param x: The time series data.
    :param period: The seasonal period.
    :param robust: Whether to use a robust version of STL.
    :param stl_result_2: An optional second STL result to overlay on the plot.
    :param labels: Optional labels for the legend if a second result is provided.
    :return: The matplotlib Figure object.
    """
    stl_result = STL(x, period=period, robust=robust).fit()
    fig = stl_result.plot()

    if stl_result_2:
        axs = fig.get_axes()
        comps = ["trend", "seasonal", "resid"]
        for ax, comp in zip(axs[1:], comps):
            series = getattr(stl_result_2, comp)
            if comp == "resid":
                ax.plot(series, marker="o", linestyle="none")
            else:
                ax.plot(series)
                if comp == "trend" and labels:
                    ax.legend(labels, frameon=False)

    plt.tight_layout()
    return fig

def apply_hp_filter(x: Union[pd.Series, np.ndarray], lamb: float = 1600) -> Tuple[pd.Series, pd.Series]:
    """Apply the Hodrick-Prescott filter to a time series.

    Args:
        x: The time series to be filtered.
        lamb: The smoothing parameter for the HP filter.

    Returns:
        A tuple containing the cyclical and trend components of the time series.
    """
    cycle, trend = hpfilter(x, lamb=lamb)
    return pd.Series(cycle, index=x.index), pd.Series(trend, index=x.index)


def plot_forecast(
    forecast: Union[pd.Series, np.ndarray],
    train_data: pd.Series,
    test_data: pd.Series,
    conf_int: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Plot the forecast against the actuals.

    :param forecast: The forecasted values.
    :param train_data: The training data.
    :param test_data: The actual test data.
    :param conf_int: A DataFrame with 'lower' and 'upper' confidence interval bounds.
    :return: The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    forecast_series = pd.Series(forecast, index=test_data.index)

    ax.plot(train_data, label="Training")
    ax.plot(test_data, label="Actual")
    ax.plot(forecast_series, label="Forecast")

    if conf_int is not None:
        ax.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color="k",
            alpha=0.15,
        )

    ax.set_title("Forecast vs Actuals")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig
