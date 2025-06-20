# tests/analysis/test_time_series_analysis.py

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.financial_analysis_platform.analysis import time_series_analysis

def test_adf_test(stationary_data, non_stationary_data):
    """Tests the Augmented Dickey-Fuller test function."""
    # A stationary series should have a p-value < 0.05
    result_stationary = time_series_analysis.adf_test(stationary_data)
    assert result_stationary['p-value'] < 0.05

    # A non-stationary series should have a p-value > 0.05
    result_non_stationary = time_series_analysis.adf_test(non_stationary_data)
    assert result_non_stationary['p-value'] > 0.05

def test_kpss_test(stationary_data, non_stationary_data):
    """Tests the KPSS test function."""
    # A stationary series should have a p-value > 0.05
    result_stationary = time_series_analysis.kpss_test(stationary_data)
    assert result_stationary['p-value'] > 0.05

    # A non-stationary series should have a p-value < 0.05
    result_non_stationary = time_series_analysis.kpss_test(non_stationary_data)
    assert result_non_stationary['p-value'] < 0.05

def test_run_stationarity_tests(stationary_data, non_stationary_data):
    """Tests the wrapper function for running both stationarity tests."""
    results = time_series_analysis.run_stationarity_tests(stationary_data)
    assert isinstance(results, pd.DataFrame)
    assert 'ADF' in results.columns
    assert 'KPSS' in results.columns

def test_plot_autocorrelation(stationary_data):
    """Tests the autocorrelation plotting function."""
    fig = time_series_analysis.plot_autocorrelation(stationary_data)
    assert fig is not None
    assert len(fig.axes) == 2  # ACF and PACF plots
    plt.close(fig)  # Close the figure to prevent it from displaying during tests

def test_plot_stl_decomposition(non_stationary_data):
    """Tests the STL decomposition plotting function."""
    # STL requires a series with an index with frequency
    data = non_stationary_data.copy()
    data.index = pd.date_range(start='2023-01-01', periods=len(data), freq='ME')
    fig = time_series_analysis.plot_stl_decomposition(data, period=12)
    assert fig is not None
    assert len(fig.axes) == 4  # Observed, Trend, Seasonal, Resid
    plt.close(fig)

def test_apply_hp_filter(non_stationary_data):
    """Tests the Hodrick-Prescott filter function."""
    cycle, trend = time_series_analysis.apply_hp_filter(non_stationary_data)
    assert isinstance(cycle, pd.Series)
    assert isinstance(trend, pd.Series)
    assert len(cycle) == len(non_stationary_data)
    assert len(trend) == len(non_stationary_data)

def test_plot_forecast(forecast_data):
    """Tests the forecast plotting function."""
    train, test, forecast, conf_int = forecast_data
    fig = time_series_analysis.plot_forecast(forecast, train, test, conf_int)
    assert fig is not None
    assert len(fig.axes) == 1
    plt.close(fig)
