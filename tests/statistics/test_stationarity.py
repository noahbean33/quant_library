import pandas as pd
import numpy as np
import pytest
from src.statistics.stationarity import adf_test, kpss_test

@pytest.fixture
def stationary_series():
    """Fixture for a stationary time series (white noise)."""
    return pd.Series(np.random.normal(0, 1, 1000))

@pytest.fixture
def non_stationary_series():
    """Fixture for a non-stationary time series (random walk)."""
    return pd.Series(np.random.normal(0, 1, 1000).cumsum())

def test_adf_test(stationary_series, non_stationary_series):
    """Test the Augmented Dickey-Fuller test."""
    # ADF null hypothesis is that the series is non-stationary
    # We expect to reject the null for the stationary series
    stationary_result = adf_test(stationary_series)
    assert stationary_result['Stationary'] == True
    
    # We expect to fail to reject the null for the non-stationary series
    non_stationary_result = adf_test(non_stationary_series)
    assert non_stationary_result['Stationary'] == False

def test_kpss_test(stationary_series, non_stationary_series):
    """Test the KPSS test."""
    # KPSS null hypothesis is that the series is stationary
    # We expect to fail to reject the null for the stationary series
    stationary_result = kpss_test(stationary_series)
    assert stationary_result['Stationary'] == True
    
    # We expect to reject the null for the non-stationary series
    non_stationary_result = kpss_test(non_stationary_series)
    assert non_stationary_result['Stationary'] == False
