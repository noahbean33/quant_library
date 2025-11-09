import numpy as np
import pandas as pd
import pytest

from src.statistics.tests import engle_granger_cointegration_test

@pytest.fixture
def cointegrated_series():
    """Generates two cointegrated time series for testing."""
    np.random.seed(42)
    # Create a random walk for the independent series
    x = np.random.normal(0, 1, size=1000).cumsum()
    # Create a dependent series that is a linear combination of x plus stationary noise
    y = 0.6 * x + np.random.normal(0, 1, size=1000)
    return pd.Series(y, name='Y'), pd.Series(x, name='X')

@pytest.fixture
def non_cointegrated_series():
    """Generates two non-cointegrated (independent random walks) time series."""
    np.random.seed(0)
    # Create two independent random walks
    x = np.random.normal(0, 1, size=1000).cumsum()
    y = np.random.normal(0, 1, size=1000).cumsum()
    return pd.Series(y, name='Y'), pd.Series(x, name='X')


class TestStatisticalTests:
    def test_cointegration_with_cointegrated_series(self, cointegrated_series):
        """Tests that the cointegration test correctly identifies cointegrated series."""
        series1, series2 = cointegrated_series
        
        # Perform the cointegration test
        adf_result = engle_granger_cointegration_test(series1, series2)
        
        # The p-value should be small (e.g., < 0.05) to reject the null hypothesis
        # of non-stationarity in the residuals, thus indicating cointegration.
        p_value = adf_result[1]
        assert p_value < 0.05

    def test_cointegration_with_non_cointegrated_series(self, non_cointegrated_series):
        """Tests that the cointegration test correctly identifies non-cointegrated series."""
        series1, series2 = non_cointegrated_series
        
        # Perform the cointegration test
        adf_result = engle_granger_cointegration_test(series1, series2)
        
        # The p-value should be large (e.g., > 0.05), failing to reject the null
        # hypothesis and thus indicating no cointegration.
        p_value = adf_result[1]
        assert p_value > 0.05
