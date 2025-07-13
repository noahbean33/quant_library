import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima_process import arma_generate_sample

from valueinvestpy.statistics.models import fit_arima


@pytest.fixture
def arma_1_1_series():
    """Generates a sample ARMA(1,1) time series for testing."""
    np.random.seed(42)
    # Define AR and MA parameters for an ARMA(1,1) process
    ar_params = np.array([.75])
    ma_params = np.array([.25])
    # Add the zero-lag convention for statsmodels
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    # Generate the time series
    series = arma_generate_sample(ar, ma, nsample=500)
    # Create a date range for the index
    dates = pd.date_range(start='2020-01-01', periods=500)
    return pd.Series(series, index=dates)


class TestTimeSeriesModels:
    def test_fit_arima_model(self, arma_1_1_series):
        """Tests that the fit_arima function correctly fits an ARMA(1,1) model."""
        # Fit an ARMA(1,1) model, which is an ARIMA(1,0,1)
        model_fit = fit_arima(arma_1_1_series, order=(1, 0, 1))
        
        # Get the estimated parameters
        # The order is [ar.L1, ma.L1, sigma2]
        estimated_params = model_fit.params
        
        # Check if the estimated AR and MA parameters are close to the true values
        # We allow for a reasonable tolerance due to estimation error.
        true_ar = 0.75
        true_ma = 0.25
        
        assert estimated_params['ar.L1'] == pytest.approx(true_ar, abs=0.1)
        assert estimated_params['ma.L1'] == pytest.approx(true_ma, abs=0.1)
