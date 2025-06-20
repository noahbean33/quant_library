import matplotlib
matplotlib.use('Agg')
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def stationary_data():
    """Generate a stationary time series."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100), name='stationary')

@pytest.fixture
def non_stationary_data():
    """Generate a non-stationary time series (random walk)."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100).cumsum(), name='non_stationary')

@pytest.fixture
def forecast_data():
    """Provides sample data for forecast plotting tests."""
    np.random.seed(42)
    train = pd.Series(np.random.randn(80).cumsum() + 100, name='train')
    test = pd.Series(np.random.randn(20).cumsum() + train.iloc[-1], index=pd.RangeIndex(start=80, stop=100), name='test')
    forecast = pd.Series(test.values + np.random.randn(20) * 2, index=test.index, name='forecast')
    conf_int = pd.DataFrame({
        'lower': forecast - 5,
        'upper': forecast + 5
    }, index=test.index)
    return train, test, forecast, conf_int
