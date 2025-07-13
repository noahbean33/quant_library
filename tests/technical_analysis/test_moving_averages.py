import pandas as pd
import numpy as np
import pytest
from valueinvestpy.technical_analysis.moving_averages import sma, ema

@pytest.fixture
def sample_data():
    """Fixture to provide a sample pandas Series for testing."""
    return pd.Series([10, 12, 15, 14, 13, 16, 18, 20, 19, 22])

def test_sma(sample_data):
    """Test the Simple Moving Average (SMA) calculation."""
    sma_5 = sma(sample_data, window=5)
    assert isinstance(sma_5, pd.Series)
    assert sma_5.isnull().sum() == 4  # First 4 values should be NaN
    # Manually calculate the first valid SMA
    expected_sma = (10 + 12 + 15 + 14 + 13) / 5
    assert np.isclose(sma_5.iloc[4], expected_sma)

def test_ema(sample_data):
    """Test the Exponential Moving Average (EMA) calculation."""
    ema_5 = ema(sample_data, window=5)
    assert isinstance(ema_5, pd.Series)
    assert not ema_5.isnull().any() # EMA should not produce NaNs at the start with pandas implementation
    # The first EMA value is just the first data point
    assert np.isclose(ema_5.iloc[0], sample_data.iloc[0])
