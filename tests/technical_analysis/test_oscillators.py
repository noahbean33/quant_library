import pandas as pd
import numpy as np
import pytest
from src.technical_analysis.oscillators import rsi, macd

@pytest.fixture
def sample_data():
    """Fixture to provide a sample pandas Series for testing."""
    # Using a longer series for more stable RSI/MACD calculation
    return pd.Series(np.random.uniform(90, 110, size=50))

def test_rsi(sample_data):
    """Test the Relative Strength Index (RSI) calculation."""
    rsi_14 = rsi(sample_data, window=14)
    assert isinstance(rsi_14, pd.Series)
    assert rsi_14.isnull().sum() == 13 # RSI has a lookback period, creating NaNs
    # RSI values should be between 0 and 100
    assert rsi_14.dropna().between(0, 100).all()

def test_macd(sample_data):
    """Test the Moving Average Convergence Divergence (MACD) calculation."""
    macd_df = macd(sample_data)
    assert isinstance(macd_df, pd.DataFrame)
    assert list(macd_df.columns) == ['MACD', 'Signal', 'Histogram']
    assert not macd_df.isnull().values.any() # EMA-based MACD shouldn't have NaNs at the start
