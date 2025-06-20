import pytest
import pandas as pd
import numpy as np
from src.financial_analysis_platform.data.data_preprocessing import (
    calculate_bollinger_bands,
    calculate_rsi,
)

@pytest.fixture
def sample_stock_data():
    """Fixture for creating sample stock data for technical analysis tests."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=50))
    data = pd.DataFrame({
        'Close': np.random.uniform(95, 105, size=50)
    }, index=dates)
    return data

def test_calculate_bollinger_bands(sample_stock_data):
    """Tests the Bollinger Bands calculation."""
    window = 20
    data = calculate_bollinger_bands(sample_stock_data, window=window)
    assert 'MA' in data.columns
    assert 'Upper' in data.columns
    assert 'Lower' in data.columns
    assert data['MA'].isnull().sum() == window - 1
    assert data['Upper'].isnull().sum() == window - 1
    assert data['Lower'].isnull().sum() == window - 1

def test_calculate_rsi(sample_stock_data):
    """Tests the RSI calculation."""
    window = 14
    data = calculate_rsi(sample_stock_data, window=window)
    assert 'RSI' in data.columns
    assert data['RSI'].isnull().sum() == window
    assert data['RSI'].min() >= 0
    assert data['RSI'].max() <= 100
