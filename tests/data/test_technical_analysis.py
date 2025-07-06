import pytest
import pandas as pd
import numpy as np
from src.financial_analysis_platform.analysis.technical_analysis import (
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_stochastic_oscillator,
)

@pytest.fixture
def sample_stock_data():
    """Fixture for creating sample stock data for technical analysis tests."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=50))
    close = np.random.uniform(95, 105, size=50)
    high = close + np.random.uniform(0, 2, size=50)
    low = close - np.random.uniform(0, 2, size=50)
    data = pd.DataFrame({
        'High': high,
        'Low': low,
        'Close': close
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

def test_calculate_macd(sample_stock_data):
    """Tests the MACD calculation."""
    data = calculate_macd(sample_stock_data)
    assert 'MACD' in data.columns
    assert 'Signal_Line' in data.columns
    assert 'MACD_Histogram' in data.columns
    # The first value of MACD, Signal Line, and Histogram should be 0.0 with adjust=False
    assert data['MACD'].iloc[0] == 0.0
    assert data['Signal_Line'].iloc[0] == 0.0
    assert data['MACD_Histogram'].iloc[0] == 0.0

def test_calculate_atr(sample_stock_data):
    """Tests the ATR calculation."""
    data = calculate_atr(sample_stock_data)
    assert 'ATR' in data.columns
    # The first value of ATR should be NaN because of the shift(1)
    assert pd.isna(data['ATR'].iloc[0])
    assert (data['ATR'].dropna() > 0).all()

def test_calculate_stochastic_oscillator(sample_stock_data):
    """Tests the Stochastic Oscillator calculation."""
    data = calculate_stochastic_oscillator(sample_stock_data)
    assert '%K' in data.columns
    assert '%D' in data.columns
    assert data['%K'].notna().any()
    assert data['%D'].notna().any()
