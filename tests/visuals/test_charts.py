import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from src.visuals.charts import plot_candlestick, plot_correlogram

@pytest.fixture
def ohlcv_data():
    """Fixture for sample OHLCV data."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    data = pd.DataFrame({
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': lambda x: x['Open'] + np.random.uniform(0, 2, 100),
        'Low': lambda x: x['Open'] - np.random.uniform(0, 2, 100),
        'Close': lambda x: (x['Open'] + x['High'] + x['Low']) / 3,
        'Volume': np.random.randint(10000, 50000, 100)
    }, index=dates)
    return data

@pytest.fixture
def time_series_data():
    """Fixture for a sample time series."""
    return pd.Series(np.random.randn(100), name='Test_Series')

@patch('src.visuals.charts.mpf.plot')
def test_plot_candlestick(mock_plot, ohlcv_data):
    """Test the candlestick plotting function by checking if it calls the underlying library correctly."""
    plot_candlestick(ohlcv_data, title='Test Chart', mav=(5, 10), show_volume=True)

    # Assert that the plot function was called once
    mock_plot.assert_called_once()

    # Assert that it was called with the correct keyword arguments
    args, kwargs = mock_plot.call_args
    assert kwargs['type'] == 'candle'
    assert kwargs['title'] == 'Test Chart'
    assert kwargs['mav'] == (5, 10)
    assert kwargs['volume'] is True

@patch('src.visuals.charts.plot_acf')
def test_plot_correlogram(mock_plot_acf, time_series_data):
    """Test the correlogram plotting function by checking if it calls the underlying library correctly."""
    plot_correlogram(time_series_data, lags=15, title='Test Correlogram')

    # Assert that the plot_acf function was called once
    mock_plot_acf.assert_called_once()

    # Assert that it was called with the correct arguments
    args, kwargs = mock_plot_acf.call_args
    pd.testing.assert_series_equal(args[0], time_series_data)
    assert kwargs['lags'] == 15
    assert kwargs['title'] == 'Test Correlogram'
