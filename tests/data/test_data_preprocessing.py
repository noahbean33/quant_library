import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, ANY

from src.financial_analysis_platform.data.data_preprocessing import (
    adjust_returns_for_inflation,
    calculate_simple_returns,
    calculate_moving_averages,
    calculate_realized_volatility,
    generate_ma_crossover_signals
)


@patch('src.financial_analysis_platform.data.data_preprocessing.nasdaqdatalink.get')
@patch('src.financial_analysis_platform.data.data_preprocessing.yf.download')
def test_adjust_returns_for_inflation(mock_yf_download, mock_nasdaq_get):
    """
    Tests the adjust_returns_for_inflation function with mocked API calls.
    """
    # 1. Mock yfinance download
    dates = pd.to_datetime(['2023-01-31', '2023-02-28', '2023-03-31'])
    prices = [150, 152, 155]
    mock_stock_df = pd.DataFrame({'Adj Close': prices}, index=dates)
    mock_yf_download.return_value = mock_stock_df

    # 2. Mock Nasdaq Data Link get
    cpi_dates = pd.to_datetime(['2022-12-31', '2023-01-31', '2023-02-28', '2023-03-31'])
    cpi_values = [100, 101, 102, 103]
    mock_cpi_df = pd.DataFrame({'Value': cpi_values}, index=cpi_dates)
    mock_nasdaq_get.return_value = mock_cpi_df

    # 3. Call the function
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    result_df = adjust_returns_for_inflation(ticker, start_date, end_date)

    # 4. Assertions
    assert not result_df.empty
    assert 'real_rtn' in result_df.columns
    assert result_df.shape[0] == 2  # Expecting 2 rows after dropping NaN

    # 5. Verify calculations
    # Expected simple return for Feb: (152-150)/150
    # Expected inflation for Feb: (102-101)/101
    # Expected real return for Feb is calculated based on the above
    assert np.isclose(result_df['simple_rtn'].iloc[0], (152-150)/150)
    assert np.isclose(result_df['inflation_rate'].iloc[0], (102-101)/101)
    assert np.isclose(result_df['real_rtn'].iloc[0], ((1 + (152-150)/150) / (1 + (102-101)/101)) - 1)


@pytest.fixture
def sample_stock_data():
    """Fixture to create a sample DataFrame for testing."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    return pd.DataFrame({'Close': prices}, index=dates)


def test_calculate_simple_returns(sample_stock_data):
    """Tests the calculate_simple_returns function."""
    df = calculate_simple_returns(sample_stock_data)
    assert 'simple_rtn' in df.columns
    assert pd.isna(df['simple_rtn'].iloc[0])
    expected_return = (102 - 100) / 100
    assert np.isclose(df['simple_rtn'].iloc[1], expected_return)


def test_calculate_moving_averages(sample_stock_data):
    """Tests the calculate_moving_averages function."""
    windows = [3, 5]
    df = calculate_moving_averages(sample_stock_data, windows)
    assert 'MA_3' in df.columns
    assert 'MA_5' in df.columns
    assert pd.isna(df['MA_3'].iloc[1])
    expected_ma_3 = (100 + 102 + 101) / 3
    assert np.isclose(df['MA_3'].iloc[2], expected_ma_3)
    expected_ma_5 = (100 + 102 + 101 + 103 + 105) / 5
    assert np.isclose(df['MA_5'].iloc[4], expected_ma_5)


def test_calculate_realized_volatility():
    """Tests the calculate_realized_volatility function."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=40, freq='D'))
    prices = 100 + np.random.randn(40).cumsum()
    df = pd.DataFrame({'Close': prices}, index=dates)
    df_rv = calculate_realized_volatility(df)
    assert isinstance(df_rv, pd.DataFrame)
    assert 'rv' in df_rv.columns
    assert df_rv.shape[0] > 0 # Should have at least one month of data


def test_generate_ma_crossover_signals():
    """Tests the generate_ma_crossover_signals function."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    # Prices designed to create a buy signal then a sell signal
    prices = [100, 101, 102, 103, 104, 100, 98, 96, 95, 94]
    df = pd.DataFrame({'Close': prices}, index=dates)
    short_window, long_window = 3, 5
    df = calculate_moving_averages(df, [short_window, long_window])
    df = generate_ma_crossover_signals(df, short_window, long_window)

    assert 'signal' in df.columns
    assert 'position' in df.columns
    # A buy signal (position=1) should occur at index 4
    # A sell signal (position=-1) should occur at index 6
    assert df['position'].iloc[4] == 1.0
    assert df['position'].iloc[6] == -1.0

