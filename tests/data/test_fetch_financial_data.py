# tests/data/test_fetch_financial_data.py

import pandas as pd
from src.financial_analysis_platform.data.fetch_financial_data import get_data_yahoo_finance, get_ticker_info

def test_get_data_yahoo_finance():
    """Tests the get_data_yahoo_finance function."""
    # Define test parameters
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-31"

    # Fetch data
    df = get_data_yahoo_finance(ticker, start_date, end_date)

    # Assert that the returned object is a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Assert that the DataFrame is not empty
    assert not df.empty

    # Assert that the DataFrame has the expected columns
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected_columns:
        assert col in df.columns


def test_get_ticker_info():
    """Tests the get_ticker_info function."""
    # Define test parameters
    ticker = "AAPL"

    # Fetch info
    info = get_ticker_info(ticker)

    # Assert that the returned object is a dict
    assert isinstance(info, dict)

    # Assert that the dict is not empty
    assert info

    # Assert that the dict has the expected keys
    assert "symbol" in info
    assert info["symbol"] == ticker
    assert "longName" in info
