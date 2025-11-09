import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_fetch import fetch_stock_data, fetch_financials

class TestDataFetch(unittest.TestCase):

    @patch('yfinance.Ticker')
    def test_fetch_stock_data_success(self, mock_ticker):
        """Test successful fetching of stock data."""
        # Mock the yfinance Ticker object and its history method
        mock_hist = pd.DataFrame({'Close': [150, 152, 151]})
        mock_ticker.return_value.history.return_value = mock_hist

        # Call the function
        data = fetch_stock_data('AAPL')

        # Assertions
        self.assertFalse(data.empty)
        self.assertEqual(len(data), 3)
        mock_ticker.assert_called_with('AAPL')

    @patch('yfinance.Ticker')
    def test_fetch_stock_data_failure(self, mock_ticker):
        """Test fetching stock data for an invalid ticker."""
        # Mock the yfinance Ticker object to return an empty DataFrame
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        # Call the function
        data = fetch_stock_data('INVALID')

        # Assertions
        self.assertTrue(data.empty)
        mock_ticker.assert_called_with('INVALID')

    @patch('yfinance.Ticker')
    def test_fetch_financials_success(self, mock_ticker):
        """Test successful fetching of financial statements."""
        # Mock the financial statement properties
        mock_ticker.return_value.income_stmt = pd.DataFrame({'Revenue': [1000]})
        mock_ticker.return_value.balance_sheet = pd.DataFrame({'Assets': [2000]})
        mock_ticker.return_value.cashflow = pd.DataFrame({'Operating Cash Flow': [500]})

        # Call the function
        financials = fetch_financials('AAPL')

        # Assertions
        self.assertIn('income_statement', financials)
        self.assertIn('balance_sheet', financials)
        self.assertIn('cash_flow', financials)
        self.assertFalse(financials['income_statement'].empty)

    @patch('yfinance.Ticker')
    def test_fetch_financials_failure(self, mock_ticker):
        """Test fetching financials for a ticker with no data."""
        # Mock the financial statement properties to be empty
        mock_ticker.return_value.income_stmt = pd.DataFrame()
        mock_ticker.return_value.balance_sheet = pd.DataFrame()
        mock_ticker.return_value.cashflow = pd.DataFrame()

        # Call the function
        financials = fetch_financials('NODATA')

        # Assertions
        self.assertEqual(financials, {})

if __name__ == '__main__':
    unittest.main()
