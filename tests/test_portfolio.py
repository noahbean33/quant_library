import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio import build_portfolio

class TestPortfolio(unittest.TestCase):

    @patch('src.portfolio.portfolio.fetch_stock_data')
    def test_build_portfolio_success(self, mock_fetch_stock_data):
        """Test successful portfolio creation."""
        # Mock the fetch_stock_data function to return sample data
        mock_fetch_stock_data.side_effect = [
            pd.DataFrame({'Close': [100, 102]}, index=pd.to_datetime(['2023-01-01', '2023-01-02'])),
            pd.DataFrame({'Close': [200, 201]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        ]

        tickers = ['AAPL', 'MSFT']
        portfolio = build_portfolio(tickers)

        self.assertEqual(portfolio.tickers, tickers)
        self.assertEqual(portfolio.weights, {'AAPL': 0.5, 'MSFT': 0.5})
        self.assertFalse(portfolio.history.empty)
        self.assertEqual(list(portfolio.history.columns), tickers)
        self.assertEqual(len(portfolio.history), 2)

    @patch('src.portfolio.portfolio.fetch_stock_data')
    def test_build_portfolio_partial_failure(self, mock_fetch_stock_data):
        """Test portfolio creation with one invalid ticker."""
        # Mock fetch_stock_data to return data for one ticker and an empty df for another
        mock_fetch_stock_data.side_effect = [
            pd.DataFrame({'Close': [100, 102]}, index=pd.to_datetime(['2023-01-01', '2023-01-02'])),
            pd.DataFrame()  # Invalid ticker returns empty DataFrame
        ]

        tickers = ['AAPL', 'INVALID']
        portfolio = build_portfolio(tickers)

        # The portfolio should be created with only the valid ticker
        self.assertEqual(portfolio.tickers, ['AAPL'])
        self.assertEqual(portfolio.weights, {'AAPL': 1.0})
        self.assertFalse(portfolio.history.empty)
        self.assertEqual(list(portfolio.history.columns), ['AAPL'])

    @patch('src.portfolio.portfolio.fetch_stock_data')
    def test_build_portfolio_total_failure(self, mock_fetch_stock_data):
        """Test portfolio creation when all tickers are invalid."""
        # Mock fetch_stock_data to always return an empty DataFrame
        mock_fetch_stock_data.return_value = pd.DataFrame()

        tickers = ['INVALID1', 'INVALID2']
        portfolio = build_portfolio(tickers)

        self.assertEqual(portfolio.tickers, [])
        self.assertEqual(portfolio.weights, {})
        self.assertTrue(portfolio.history.empty)

if __name__ == '__main__':
    unittest.main()
