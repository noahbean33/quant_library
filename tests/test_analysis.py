import unittest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.analysis import calculate_ratios, calculate_graham_number

class TestAnalysis(unittest.TestCase):

    @patch('yfinance.Ticker')
    def test_calculate_ratios_success(self, mock_ticker):
        """Test successful calculation of financial ratios."""
        # Mock the .info attribute
        mock_info = {
            'trailingPE': 25.0,
            'priceToBook': 5.0,
            'priceToSalesTrailing12Months': 4.0,
            'returnOnEquity': 0.20,
            'returnOnAssets': 0.10,
            'profitMargins': 0.15,
            'debtToEquity': 50.0,
            'currentRatio': 2.0
        }
        mock_ticker.return_value.info = mock_info

        ratios = calculate_ratios('AAPL')

        self.assertEqual(ratios['pe_ratio'], 25.0)
        self.assertEqual(ratios['pb_ratio'], 5.0)
        self.assertAlmostEqual(ratios['earnings_yield'], 1/25.0)
        self.assertEqual(ratios['return_on_equity'], 0.20)
        self.assertEqual(ratios['debt_to_equity'], 50.0)

    @patch('yfinance.Ticker')
    def test_calculate_ratios_failure(self, mock_ticker):
        """Test ratio calculation with missing data."""
        mock_ticker.return_value.info = {}
        ratios = calculate_ratios('INVALID')
        self.assertEqual(ratios, {})

    @patch('yfinance.Ticker')
    def test_calculate_graham_number_success(self, mock_ticker):
        """Test successful calculation of the Graham Number."""
        mock_ticker.return_value.info = {'trailingEps': 10.0, 'bookValue': 15.0}

        graham_num = calculate_graham_number('AAPL')
        expected = (22.5 * 10.0 * 15.0) ** 0.5
        self.assertAlmostEqual(graham_num, expected)

    @patch('yfinance.Ticker')
    def test_calculate_graham_number_negative_eps(self, mock_ticker):
        """Test Graham Number calculation with negative EPS."""
        mock_ticker.return_value.info = {'trailingEps': -5.0, 'bookValue': 15.0}

        graham_num = calculate_graham_number('AAPL')
        self.assertEqual(graham_num, 0.0)

    @patch('yfinance.Ticker')
    def test_calculate_graham_number_missing_data(self, mock_ticker):
        """Test Graham Number calculation with missing data."""
        mock_ticker.return_value.info = {'trailingEps': 10.0} # Missing bookValue

        graham_num = calculate_graham_number('AAPL')
        self.assertIsNone(graham_num)

if __name__ == '__main__':
    unittest.main()
