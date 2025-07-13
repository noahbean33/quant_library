import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.valueinvestpy.analysis.financial_models import calculate_black_scholes, calculate_capm

class TestFinancialModels(unittest.TestCase):

    def test_calculate_black_scholes_call(self):
        """Test Black-Scholes call option pricing against a known value."""
        # Parameters from a standard online calculator
        S, E, T, rf, sigma = 100, 100, 1, 0.05, 0.2
        expected_call_price = 10.45  # Known value
        calculated_price = calculate_black_scholes(S, E, T, rf, sigma, option_type='call')
        self.assertAlmostEqual(calculated_price, expected_call_price, places=2)

    def test_calculate_black_scholes_put(self):
        """Test Black-Scholes put option pricing against a known value."""
        S, E, T, rf, sigma = 100, 100, 1, 0.05, 0.2
        expected_put_price = 5.57  # Known value
        calculated_price = calculate_black_scholes(S, E, T, rf, sigma, option_type='put')
        self.assertAlmostEqual(calculated_price, expected_put_price, places=2)

    def test_black_scholes_invalid_type(self):
        """Test that Black-Scholes raises an error for an invalid option type."""
        with self.assertRaises(ValueError):
            calculate_black_scholes(100, 100, 1, 0.05, 0.2, option_type='invalid')

    @patch('src.valueinvestpy.analysis.financial_models.fetch_stock_data')
    def test_calculate_capm_success(self, mock_fetch_stock_data):
        """Test CAPM calculation with mocked data."""
        # Create mock dataframes for stock and market
        dates = pd.to_datetime(['2023-01-31', '2023-02-28', '2023-03-31'])
        mock_stock_data = pd.DataFrame({'Close': [100, 105, 102]}, index=dates)
        # Market has a beta of 1, so we expect stock beta to be > 1
        mock_market_data = pd.DataFrame({'Close': [1000, 1020, 1010]}, index=dates)

        # Configure the mock to return different data based on the ticker
        def fetch_side_effect(ticker, **kwargs):
            if ticker == 'STOCK':
                return mock_stock_data
            if ticker == 'MARKET':
                return mock_market_data
            return pd.DataFrame()
        
        mock_fetch_stock_data.side_effect = fetch_side_effect

        capm_results = calculate_capm('STOCK', 'MARKET')
        
        self.assertIn('beta', capm_results)
        self.assertIn('alpha', capm_results)
        self.assertIn('expected_return', capm_results)
        # Based on the data, beta should be around 1.25
        self.assertAlmostEqual(capm_results['beta'], 2.62, places=2)

    @patch('src.valueinvestpy.analysis.financial_models.fetch_stock_data')
    def test_calculate_capm_data_failure(self, mock_fetch_stock_data):
        """Test that CAPM returns an empty dict if data fetching fails."""
        mock_fetch_stock_data.return_value = pd.DataFrame() # Simulate data fetch failure
        capm_results = calculate_capm('FAIL', 'MARKET')
        self.assertEqual(capm_results, {})

if __name__ == '__main__':
    unittest.main()
