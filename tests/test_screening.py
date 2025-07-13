import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.screening import screen_stocks

class TestScreening(unittest.TestCase):

    @patch('src.valueinvestpy.screening.screener.calculate_ratios')
    def test_screen_stocks(self, mock_calculate_ratios):
        """Test the stock screening functionality."""
        # Define the mock return values for calculate_ratios
        def ratios_side_effect(ticker):
            if ticker == 'GOOD_STOCK':
                return {'pe_ratio': 10, 'pb_ratio': 1, 'return_on_equity': 0.25}
            elif ticker == 'BAD_STOCK':
                return {'pe_ratio': 50, 'pb_ratio': 5, 'return_on_equity': 0.05}
            elif ticker == 'OK_STOCK':
                return {'pe_ratio': 15, 'pb_ratio': 2, 'return_on_equity': 0.15}
            else:
                return {}

        mock_calculate_ratios.side_effect = ratios_side_effect

        tickers = ['GOOD_STOCK', 'BAD_STOCK', 'OK_STOCK', 'MISSING_DATA']
        criteria = {
            'pe_ratio_max': 20,
            'pb_ratio_max': 3,
            'return_on_equity_min': 0.10
        }

        # Run the screener
        screened_stocks = screen_stocks(tickers, criteria)

        # Assertions
        self.assertIn('GOOD_STOCK', screened_stocks['ticker'].values)
        self.assertIn('OK_STOCK', screened_stocks['ticker'].values)
        self.assertNotIn('BAD_STOCK', screened_stocks['ticker'].values)
        self.assertNotIn('MISSING_DATA', screened_stocks['ticker'].values)
        self.assertEqual(len(screened_stocks), 2)

    @patch('src.valueinvestpy.screening.screener.calculate_ratios')
    def test_screen_stocks_no_matches(self, mock_calculate_ratios):
        """Test the screener when no stocks match the criteria."""
        mock_calculate_ratios.return_value = {'pe_ratio': 100, 'pb_ratio': 10}

        tickers = ['BAD_STOCK_1', 'BAD_STOCK_2']
        criteria = {'pe_ratio_max': 20}

        screened_stocks = screen_stocks(tickers, criteria)

        self.assertTrue(screened_stocks.empty)

if __name__ == '__main__':
    unittest.main()
