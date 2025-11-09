import unittest
# A mock class to simulate the Portfolio object for testing
class MockPortfolio:
    def __init__(self, history, weights):
        self.history = history
        self.weights = weights
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest import run_backtest

class TestBacktest(unittest.TestCase):

    def test_run_backtest(self):
        """Test the backtesting engine with a sample portfolio."""
        # Create a sample history DataFrame
        dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2021-01-01'])
        history = pd.DataFrame({
            'AAPL': [100, 102, 101, 120],
            'MSFT': [200, 201, 202, 240]
        }, index=dates)
        weights = {'AAPL': 0.5, 'MSFT': 0.5}

        # Create a mock portfolio object
        mock_portfolio = MockPortfolio(history, weights)

        # Run the backtest
        performance = run_backtest(mock_portfolio)

        # Expected values (manual calculation for verification)
        # Portfolio value: [150, 151.5, 151.5, 180]
        # Returns: [nan, 0.01, 0.0, 0.1881188]
        # Total Return: (180/150) - 1 = 0.2
        # Annualized Return: (1 + 0.2) ^ (252/len(days)) - 1. Simplified here.
        # Sharpe Ratio: Needs risk-free rate, but let's check if it's a float.

        # The expected total return is calculated from the cumulative product of daily returns
        # not just the start and end portfolio values, which is more accurate.
        self.assertAlmostEqual(performance['total_return'], 0.2000658, places=4)
        self.assertTrue(isinstance(performance['annualized_return'], float))
        self.assertTrue(isinstance(performance['sharpe_ratio'], float))
        self.assertTrue(isinstance(performance['max_drawdown'], float))
        # The cumulative_returns series will have one less entry than the history because of pct_change()
        self.assertEqual(len(performance['cumulative_returns']), 3)

    def test_run_backtest_flat_return(self):
        """Test backtesting with a portfolio that has zero volatility."""
        history = pd.DataFrame({'FLAT': [100, 100, 100]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        weights = {'FLAT': 1.0}
        mock_portfolio = MockPortfolio(history, weights)

        performance = run_backtest(mock_portfolio)

        self.assertAlmostEqual(performance['total_return'], 0.0)
        self.assertAlmostEqual(performance['annualized_volatility'], 0.0)
        # Sharpe ratio should be 0 if volatility is 0
        self.assertAlmostEqual(performance['sharpe_ratio'], 0.0)

if __name__ == '__main__':
    unittest.main()
