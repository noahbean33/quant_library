import unittest
import pandas as pd
import numpy as np
from src.analysis.returns import calculate_log_returns

class TestReturns(unittest.TestCase):

    def setUp(self):
        """Set up a sample price series for testing."""
        self.prices = pd.Series([100, 102, 99, 105, 104])

    def test_calculate_log_returns(self):
        """Test that log returns are calculated correctly."""
        log_returns = calculate_log_returns(self.prices)
        
        # Expected values: ln(102/100), ln(99/102), etc.
        expected_returns = pd.Series([
            np.nan,
            np.log(102/100),
            np.log(99/102),
            np.log(105/99),
            np.log(104/105)
        ])
        
        # Check that the first value is NaN
        self.assertTrue(pd.isna(log_returns.iloc[0]))
        
        # Check that the calculated values are close to the expected values
        pd.testing.assert_series_equal(
            log_returns, 
            expected_returns, 
            check_names=False
        )

if __name__ == '__main__':
    unittest.main()
