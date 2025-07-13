import unittest
import pandas as pd
import numpy as np
from src.valueinvestpy.analysis.risk import calculate_var, calculate_var_monte_carlo

class TestRisk(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for VaR tests."""
        # Generate a sample series of log returns
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(loc=0.0005, scale=0.02, size=1000))
        self.investment = 1_000_000
        self.confidence_level = 0.99
        self.horizon_days = 1
        self.num_simulations = 20000  # A reasonable number for testing
        self.tolerance = 0.05  # 5% tolerance for comparing MC to analytical

    def test_calculate_var_analytical(self):
        """Test the analytical VaR calculation against a pre-calculated value."""
        # Manually calculate the expected VaR for the generated returns
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        z_score = 2.3263  # Corresponds to 99% confidence
        expected_var = self.investment * (mu * self.horizon_days - sigma * np.sqrt(self.horizon_days) * (-z_score))
        
        calculated_var = calculate_var(
            self.investment, self.returns, self.confidence_level, self.horizon_days
        )
        # We use a wider delta because norm.ppf is more precise than a hardcoded z-score
        self.assertAlmostEqual(calculated_var, expected_var, delta=50.0)

    def test_calculate_var_monte_carlo(self):
        """Test that the Monte Carlo VaR is close to the analytical VaR."""
        np.random.seed(42) # for reproducibility
        analytical_var = calculate_var(
            self.investment, self.returns, self.confidence_level, self.horizon_days
        )
        mc_var = calculate_var_monte_carlo(
            self.investment, self.returns, self.confidence_level, 
            self.horizon_days, self.num_simulations
        )
        
        # Check that the Monte Carlo result is within the tolerance of the analytical result
        self.assertAlmostEqual(mc_var, analytical_var, delta=analytical_var * self.tolerance)

    def test_invalid_returns_type(self):
        """Test that a TypeError is raised for invalid 'returns' input."""
        with self.assertRaises(TypeError):
            calculate_var(self.investment, [0.01, 0.02], self.confidence_level)
        
        with self.assertRaises(TypeError):
            calculate_var_monte_carlo(self.investment, [0.01, 0.02], self.confidence_level)

if __name__ == '__main__':
    unittest.main()
