import unittest
import numpy as np
from src.valueinvestpy.analysis.pricing import black_scholes_analytical, black_scholes_monte_carlo

class TestPricing(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for option pricing tests."""
        self.S0 = 100.0
        self.E = 100.0
        self.T = 1.0
        self.rf = 0.05
        self.sigma = 0.2
        self.num_simulations = 50000  # Use a sufficient number for convergence
        self.tolerance = 0.15  # Price difference tolerance for Monte Carlo

    def test_black_scholes_analytical_call(self):
        """Test the analytical Black-Scholes formula for a call option."""
        # Pre-calculated value for these parameters
        expected_price = 10.45058
        calculated_price = black_scholes_analytical(
            self.S0, self.E, self.T, self.rf, self.sigma, option_type='call'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=5)

    def test_black_scholes_analytical_put(self):
        """Test the analytical Black-Scholes formula for a put option."""
        # Pre-calculated value for these parameters
        expected_price = 5.573526
        calculated_price = black_scholes_analytical(
            self.S0, self.E, self.T, self.rf, self.sigma, option_type='put'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=5)

    def test_black_scholes_monte_carlo_call(self):
        """Test the Monte Carlo pricer for a call option against the analytical price."""
        np.random.seed(42)  # for reproducibility
        analytical_price = black_scholes_analytical(
            self.S0, self.E, self.T, self.rf, self.sigma, option_type='call'
        )
        mc_price = black_scholes_monte_carlo(
            self.S0, self.E, self.T, self.rf, self.sigma, 
            num_simulations=self.num_simulations, option_type='call'
        )
        self.assertAlmostEqual(mc_price, analytical_price, delta=self.tolerance)

    def test_black_scholes_monte_carlo_put(self):
        """Test the Monte Carlo pricer for a put option against the analytical price."""
        np.random.seed(42)  # for reproducibility
        analytical_price = black_scholes_analytical(
            self.S0, self.E, self.T, self.rf, self.sigma, option_type='put'
        )
        mc_price = black_scholes_monte_carlo(
            self.S0, self.E, self.T, self.rf, self.sigma, 
            num_simulations=self.num_simulations, option_type='put'
        )
        self.assertAlmostEqual(mc_price, analytical_price, delta=self.tolerance)

    def test_invalid_option_type(self):
        """Test that an invalid option type raises a ValueError."""
        with self.assertRaises(ValueError):
            black_scholes_analytical(self.S0, self.E, self.T, self.rf, self.sigma, option_type='invalid')
        with self.assertRaises(ValueError):
            black_scholes_monte_carlo(self.S0, self.E, self.T, self.rf, self.sigma, option_type='invalid')

if __name__ == '__main__':
    unittest.main()
