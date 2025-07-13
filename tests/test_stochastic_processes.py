import unittest
import pandas as pd
from unittest.mock import patch
from src.valueinvestpy.simulation.stochastic_processes import (
    simulate_gbm, 
    simulate_ornstein_uhlenbeck, 
    simulate_wiener_process, 
    simulate_random_walk,
    simulate_vasicek_model
)

class TestStochasticProcesses(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for the tests."""
        # GBM parameters
        self.S0 = 100
        self.mu = 0.05
        self.sigma = 0.2
        self.T = 1
        self.N = 252
        self.num_simulations = 10

        # Ornstein-Uhlenbeck parameters
        self.x0 = 0.03
        self.theta = 0.5
        self.ou_mu = 0.05
        self.ou_sigma = 0.02

    def test_simulate_gbm_output_shape(self):
        """Test that the output DataFrame has the correct shape."""
        sim_df = simulate_gbm(
            S0=self.S0, 
            mu=self.mu, 
            sigma=self.sigma, 
            T=self.T, 
            N=self.N, 
            num_simulations=self.num_simulations
        )
        expected_rows = self.N + 1
        expected_cols = self.num_simulations
        self.assertEqual(sim_df.shape, (expected_rows, expected_cols))

    def test_simulate_gbm_initial_price(self):
        """Test that the initial price in all simulations is S0."""
        sim_df = simulate_gbm(S0=self.S0, mu=self.mu, sigma=self.sigma, num_simulations=self.num_simulations)
        initial_prices = sim_df.iloc[0]
        self.assertTrue((initial_prices == self.S0).all())

    def test_simulate_gbm_data_type(self):
        """Test that the DataFrame contains float values."""
        sim_df = simulate_gbm(S0=self.S0, mu=self.mu, sigma=self.sigma)
        self.assertTrue(all(sim_df.dtypes == 'float64'))

    def test_simulate_gbm_single_simulation(self):
        """Test that the function runs correctly for a single simulation."""
        sim_df = simulate_gbm(
            S0=self.S0, 
            mu=self.mu, 
            sigma=self.sigma, 
            num_simulations=1
        )
        self.assertIsInstance(sim_df, pd.DataFrame)
        self.assertEqual(sim_df.shape[1], 1)

    def test_simulate_ou_output_shape(self):
        """Test that the OU output DataFrame has the correct shape."""
        sim_df = simulate_ornstein_uhlenbeck(
            x0=self.x0,
            theta=self.theta,
            mu=self.ou_mu,
            sigma=self.ou_sigma,
            T=self.T,
            N=self.N,
            num_simulations=self.num_simulations
        )
        expected_rows = self.N + 1
        expected_cols = self.num_simulations
        self.assertEqual(sim_df.shape, (expected_rows, expected_cols))

    def test_simulate_ou_initial_value(self):
        """Test that the initial value in all OU simulations is x0."""
        sim_df = simulate_ornstein_uhlenbeck(
            x0=self.x0, theta=self.theta, mu=self.ou_mu, sigma=self.ou_sigma, num_simulations=self.num_simulations
        )
        initial_values = sim_df.iloc[0]
        self.assertTrue((initial_values == self.x0).all())

    def test_simulate_ou_single_simulation(self):
        """Test that the OU function runs correctly for a single simulation."""
        sim_df = simulate_ornstein_uhlenbeck(
            x0=self.x0,
            theta=self.theta,
            mu=self.ou_mu,
            sigma=self.ou_sigma,
            num_simulations=1
        )
        self.assertIsInstance(sim_df, pd.DataFrame)
        self.assertEqual(sim_df.shape[1], 1)


    def test_simulate_wiener_process_output_shape(self):
        """Test that the Wiener process output DataFrame has the correct shape."""
        sim_df = simulate_wiener_process(
            T=self.T, N=self.N, num_simulations=self.num_simulations
        )
        expected_rows = self.N + 1
        expected_cols = self.num_simulations
        self.assertEqual(sim_df.shape, (expected_rows, expected_cols))

    def test_simulate_wiener_process_initial_value(self):
        """Test that the initial value in all Wiener process simulations is 0."""
        sim_df = simulate_wiener_process(num_simulations=self.num_simulations)
        initial_values = sim_df.iloc[0]
        self.assertTrue((initial_values == 0).all())

    def test_simulate_wiener_process_single_simulation(self):
        """Test that the Wiener process function runs correctly for a single simulation."""
        sim_df = simulate_wiener_process(num_simulations=1)
        self.assertIsInstance(sim_df, pd.DataFrame)
        self.assertEqual(sim_df.shape[1], 1)


    def test_simulate_random_walk_output_shape(self):
        """Test that the random walk output DataFrame has the correct shape."""
        num_steps = 100
        sim_df = simulate_random_walk(
            num_steps=num_steps, num_simulations=self.num_simulations
        )
        expected_rows = num_steps + 1
        expected_cols = self.num_simulations
        self.assertEqual(sim_df.shape, (expected_rows, expected_cols))

    def test_simulate_random_walk_initial_value(self):
        """Test that the initial value in all random walk simulations is 0."""
        sim_df = simulate_random_walk(num_simulations=self.num_simulations)
        initial_values = sim_df.iloc[0]
        self.assertTrue((initial_values == 0).all())

    def test_simulate_random_walk_step_values(self):
        """Test that the steps in the random walk are either +1 or -1."""
        sim_df = simulate_random_walk(num_steps=100, num_simulations=10)
        # Calculate the differences between consecutive steps
        steps = sim_df.diff().dropna()
        # Check if all step values are either 1 or -1
        self.assertTrue(steps.isin([1, -1]).all().all())

    def test_simulate_random_walk_single_simulation(self):
        """Test that the random walk function runs correctly for a single simulation."""
        sim_df = simulate_random_walk(num_simulations=1)
        self.assertIsInstance(sim_df, pd.DataFrame)
        self.assertEqual(sim_df.shape[1], 1)


    def test_simulate_vasicek_model_output_shape(self):
        """Test that the Vasicek model output DataFrame has the correct shape."""
        sim_df = simulate_vasicek_model(
            r0=0.03, kappa=0.15, theta=0.05, sigma=0.01,
            T=self.T, N=self.N, num_simulations=self.num_simulations
        )
        expected_rows = self.N + 1
        expected_cols = self.num_simulations
        self.assertEqual(sim_df.shape, (expected_rows, expected_cols))

    @patch('src.valueinvestpy.simulation.stochastic_processes.simulate_ornstein_uhlenbeck')
    def test_simulate_vasicek_model_calls_ornstein_uhlenbeck(self, mock_ou):
        """Test that the Vasicek model correctly calls the Ornstein-Uhlenbeck simulation."""
        r0, kappa, theta, sigma = 0.03, 0.15, 0.05, 0.01
        
        simulate_vasicek_model(
            r0=r0, kappa=kappa, theta=theta, sigma=sigma,
            T=self.T, N=self.N, num_simulations=self.num_simulations
        )
        
        self.assertTrue(mock_ou.called)
        mock_ou.assert_called_once_with(
            x0=r0,
            theta=kappa,
            mu=theta,
            sigma=sigma,
            T=self.T,
            N=self.N,
            num_simulations=self.num_simulations
        )


if __name__ == '__main__':
    unittest.main()
