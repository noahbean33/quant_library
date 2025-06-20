import pandas as pd
import numpy as np
import pytest
from src.financial_analysis_platform.analysis.monte_carlo import monte_carlo_simulation

def test_monte_carlo_simulation(non_stationary_data):
    """Tests the monte_carlo_simulation function."""
    # Prepare a DataFrame with a 'Close' column as the function expects
    # Ensure data is positive to represent stock prices, preventing log(negative) errors.
    positive_data = non_stationary_data + abs(non_stationary_data.min()) + 1
    df = pd.DataFrame({'Close': positive_data})
    n_simulations = 10
    n_days = 30

    # Run the simulation
    simulation_df = monte_carlo_simulation(df, n_simulations, n_days)

    # Assertions
    assert isinstance(simulation_df, pd.DataFrame)
    assert simulation_df.shape == (n_days, n_simulations)

    # The first price in each simulation path should be the last price of the input data
    last_known_price = df['Close'].iloc[-1]
    assert np.allclose(simulation_df.iloc[0], last_known_price)
