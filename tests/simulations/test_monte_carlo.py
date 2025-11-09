import pandas as pd
import numpy as np
import pytest
from src.simulations.monte_carlo import geometric_brownian_motion

@pytest.fixture
def gbm_params():
    """Fixture for GBM parameters."""
    return {'S0': 100, 'mu': 0.05, 'sigma': 0.2, 'T': 1, 'n_steps': 252, 'n_sims': 10}

def test_geometric_brownian_motion(gbm_params):
    """Test the Geometric Brownian Motion simulation."""
    sims = geometric_brownian_motion(**gbm_params)
    
    # Check output type and shape
    assert isinstance(sims, pd.DataFrame)
    assert sims.shape == (gbm_params['n_steps'] + 1, gbm_params['n_sims'])
    
    # The first row of prices should be the initial price S0
    assert np.allclose(sims.iloc[0], gbm_params['S0'])
    
    # All simulated prices should be positive
    assert (sims.values > 0).all()
