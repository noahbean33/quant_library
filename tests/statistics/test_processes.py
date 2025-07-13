import numpy as np
import pytest

from valueinvestpy.statistics.processes import generate_ornstein_uhlenbeck_process


class TestStochasticProcesses:
    def test_generate_ornstein_uhlenbeck_process(self):
        """Tests the generation of an Ornstein-Uhlenbeck process."""
        n = 5000
        mu = 0.5
        
        # Generate the process
        process = generate_ornstein_uhlenbeck_process(n=n, mu=mu, dt=0.1)
        
        # 1. Check that the output has the correct length
        assert len(process) == n
        
        # 2. Check that the process is mean-reverting
        # For a long enough series, the mean should be close to mu.
        # We allow for a reasonable tolerance due to the stochastic nature.
        assert np.mean(process) == pytest.approx(mu, abs=0.1)
