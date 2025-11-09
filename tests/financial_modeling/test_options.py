import numpy as np
import pytest
from src.financial_modeling.options import black_scholes

@pytest.fixture
def option_params():
    """Fixture for Black-Scholes parameters."""
    return {'S': 100, 'K': 100, 'T': 1, 'r': 0.05, 'sigma': 0.2}

def test_black_scholes_call(option_params):
    """Test the Black-Scholes call option pricing."""
    # Known value from online calculator for these parameters
    expected_price = 10.45
    price = black_scholes(**option_params, option_type='call')
    assert np.isclose(price, expected_price, atol=0.01)

def test_black_scholes_put(option_params):
    """Test the Black-Scholes put option pricing."""
    # Known value from online calculator for these parameters
    expected_price = 5.57
    price = black_scholes(**option_params, option_type='put')
    assert np.isclose(price, expected_price, atol=0.01)

def test_black_scholes_invalid_type(option_params):
    """Test that an invalid option type raises a ValueError."""
    with pytest.raises(ValueError):
        black_scholes(**option_params, option_type='invalid')
