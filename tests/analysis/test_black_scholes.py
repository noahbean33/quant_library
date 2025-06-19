# tests/analysis/test_black_scholes.py

import pytest
from src.financial_analysis_platform.analysis import black_scholes

@pytest.fixture
def option_params():
    """Provides a standard set of option parameters for testing."""
    return {
        'S': 100,       # Underlying stock price
        'E': 100,       # Strike price
        'T': 1,         # Time to expiration in years
        'rf': 0.05,     # Risk-free interest rate
        'sigma': 0.2,   # Volatility
        'iterations': 1000000 # Monte Carlo iterations
    }

def test_call_option_price(option_params):
    """Tests the analytical Black-Scholes call option price."""
    price = black_scholes.call_option_price(
        S=option_params['S'],
        E=option_params['E'],
        T=option_params['T'],
        rf=option_params['rf'],
        sigma=option_params['sigma']
    )
    # Known value for these parameters is ~10.45
    assert price == pytest.approx(10.45, abs=0.01)

def test_put_option_price(option_params):
    """Tests the analytical Black-Scholes put option price."""
    price = black_scholes.put_option_price(
        S=option_params['S'],
        E=option_params['E'],
        T=option_params['T'],
        rf=option_params['rf'],
        sigma=option_params['sigma']
    )
    # Known value for these parameters is ~5.57
    assert price == pytest.approx(5.57, abs=0.01)

def test_monte_carlo_simulation(option_params):
    """Tests the Monte Carlo simulation against the analytical solution."""
    # Analytical prices
    analytical_call = black_scholes.call_option_price(**{k: v for k, v in option_params.items() if k != 'iterations'})
    analytical_put = black_scholes.put_option_price(**{k: v for k, v in option_params.items() if k != 'iterations'})

    # Monte Carlo simulation - adjust params for OptionPricing class
    mc_params = option_params.copy()
    mc_params['S0'] = mc_params.pop('S')
    mc_pricer = black_scholes.OptionPricing(**mc_params)
    mc_call = mc_pricer.call_option_simulation()
    mc_put = mc_pricer.put_option_simulation()

    # The MC simulation should be close to the analytical price
    assert mc_call == pytest.approx(analytical_call, abs=0.1)
    assert mc_put == pytest.approx(analytical_put, abs=0.1)
