import pandas as pd
import numpy as np
import pytest
from valueinvestpy.portfolio.optimization import (
    portfolio_annualized_performance,
    max_sharpe_ratio,
    min_variance
)

@pytest.fixture
def portfolio_data():
    """Fixture for portfolio optimization test data."""
    # 4 assets, 100 days of returns
    returns = pd.DataFrame(np.random.normal(0.001, 0.02, (100, 4)), columns=['A', 'B', 'C', 'D'])
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.01
    return mean_returns, cov_matrix, risk_free_rate

def test_portfolio_annualized_performance(portfolio_data):
    """Test the portfolio annualized performance calculation."""
    mean_returns, cov_matrix, _ = portfolio_data
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    std, ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    assert isinstance(std, float)
    assert isinstance(ret, float)
    assert std > 0

def test_max_sharpe_ratio(portfolio_data):
    """Test the max Sharpe ratio optimization."""
    mean_returns, cov_matrix, risk_free_rate = portfolio_data
    result = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    assert result['success'] # Check if the optimization was successful
    # The weights should sum to 1
    assert np.isclose(np.sum(result['x']), 1)
    # Weights should be between 0 and 1
    assert all(0 <= w <= 1 for w in result['x'])

def test_min_variance(portfolio_data):
    """Test the minimum variance optimization."""
    mean_returns, cov_matrix, _ = portfolio_data
    result = min_variance(mean_returns, cov_matrix)
    assert result['success'] # Check if the optimization was successful
    # The weights should sum to 1
    assert np.isclose(np.sum(result['x']), 1)
    # Weights should be between 0 and 1
    assert all(0 <= w <= 1 for w in result['x'])
