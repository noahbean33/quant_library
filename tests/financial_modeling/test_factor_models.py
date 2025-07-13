import pandas as pd
import numpy as np
import pytest
import statsmodels.api as sm
from valueinvestpy.financial_modeling.factor_models import run_factor_model

@pytest.fixture
def factor_model_data():
    """Fixture for factor model test data."""
    # Create 100 days of data
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    
    # Portfolio excess returns
    portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    
    # Fama-French 3-factor returns
    factor_returns = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.0005, 0.01, 100),
        'SMB': np.random.normal(0.0001, 0.005, 100),
        'HML': np.random.normal(0.0002, 0.006, 100)
    }, index=dates)
    
    return portfolio_returns, factor_returns

def test_run_factor_model(factor_model_data):
    """Test the factor model regression."""
    portfolio_returns, factor_returns = factor_model_data
    model_results = run_factor_model(portfolio_returns, factor_returns)
    
    # Check if the result is a statsmodels results wrapper
    assert isinstance(model_results, sm.regression.linear_model.RegressionResultsWrapper)
    
    # The number of parameters should be the number of factors + 1 (for the constant/alpha)
    assert len(model_results.params) == factor_returns.shape[1] + 1
    
    # Check if the parameter names are correct
    expected_params = ['const'] + factor_returns.columns.tolist()
    assert model_results.params.index.tolist() == expected_params
