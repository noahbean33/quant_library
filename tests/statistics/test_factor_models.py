import pandas as pd
import numpy as np
import pytest
from src.valueinvestpy.statistics.factor_models import perform_pca_on_returns

@pytest.fixture
def sample_returns_data():
    """Creates a sample DataFrame of stock returns for testing."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    data = {
        'STOCK_A': np.random.normal(0.001, 0.02, 100),
        'STOCK_B': np.random.normal(0.0005, 0.025, 100),
        'STOCK_C': np.random.normal(-0.0002, 0.015, 100),
        'STOCK_D': np.random.normal(0.0008, 0.022, 100),
    }
    return pd.DataFrame(data, index=dates)

def test_perform_pca_on_returns(sample_returns_data):
    """Tests the perform_pca_on_returns function."""
    n_components = 3
    explained_variance, factor_exposures, factor_returns = perform_pca_on_returns(
        sample_returns_data, n_components=n_components
    )

    # Check output types
    assert isinstance(explained_variance, pd.Series)
    assert isinstance(factor_exposures, pd.DataFrame)
    assert isinstance(factor_returns, pd.DataFrame)

    # Check output shapes
    assert len(explained_variance) == n_components
    assert factor_exposures.shape == (sample_returns_data.shape[1], n_components)
    assert factor_returns.shape == (sample_returns_data.shape[0], n_components)

    # Check that explained variance is ordered and sums to less than 1
    assert all(explained_variance.iloc[i] >= explained_variance.iloc[i+1] for i in range(len(explained_variance)-1))
    assert explained_variance.sum() <= 1.0
