import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine import run_backtest
from src.strategies.pairs_trading import PairsTradingStrategy

@pytest.fixture
def cointegrated_pair_data():
    """Generates a pair of cointegrated time series for testing."""
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=300))
    
    # Create a random walk for the second asset
    price2 = 100 + np.random.randn(300).cumsum()
    
    # Create a stationary series for the error term
    error_term = np.random.randn(300) * 2
    
    # Create the first asset as a linear combination of the second plus the error
    hedge_ratio = 0.8
    price1 = hedge_ratio * price2 + 5 + error_term

    data1 = pd.DataFrame({'Open': price1, 'High': price1, 'Low': price1, 'Close': price1, 'Volume': np.ones(300)*1e6}, index=dates)
    data2 = pd.DataFrame({'Open': price2, 'High': price2, 'Low': price2, 'Close': price2, 'Volume': np.ones(300)*1e6}, index=dates)

    return {'ASSET_A': data1, 'ASSET_B': data2}

def test_pairs_trading_strategy_runs(cointegrated_pair_data):
    """
    Tests that the PairsTradingStrategy can be run by the backtesting engine
    with a pair of cointegrated data feeds without raising an error.
    """
    try:
        final_value = run_backtest(PairsTradingStrategy, data=cointegrated_pair_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"PairsTradingStrategy failed to run in backtest engine: {e}")
