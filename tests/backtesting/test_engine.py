import pandas as pd
import numpy as np
import pytest
from valueinvestpy.backtesting.engine import run_backtest
from valueinvestpy.backtesting.strategies.moving_average import SmaCross

@pytest.fixture
def ohlcv_data():
    """Fixture for sample OHLCV data."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    data = pd.DataFrame(index=dates)
    data['Open'] = 100 + np.random.randn(100).cumsum()
    data['High'] = data['Open'] + np.random.uniform(0, 2, 100)
    data['Low'] = data['Open'] - np.random.uniform(0, 2, 100)
    data['Close'] = (data['Open'] + data['High'] + data['Low']) / 3
    data['Volume'] = np.random.randint(10000, 50000, 100)
    return data

def test_run_backtest(ohlcv_data):
    """Test the backtrader engine runner."""
    # Run the backtest with the SmaCross strategy
    final_value = run_backtest(SmaCross, ohlcv_data)
    
    # The final value should be a float
    assert isinstance(final_value, float)
    
    # The final value should be positive (unless the strategy is exceptionally bad)
    assert final_value > 0
