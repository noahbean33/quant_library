import pandas as pd
import numpy as np
import pytest
from src.valueinvestpy.backtesting.engine import run_backtest
from src.valueinvestpy.strategies.technical_strategies import EMACrossoverStrategy, BollingerBandStrategy, MovingAverageRSIStrategy, MomentumStrategy

@pytest.fixture
def sample_ohlc_data():
    """Creates a sample OHLCV DataFrame for backtesting."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200))
    data = {
        'Open': np.random.uniform(98, 102, 200),
        'High': np.random.uniform(100, 105, 200),
        'Low': np.random.uniform(95, 100, 200),
        'Close': np.random.uniform(99, 103, 200),
        'Volume': np.random.uniform(1e6, 5e6, 200),
    }
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    return df

@pytest.fixture
def multi_asset_ohlc_data():
    """Creates a dictionary of sample OHLCV DataFrames for multi-asset backtesting."""
    assets = ['MARKET_INDEX', 'STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
    data_dict = {}
    for asset in assets:
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200))
        data = {
            'Open': np.random.uniform(98, 102, 200),
            'High': np.random.uniform(100, 105, 200),
            'Low': np.random.uniform(95, 100, 200),
            'Close': np.random.uniform(99, 103, 200),
            'Volume': np.random.uniform(1e6, 5e6, 200),
        }
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        data_dict[asset] = df
    return data_dict


def test_ema_crossover_strategy_runs(sample_ohlc_data):
    """
    Tests that the EMACrossoverStrategy can be run with a single data feed.
    """
    try:
        final_value = run_backtest(EMACrossoverStrategy, data=sample_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"EMACrossoverStrategy failed to run in backtest engine: {e}")

def test_bollinger_band_strategy_runs(sample_ohlc_data):
    """
    Tests that the BollingerBandStrategy can be run with a single data feed.
    """
    try:
        final_value = run_backtest(BollingerBandStrategy, data=sample_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"BollingerBandStrategy failed to run in backtest engine: {e}")

def test_moving_average_rsi_strategy_runs(sample_ohlc_data):
    """
    Tests that the MovingAverageRSIStrategy can be run with a single data feed.
    """
    try:
        final_value = run_backtest(MovingAverageRSIStrategy, data=sample_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"MovingAverageRSIStrategy failed to run in backtest engine: {e}")

def test_momentum_strategy_runs(multi_asset_ohlc_data):
    """
    Tests that the MomentumStrategy can be run by the backtesting engine
    with multiple data feeds without raising an error.
    """
    try:
        # The first data feed is the market index, the rest are stocks
        final_value = run_backtest(MomentumStrategy, data=multi_asset_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"MomentumStrategy failed to run in backtest engine: {e}")

