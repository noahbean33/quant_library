import pandas as pd
import numpy as np
import pytest
from src.valueinvestpy.backtesting.engine import run_backtest
from src.valueinvestpy.strategies.ml_strategies import (
    LogisticRegressionStrategy, SVMStrategy, RegressionMomentumStrategy, 
    CrossSectionalMeanReversion, TimeRebalancedMomentumStrategy, SVMCombinedStrategy, SVMTuningStrategy
)

@pytest.fixture
def sample_ohlc_data():
    """Creates a sample DataFrame of OHLC data for backtesting."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200))
    data = {
        'Open': np.random.uniform(98, 102, 200),
        'High': np.random.uniform(100, 105, 200),
        'Low': np.random.uniform(95, 100, 200),
        'Close': np.random.uniform(99, 103, 200),
        'Volume': np.random.uniform(1e6, 5e6, 200),
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure High is always the highest and Low is the lowest
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    return df

def test_logistic_regression_strategy_runs(sample_ohlc_data):
    """
    Tests that the LogisticRegressionStrategy can be run by the backtesting engine
    without raising an error.
    """
    try:
        # We are not checking the final value, just that it runs to completion.
        final_value = run_backtest(LogisticRegressionStrategy, data=sample_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"LogisticRegressionStrategy failed to run in backtest engine: {e}")

def test_svm_strategy_runs(sample_ohlc_data):
    """
    Tests that the SVMStrategy can be run by the backtesting engine
    without raising an error.
    """
    try:
        # We are not checking the final value, just that it runs to completion.
        final_value = run_backtest(SVMStrategy, data=sample_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"SVMStrategy failed to run in backtest engine: {e}")

@pytest.fixture
def sample_multi_asset_data():
    """Creates a dictionary of sample OHLCV DataFrames for multi-asset backtesting."""
    assets = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E', 'STOCK_F']
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

def test_regression_momentum_strategy_runs(sample_multi_asset_data):
    """
    Tests that the RegressionMomentumStrategy can be run with multiple data feeds.
    """
    try:
        final_value = run_backtest(RegressionMomentumStrategy, data=sample_multi_asset_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"RegressionMomentumStrategy failed to run in backtest engine: {e}")

def test_cross_sectional_mean_reversion_strategy_runs(sample_multi_asset_data):
    """
    Tests that the CrossSectionalMeanReversion strategy can be run with multiple data feeds.
    """
    try:
        final_value = run_backtest(CrossSectionalMeanReversion, data=sample_multi_asset_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"CrossSectionalMeanReversion failed to run in backtest engine: {e}")

def test_time_rebalanced_momentum_strategy_runs(sample_multi_asset_data):
    """
    Tests that the TimeRebalancedMomentumStrategy can be run with a market index
    and multiple data feeds.
    """
    try:
        # The strategy expects the first data feed to be the market index.
        # We'll use one of the assets as a proxy for the index.
        data_feeds = {'market_proxy': sample_multi_asset_data['STOCK_A']} 
        data_feeds.update({k: v for k, v in sample_multi_asset_data.items() if k != 'STOCK_A'})

        final_value = run_backtest(TimeRebalancedMomentumStrategy, data=data_feeds)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"TimeRebalancedMomentumStrategy failed to run in backtest engine: {e}")

def test_svm_combined_strategy_runs(sample_ohlc_data):
    """
    Tests that the SVMCombinedStrategy can be run by the backtesting engine
    without raising an error.
    """
    try:
        final_value = run_backtest(SVMCombinedStrategy, data=sample_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"SVMCombinedStrategy failed to run in backtest engine: {e}")

def test_svm_tuning_strategy_runs(sample_ohlc_data):
    """
    Tests that the SVMTuningStrategy can be run by the backtesting engine
    without raising an error.
    """
    try:
        final_value = run_backtest(SVMTuningStrategy, data=sample_ohlc_data)
        assert isinstance(final_value, float)
    except Exception as e:
        pytest.fail(f"SVMTuningStrategy failed to run in backtest engine: {e}")


