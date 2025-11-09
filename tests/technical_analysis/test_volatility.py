import pandas as pd
import numpy as np
import pytest
from src.technical_analysis.volatility import bollinger_bands, calculate_bollinger_bands
from src.technical_analysis.moving_averages import sma

@pytest.fixture
def sample_data():
    """Fixture to provide a sample pandas Series for testing."""
    return pd.Series(np.random.uniform(90, 110, size=50))

@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample pandas DataFrame for testing."""
    data = {
        'High': np.arange(100, 150),
        'Low': np.arange(90, 140),
        'Close': np.arange(95, 145)
    }
    return pd.DataFrame(data)

def test_bollinger_bands(sample_data):
    """Test the Bollinger Bands calculation."""
    bbands = bollinger_bands(sample_data, window=20)
    assert isinstance(bbands, pd.DataFrame)
    assert list(bbands.columns) == ['Upper', 'Middle', 'Lower']
    assert bbands.isnull().sum().sum() == 3 * 19 # NaN values for the lookback period

    # The middle band should be the SMA
    expected_middle = sma(sample_data, window=20)
    pd.testing.assert_series_equal(bbands['Middle'], expected_middle, check_names=False)

    # Upper band should always be >= Lower band
    assert (bbands.dropna()['Upper'] >= bbands.dropna()['Lower']).all()

def test_calculate_bollinger_bands(sample_dataframe):
    """Test the calculate_bollinger_bands function."""
    bands = calculate_bollinger_bands(sample_dataframe, period=20, std_dev=2)
    assert isinstance(bands, pd.DataFrame)
    assert list(bands.columns) == ['MiddleBand', 'UpperBand', 'LowerBand']
    assert bands.isnull().sum().sum() == 3 * 19  # NaN values for the lookback period

    # Check calculation for a specific point (e.g., index 19, the first valid point)
    close_prices = sample_dataframe['Close'][:20]
    expected_middle = close_prices.mean()
    expected_std = close_prices.std()
    
    assert np.isclose(bands['MiddleBand'].iloc[19], expected_middle)
    assert np.isclose(bands['UpperBand'].iloc[19], expected_middle + 2 * expected_std)
    assert np.isclose(bands['LowerBand'].iloc[19], expected_middle - 2 * expected_std)

    # Upper band should always be >= Lower band for non-null values
    assert (bands.dropna()['UpperBand'] >= bands.dropna()['LowerBand']).all()

    # Test with 'Typical' price source
    bands_typical = calculate_bollinger_bands(sample_dataframe, period=20, std_dev=2, price_source='Typical')
    assert isinstance(bands_typical, pd.DataFrame)
    
    typical_price = (sample_dataframe['High'] + sample_dataframe['Low'] + sample_dataframe['Close']) / 3
    tp_prices = typical_price[:20]
    expected_middle_tp = tp_prices.mean()
    expected_std_tp = tp_prices.std()

    assert np.isclose(bands_typical['MiddleBand'].iloc[19], expected_middle_tp)
    assert np.isclose(bands_typical['UpperBand'].iloc[19], expected_middle_tp + 2 * expected_std_tp)
    assert np.isclose(bands_typical['LowerBand'].iloc[19], expected_middle_tp - 2 * expected_std_tp)
