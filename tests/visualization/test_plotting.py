import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.financial_analysis_platform.data.data_preprocessing import (
    calculate_moving_averages,
    generate_ma_crossover_signals,
)
from src.financial_analysis_platform.visualization.plotting import (
    plot_ma_crossover_strategy,
    plot_monte_carlo_simulation,
    plot_bollinger_bands,
    plot_rsi,
    plot_macd,
    plot_atr,
    plot_stochastic_oscillator,
)
from src.financial_analysis_platform.analysis.technical_analysis import (
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_stochastic_oscillator,
)

@pytest.fixture
def sample_stock_data():
    """Fixture for creating sample stock data for plotting tests."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=150))
    close = np.random.uniform(95, 105, size=150)
    high = close + np.random.uniform(0, 2, size=150)
    low = close - np.random.uniform(0, 2, size=150)
    data = pd.DataFrame({
        'High': high,
        'Low': low,
        'Close': close
    }, index=dates)
    return data

@pytest.fixture
def ma_crossover_data(sample_stock_data):
    """Fixture to generate data with moving average crossover signals."""
    short_window = 40
    long_window = 100
    data = calculate_moving_averages(sample_stock_data, [short_window, long_window])
    data = generate_ma_crossover_signals(data, short_window, long_window)
    return data, short_window, long_window

@pytest.fixture
def monte_carlo_data():
    """Fixture for creating sample Monte Carlo simulation data."""
    return pd.DataFrame(np.random.rand(252, 10))

def test_plot_ma_crossover_strategy_plotly(ma_crossover_data):
    """Tests the interactive plot_ma_crossover_strategy function."""
    data, short_window, long_window = ma_crossover_data
    fig = plot_ma_crossover_strategy(data, short_window, long_window)
    assert isinstance(fig, go.Figure)
    # 3 traces for prices/MAs + 2 for signals
    assert len(fig.data) == 5

@pytest.fixture
def technical_analysis_data(sample_stock_data):
    """Fixture to generate data with Bollinger Bands and RSI."""
    data = calculate_bollinger_bands(sample_stock_data)
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = calculate_atr(data)
    data = calculate_stochastic_oscillator(data)
    return data

def test_plot_monte_carlo_simulation_plotly(monte_carlo_data):
    """Tests the interactive plot_monte_carlo_simulation function."""
    ticker = "TEST"
    fig = plot_monte_carlo_simulation(monte_carlo_data, ticker)
    assert isinstance(fig, go.Figure)
    # 10 simulation traces + 1 for the legend
    assert len(fig.data) == 11

def test_plot_bollinger_bands_plotly(technical_analysis_data):
    """Tests the interactive plot_bollinger_bands function."""
    ticker = "TEST"
    fig = plot_bollinger_bands(technical_analysis_data, ticker)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4

def test_plot_rsi_plotly(technical_analysis_data):
    """Tests the interactive plot_rsi function."""
    ticker = "TEST"
    fig = plot_rsi(technical_analysis_data, ticker)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 2  # for hlines

def test_plot_macd_plotly(technical_analysis_data):
    """Tests the interactive plot_macd function."""
    ticker = "TEST"
    fig = plot_macd(technical_analysis_data, ticker)
    assert isinstance(fig, go.Figure)
    # Price, MACD, Signal, Histogram
    assert len(fig.data) == 4

def test_plot_atr_plotly(technical_analysis_data):
    """Tests the interactive plot_atr function."""
    ticker = "TEST"
    fig = plot_atr(technical_analysis_data, ticker)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

def test_plot_stochastic_oscillator_plotly(technical_analysis_data):
    """Tests the interactive plot_stochastic_oscillator function."""
    ticker = "TEST"
    fig = plot_stochastic_oscillator(technical_analysis_data, ticker)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 2  # for hlines
