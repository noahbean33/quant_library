import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from financial_analysis_platform.analysis.time_series_analysis import plot_forecast

def test_plot_forecast(stationary_data):
    train_data = stationary_data.iloc[:80]
    test_data = stationary_data.iloc[80:]

    # Create a dummy forecast
    forecast = np.random.randn(len(test_data))
    
    # Create dummy confidence intervals
    conf_int = pd.DataFrame({
        'lower': forecast - 0.5,
        'upper': forecast + 0.5
    }, index=test_data.index)

    fig = plot_forecast(forecast, train_data, test_data, conf_int=conf_int)
    
    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == 'Forecast vs Actuals'
    assert ax.get_legend() is not None
    plt.close(fig)

def test_plot_forecast_no_conf_int(stationary_data):
    train_data = stationary_data.iloc[:80]
    test_data = stationary_data.iloc[80:]

    # Create a dummy forecast
    forecast = np.random.randn(len(test_data))
    
    fig = plot_forecast(forecast, train_data, test_data)
    
    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == 'Forecast vs Actuals'
    assert ax.get_legend() is not None
    # Check that there are no fill_between collections
    assert len(ax.collections) == 0
    plt.close(fig)
