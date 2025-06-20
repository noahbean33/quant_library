# tests/analysis/test_time_series_decomposition.py

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from financial_analysis_platform.analysis.time_series_analysis import (
    plot_stl_decomposition, apply_hp_filter
)

def test_plot_stl_decomposition(stationary_data):
    """Test plot_stl_decomposition function."""
    fig = plot_stl_decomposition(stationary_data, period=7)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_apply_hp_filter(non_stationary_data):
    """Test apply_hp_filter function."""
    cycle, trend = apply_hp_filter(non_stationary_data)
    assert isinstance(cycle, pd.Series)
    assert isinstance(trend, pd.Series)
    assert cycle.shape == non_stationary_data.shape
    assert trend.shape == non_stationary_data.shape

def test_plot_stl_decomposition_with_overlay(stationary_data):
    stl_result_2 = STL(stationary_data, period=12, robust=False).fit()
    fig = plot_stl_decomposition(
        stationary_data,
        period=12,
        robust=True,
        stl_result_2=stl_result_2,
        labels=["Robust", "Non-robust"],
    )
    assert fig is not None
    # Check if legend is present
    ax_trend = fig.get_axes()[1]
    assert ax_trend.get_legend() is not None
    plt.close(fig)
