import pandas as pd
import mplfinance as mpf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot_candlestick(data: pd.DataFrame, title: str = 'Candlestick Chart', mav: tuple = (10, 20), bbands: tuple = None, show_volume: bool = True):
    """Plots an OHLCV candlestick chart with optional moving averages and volume.

    Args:
        data (pd.DataFrame): A DataFrame with a DatetimeIndex and columns for
                             'Open', 'High', 'Low', 'Close', and 'Volume'.
        title (str, optional): The title of the chart. Defaults to 'Candlestick Chart'.
        mav (tuple, optional): A tuple of moving average windows to plot. Defaults to (10, 20).
        show_volume (bool, optional): Whether to show the volume panel. Defaults to True.
    """
    # Ensure the column names are in the format mplfinance expects (lowercase with first letter capitalized)
    data.columns = [col.capitalize() for col in data.columns]

    mpf.plot(data,
             type='candle',
             style='yahoo',
             title=title,
             ylabel='Price',
             mav=mav,
             bollinger=bbands,
             volume=show_volume,
             ylabel_lower='Volume')

def plot_correlogram(data: pd.Series, lags: int = 20, title: str = 'Autocorrelation Function (ACF)'):
    """
    Plots the correlogram (ACF) for a given time series.

    Args:
        data (pd.Series): The time series data to analyze.
        lags (int, optional): The number of lags to include in the plot. Defaults to 20.
        title (str, optional): The title of the chart. Defaults to 'Autocorrelation Function (ACF)'.
    """
    plt.figure()
    plot_acf(data, lags=lags, title=title)
    plt.show()

def plot_pca_results(explained_variance, factor_exposures):
    """
    Visualizes the results of a PCA analysis.

    Creates three plots:
    1. A bar chart of the explained variance by each principal component.
    2. A line chart of the cumulative explained variance.
    3. A scatter plot of the assets based on their exposure to the first two factors.

    Args:
        explained_variance (pd.Series): The explained variance ratio for each component.
        factor_exposures (pd.DataFrame): The factor exposures for each asset.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Explained Variance by Component',
            'Cumulative Explained Variance',
            'Factor Exposure (PC1 vs PC2)'
        )
    )

    # 1. Bar chart for individual explained variance
    fig.add_trace(
        go.Bar(
            x=explained_variance.index,
            y=explained_variance.values,
            name='Explained Variance'
        ),
        row=1, col=1
    )

    # 2. Line chart for cumulative explained variance
    cumulative_variance = explained_variance.cumsum()
    fig.add_trace(
        go.Scatter(
            x=cumulative_variance.index,
            y=cumulative_variance.values,
            mode='lines+markers',
            name='Cumulative Variance'
        ),
        row=1, col=2
    )

    # 3. Scatter plot for factor exposures
    fig.add_trace(
        go.Scatter(
            x=factor_exposures['PC1'],
            y=factor_exposures['PC2'],
            mode='markers+text',
            text=factor_exposures.index,
            textposition='top center',
            name='Assets'
        ),
        row=1, col=3
    )

    fig.update_layout(
        title_text='Principal Component Analysis (PCA) Results',
        showlegend=False,
        height=500
    )
    fig.show()

def plot_bollinger_bands(data: pd.DataFrame, title: str = 'Bollinger Bands'):
    """
    Plots Bollinger Bands using Plotly.

    Args:
        data (pd.DataFrame): A DataFrame with a DatetimeIndex and columns for
                             'Close', 'UpperBand', 'MiddleBand', and 'LowerBand'.
        title (str, optional): The title of the chart. Defaults to 'Bollinger Bands'.
    """
    fig = go.Figure()

    # Add traces for the upper and lower bands
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['UpperBand'],
        line=dict(color='rgba(255, 165, 0, 0.5)'),
        name='Upper Band'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['LowerBand'],
        line=dict(color='rgba(255, 165, 0, 0.5)'),
        fill='tonexty',  # Fill the area between the upper and lower bands
        fillcolor='rgba(255, 165, 0, 0.1)',
        name='Lower Band'
    ))

    # Add trace for the close price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        line=dict(color='blue'),
        name='Close Price'
    ))

    # Add trace for the middle band (moving average)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MiddleBand'],
        line=dict(color='orange', dash='dash'),
        name='Middle Band'
    ))

    fig.update_layout(
        title_text=title,
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True
    )

    fig.show()
