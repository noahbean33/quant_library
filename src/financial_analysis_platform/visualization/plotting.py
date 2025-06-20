import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_close_price(df: pd.DataFrame, ticker: str):
    """
    Plots the close price for a given ticker.
    Assumes the 'Close' column is the adjusted price (yfinance default).

    Args:
        df (pd.DataFrame): DataFrame with stock data, including 'Close'.
        ticker (str): The ticker symbol for the plot title.
    """
    sns.set_style('darkgrid')
    plt.figure(figsize=(15, 8))
    df['Close'].plot(linewidth=2)
    plt.title(f'Close Price for {ticker}', fontsize=16)
    plt.ylabel('Price ($)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.grid(True)
    plt.show()

def plot_simple_returns(df: pd.DataFrame, ticker: str):
    """
    Plots the simple returns for a given ticker.

    Args:
        df (pd.DataFrame): DataFrame with stock data, including 'simple_rtn'.
        ticker (str): The ticker symbol for the plot title.
    """
    sns.set_style('darkgrid')
    plt.figure(figsize=(15, 8))
    df['simple_rtn'].plot(linewidth=2)
    plt.title(f'Simple Returns for {ticker}', fontsize=16)
    plt.ylabel('Returns', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.grid(True)
    plt.show()

def plot_return_distribution(df: pd.DataFrame, ticker: str):
    """
    Plots the distribution of simple returns for a given ticker.

    Args:
        df (pd.DataFrame): DataFrame with stock data, including 'simple_rtn'.
        ticker (str): The ticker symbol for the plot title.
    """
    sns.set_style('darkgrid')
    plt.figure(figsize=(15, 8))
    sns.histplot(df['simple_rtn'], bins=50, kde=True, stat='density')
    plt.title(f'Distribution of Simple Returns for {ticker}', fontsize=16)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlabel('Simple Returns', fontsize=14)
    plt.grid(True)
    plt.show()

def plot_moving_averages(df: pd.DataFrame, ticker: str, windows: list):
    """
    Plots the close price and moving averages for a given ticker.

    Args:
        df (pd.DataFrame): DataFrame with stock data, including 'Close' and MA columns.
        ticker (str): The ticker symbol for the plot title.
        windows (list): A list of integers representing the window sizes for the MAs.
    """
    sns.set_style('darkgrid')
    plt.figure(figsize=(15, 8))
    df['Close'].plot(linewidth=2, label='Close Price')
    for window in windows:
        df[f'MA_{window}'].plot(linewidth=2, label=f'MA {window}')
    plt.title(f'Close Price and Moving Averages for {ticker}', fontsize=16)
    plt.ylabel('Price ($)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_realized_volatility(df_rv: pd.DataFrame, ticker: str):
    """
    Plots the realized volatility for a given ticker.

    Args:
        df_rv (pd.DataFrame): DataFrame with realized volatility data.
        ticker (str): The ticker symbol for the plot title.
    """
    sns.set_style('darkgrid')
    plt.figure(figsize=(15, 8))
    df_rv['rv'].plot(linewidth=2)
    plt.title(f'Realized Volatility for {ticker}', fontsize=16)
    plt.ylabel('Volatility', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.grid(True)
    plt.show()

def plot_ma_crossover_strategy(df: pd.DataFrame, short_window: int, long_window: int):
    """Plots the Moving Average Crossover Strategy with Plotly for interactivity."""
    fig = go.Figure()

    # Add traces for price and moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{short_window}'], mode='lines', name=f'MA {short_window}', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{long_window}'], mode='lines', name=f'MA {long_window}', line=dict(dash='dash')))

    # Add Buy signals
    buy_signals = df[df['position'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals[f'MA_{short_window}'],
                             mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))

    # Add Sell signals
    sell_signals = df[df['position'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals[f'MA_{short_window}'],
                              mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))

    # Update layout
    fig.update_layout(
        title='Moving Average Crossover Strategy',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_dark'
    )

    return fig

def plot_monte_carlo_simulation(simulation_df: pd.DataFrame, ticker: str):
    """Plots the Monte Carlo simulation results using Plotly."""
    fig = go.Figure()

    # Add a trace for each simulation path
    for i, col in enumerate(simulation_df.columns):
        fig.add_trace(go.Scatter(
            x=simulation_df.index,
            y=simulation_df[col],
            mode='lines',
            name=f'Sim {i+1}',
            line=dict(width=1),
            showlegend=False  # Hide individual traces from the legend
        ))

    # Update layout
    fig.update_layout(
        title=f'Monte Carlo Simulation for {ticker}',
        xaxis_title='Day',
        yaxis_title='Price',
        template='plotly_dark',
        showlegend=True # Show the main legend
    )
    
    # Add a single dummy trace for a clean legend entry
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Simulations', line=dict(color='grey')))

    return fig

def plot_bollinger_bands(df: pd.DataFrame, ticker: str):
    """Plots Bollinger Bands using Plotly."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Upper Band', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA'], mode='lines', name='Moving Average', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Lower Band', line=dict(color='green', dash='dash')))

    fig.update_layout(
        title=f'Bollinger Bands for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    return fig

def plot_rsi(df: pd.DataFrame, ticker: str):
    """Plots the RSI with overbought/oversold levels using Plotly."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Price for {ticker}', 'RSI'))

    # Price Plot
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'), row=1, col=1)

    # RSI Plot
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        title_text=f'RSI Analysis for {ticker}',
        template='plotly_dark',
        showlegend=False
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig
