import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..data_fetch import fetch_stock_data
import numpy as np
from scipy.stats import norm

def plot_price_history(ticker: str, period: str = '5y'):
    """
    Plots the historical closing price of a stock.

    Args:
        ticker (str): The stock ticker symbol.
        period (str, optional): The period for which to fetch data. 
                                Defaults to '5y'.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
                                  Returns None if data cannot be fetched.
    """
    hist = fetch_stock_data(ticker, period=period)
    if hist.empty:
        return None

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(hist.index, hist['Close'], label=f'{ticker} Closing Price')
    
    ax.set_title(f'{ticker} Price History ({period})', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

import pandas as pd
import mplfinance as mpf
from typing import List, Optional
from ..analysis.technical import calculate_sma, calculate_ema, calculate_rsi

def plot_candlestick(
    data: pd.DataFrame,
    ticker: str,
    moving_averages: Optional[List[int]] = None,
    rsi_window: Optional[int] = 14,
    volume: bool = True,
    title: Optional[str] = None
):
    """
    Generates an interactive candlestick chart with optional technical indicators.

    Args:
        data (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'.
        ticker (str): The stock ticker symbol for the chart title.
        moving_averages (Optional[List[int]]): A list of window sizes for SMAs to plot.
        rsi_window (Optional[int]): The window for the RSI calculation. If None, RSI is not plotted.
        volume (bool): Whether to display the volume panel.
        title (Optional[str]): A custom title for the chart.
    """
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Data must contain 'Open', 'High', 'Low', 'Close' columns.")

    # Prepare add-on plots
    add_plots = []
    
    # Calculate and add moving averages
    if moving_averages:
        for window in moving_averages:
            sma = calculate_sma(data['Close'], window=window)
            add_plots.append(mpf.make_addplot(sma, panel=0))

    # Calculate and add RSI
    if rsi_window:
        rsi = calculate_rsi(data['Close'], window=rsi_window)
        # Add RSI to a new panel (panel=2) and an overbought/oversold line
        add_plots.append(mpf.make_addplot(rsi, panel=2, color='orange', ylabel='RSI'))
        add_plots.append(mpf.make_addplot([70] * len(data), panel=2, color='r', linestyle='--'))
        add_plots.append(mpf.make_addplot([30] * len(data), panel=2, color='g', linestyle='--'))


    # Set chart title
    chart_title = title if title else f'{ticker} Candlestick Chart'

    # Adjust panel ratios based on whether RSI is plotted
    panel_ratios = (6, 1, 2) if rsi_window else (3, 1)

    # Plot using mplfinance
    mpf.plot(
        data,
        type='candle',
        style='yahoo',
        title=chart_title,
        ylabel='Price ($)',
        volume=volume,
        addplot=add_plots,
        panel_ratios=panel_ratios,
        figscale=1.5 # Make the plot larger
    )

def plot_efficient_frontier(simulated_portfolios, max_sharpe_portfolio, min_vol_portfolio):
    """
    Plots the efficient frontier from Markowitz optimization results.

    Args:
        simulated_portfolios (pd.DataFrame): DataFrame of simulated portfolios.
        max_sharpe_portfolio (dict): Dictionary with stats for the max Sharpe ratio portfolio.
        min_vol_portfolio (dict): Dictionary with stats for the min volatility portfolio.
    """
    plt.figure(figsize=(12, 7))
    
    # Plot all the simulated portfolios
    plt.scatter(
        simulated_portfolios['volatility'],
        simulated_portfolios['return'],
        c=simulated_portfolios['sharpe_ratio'],
        cmap='viridis',
        marker='o',
        s=10,
        alpha=0.5,
        label='Simulated Portfolios'
    )
    
    # Highlight the max Sharpe ratio and min volatility portfolios
    plt.scatter(
        max_sharpe_portfolio['volatility'],
        max_sharpe_portfolio['return'],
        marker='*',
        color='r',
        s=200,
        label='Max Sharpe Ratio'
    )
    plt.scatter(
        min_vol_portfolio['volatility'],
        min_vol_portfolio['return'],
        marker='*',
        color='b',
        s=200,
        label='Min Volatility'
    )
    
    plt.title('Efficient Frontier')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_capm(capm_results: dict, stock_ticker: str, market_ticker: str):
    """
    Plots the Security Characteristic Line (SCL) from CAPM results.

    Args:
        capm_results (dict): The results dictionary from calculate_capm.
        stock_ticker (str): The ticker symbol of the stock.
        market_ticker (str): The ticker symbol of the market index.
    """
    returns_data = capm_results['returns_data']
    alpha = capm_results['alpha']
    beta = capm_results['beta']

    plt.figure(figsize=(12, 7))
    
    # Scatter plot of the monthly returns
    plt.scatter(
        returns_data['market'], 
        returns_data['stock'], 
        alpha=0.6, 
        label='Monthly Returns'
    )
    
    # Regression line (SCL)
    plt.plot(
        returns_data['market'], 
        beta * returns_data['market'] + alpha, 
        color='red', 
        linewidth=2,
        label='Security Characteristic Line (SCL)'
    )
    
    plt.title(f'CAPM Analysis: {stock_ticker} vs. {market_ticker}', fontsize=16)
    plt.xlabel(f'Market Returns ({market_ticker})', fontsize=12)
    plt.ylabel(f'Stock Returns ({stock_ticker})', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Add text for alpha and beta
    text_str = f'α (Alpha): {alpha:.4f}\nβ (Beta): {beta:.4f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
             
    plt.show()

def plot_returns_distribution(returns: pd.Series, title: str, bins: int = 100):
    """
    Plots a histogram of returns and overlays a normal distribution curve.

    Args:
        returns (pd.Series): A pandas Series of asset returns.
        title (str): The title for the plot.
        bins (int, optional): The number of bins for the histogram. Defaults to 100.
    """
    # Drop NA values for calculation
    returns = returns.dropna()
    
    # Calculate statistics
    mu = returns.mean()
    sigma = returns.std()
    
    plt.figure(figsize=(12, 7))
    
    # Plot histogram
    plt.hist(returns, bins=bins, density=True, alpha=0.7, label='Log Returns')
    
    # Plot normal distribution curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
    
    plt.title(title)
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stochastic_process(sim_df: pd.DataFrame, title: str, xlabel: str, ylabel: str = 'Value'):
    """
    Plots the simulated paths from a stochastic process simulation.

    Args:
        sim_df (pd.DataFrame): DataFrame containing the simulation paths, where each 
                             column is a separate simulation.
        title (str): The title for the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Value'.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(sim_df)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()