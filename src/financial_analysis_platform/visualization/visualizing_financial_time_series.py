# visualizing_financial_time_series.py

#pip install pandas numpy yfinance matplotlib seaborn plotly cufflinks nasdaq-data-link

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Set plotting style
sns.set_theme(context="talk", style="whitegrid",
              palette="colorblind", color_codes=True,
              rc={"figure.figsize": [12, 8]})


def basic_visualization(ticker, start_date, end_date):
    """
    Plots the adjusted close prices and simple returns of a stock.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'MSFT')
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    - None
    """
    # Download stock data
    #df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    df = get_data_yahoo_finance(ticker, start_date, end_date)

    # Calculate simple returns
    df['simple_rtn'] = df['Adj Close'].pct_change()
    df = df.dropna()

    # Plot adjusted close prices and simple returns
    fig, ax = plt.subplots(2, 1, sharex=True)
    
    # Adjusted Close Price
    df['Adj Close'].plot(ax=ax[0])
    ax[0].set(title=f"{ticker} Stock Prices", ylabel="Adjusted Close Price ($)")
    
    # Simple Returns
    df['simple_rtn'].plot(ax=ax[1])
    ax[1].set(ylabel="Simple Returns")
    
    sns.despine()
    plt.tight_layout()
    plt.show()

'''
def visualize_seasonal_patterns(start_date, end_date, nasdaq_api_key):
    """
    Visualizes seasonal patterns in unemployment data.

    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format
    - nasdaq_api_key (str): Your Nasdaq Data Link API key

    Returns:
    - None
    """
    # Import Nasdaq Data Link
    import nasdaqdatalink
    nasdaqdatalink.ApiConfig.api_key = nasdaq_api_key
  
    df = get_data_yahoo_finance(ticker, start_date, end_date)

    # Download unemployment data
    df = nasdaqdatalink.get(dataset="FRED/UNRATENSA", start_date=start_date, end_date=end_date)
    df = df.rename(columns={"Value": "unemp_rate"})

    # Create 'year' and 'month' columns
    df['year'] = df.index.year
    df['month'] = df.index.strftime('%b')

    # Plot seasonal patterns
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='month', y='unemp_rate', hue='year', style='year', palette='colorblind')
    plt.title("Unemployment Rate - Seasonal Plot")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    sns.despine()
    plt.tight_layout()
    plt.show()
'''

def create_interactive_visualizations(ticker, start_date, end_date):
    """
    Creates interactive plots of stock prices using Plotly.

    Parameters:
    - ticker (str): Stock ticker symbol
    - start_date (str): Start date
    - end_date (str): End date

    Returns:
    - None
    """
    # Import Plotly and Cufflinks
    import cufflinks as cf
    import plotly.express as px
    cf.go_offline()

    # Download stock data
    #df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    df = get_data_yahoo_finance(ticker, start_date, end_date)
    df['simple_rtn'] = df['Adj Close'].pct_change()
    df = df[['Adj Close', 'simple_rtn']].dropna()

    # Create interactive plot
    fig = px.line(df, y='Adj Close', title=f"{ticker} Stock Prices")
    fig.show()


def create_candlestick_chart(ticker, start_date, end_date):
    """
    Creates a candlestick chart for a given stock.

    Parameters:
    - ticker (str): Stock ticker symbol
    - start_date (str): Start date
    - end_date (str): End date

    Returns:
    - None
    """
    # Import Plotly Graph Objects
    import plotly.graph_objects as go

    # Download stock data
    #df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    df = get_data_yahoo_finance(ticker, start_date, end_date)

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.update_layout(title=f"{ticker} Stock Prices",
                      yaxis_title='Price ($)')
    fig.show()


def main():
    # Basic Visualization
    print("Basic Visualization of MSFT stock in 2020")
    basic_visualization('MSFT', '2020-01-01', '2020-12-31')
    '''
    # Visualize Seasonal Patterns
    print("\nVisualizing Seasonal Patterns in Unemployment Data")
    #NASDAQ_API_KEY = nasdaq_api_key #'YOUR_NASDAQ_API_KEY'  # Replace with your Nasdaq Data Link API key
    visualize_seasonal_patterns('2014-01-01', '2019-12-31', NASDAQ_API_KEY)
    '''
    # Interactive Visualization
    print("\nCreating Interactive Visualization for MSFT")
    create_interactive_visualizations('MSFT', '2020-01-01', '2020-12-31')

    # Candlestick Chart
    print("\nCreating Candlestick Chart for MSFT in 2018")
    create_candlestick_chart('MSFT', '2018-01-01', '2018-12-31')


if __name__ == "__main__":
    main()
