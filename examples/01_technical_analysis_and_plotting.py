# examples/01_technical_analysis_and_plotting.py

"""
This example demonstrates a common workflow:
1. Fetch historical stock data.
2. Calculate a technical indicator (Bollinger Bands).
3. Visualize the stock price and the indicator on an interactive chart.
"""

import yfinance as yf
import sys
import os

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from valueinvestpy.technical_analysis.volatility import calculate_bollinger_bands
from valueinvestpy.visuals.charts import plot_bollinger_bands

def run_example():
    """Fetches data, calculates indicator, and plots chart."""
    # Define the stock and the time period
    ticker = 'MSFT'
    start_date = '2021-01-01'
    end_date = '2023-01-01'

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    # Fetch data using yfinance
    # We use auto_adjust=True to get adjusted closing prices
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        print(f"No data found for ticker {ticker}. Exiting.")
        return

    # Define Bollinger Bands parameters
    window = 20
    num_std_dev = 2

    print(f"Calculating Bollinger Bands with a {window}-day window and {num_std_dev} standard deviations...")

    # Calculate Bollinger Bands using our library function
    data_with_bbands = calculate_bollinger_bands(data, window=window, num_std_dev=num_std_dev)

    print("Generating interactive plot...")

    # Plot the data and the Bollinger Bands
    # This will open a new browser window with an interactive Plotly chart.
    plot_bollinger_bands(data_with_bbands, window=window, num_std_dev=num_std_dev, ticker=ticker)

    print("Example finished. Check the plot in your browser.")

if __name__ == '__main__':
    run_example()
