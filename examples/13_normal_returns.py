import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.data_fetch import fetch_stock_data
from src.valueinvestpy.analysis.returns import calculate_log_returns
from src.valueinvestpy.visuals.plotting import plot_returns_distribution

def main():
    """Main function to run the returns analysis example."""
    # 1. Define parameters
    ticker = 'AAPL'
    period = '10y'

    print(f"--- Log Return Distribution Analysis for {ticker} ---")
    print(f"Fetching data for the last {period}...")

    # 2. Fetch stock data
    stock_data = fetch_stock_data(ticker, period=period)
    
    if stock_data.empty:
        print(f"Could not fetch data for {ticker}. Exiting.")
        return

    # 3. Calculate log returns
    log_returns = calculate_log_returns(stock_data['Close'])
    
    # 4. Print summary statistics
    print("\n--- Log Return Statistics ---")
    print(f"Mean Daily Log Return: {log_returns.mean():.6f}")
    print(f"Std Dev of Daily Log Return: {log_returns.std():.6f}")

    # 5. Plot the distribution
    print("\nDisplaying returns distribution plot...")
    plot_returns_distribution(
        log_returns,
        title=f'Log Return Distribution for {ticker} ({period})'
    )

if __name__ == '__main__':
    main()
