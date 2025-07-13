import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.analysis.financial_models import calculate_capm
from src.valueinvestpy.visuals.plotting import plot_capm

def main():
    """Main function to run the CAPM analysis example."""
    # 1. Define the stock and market tickers
    stock_ticker = 'AAPL'
    market_ticker = '^GSPC' # S&P 500 Index
    start_date = '2018-01-01'
    end_date = '2023-01-01'

    # 2. Run the CAPM analysis
    print(f"Running CAPM analysis for {stock_ticker} against {market_ticker}...")
    capm_results = calculate_capm(
        stock_ticker=stock_ticker,
        market_ticker=market_ticker,
        start_date=start_date,
        end_date=end_date
    )

    if not capm_results:
        print("Failed to perform CAPM analysis.")
        return

    # 3. Print the results
    print("\n--- CAPM Results ---")
    print(f"Beta: {capm_results['beta']:.4f}")
    print(f"Alpha: {capm_results['alpha']:.4f}")
    print(f"Expected Annual Return: {capm_results['expected_return']:.2%}")

    # 4. Visualize the results
    print(f"\nDisplaying the Security Characteristic Line (SCL) for {stock_ticker}...")
    plot_capm(capm_results, stock_ticker, market_ticker)

if __name__ == '__main__':
    main()
