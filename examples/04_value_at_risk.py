import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.data_fetch import fetch_stock_data
from src.valueinvestpy.analysis.risk import calculate_var

def main():
    """Main function to run the Value at Risk (VaR) calculation example."""
    # 1. Define the parameters
    stock_ticker = 'C' # Citigroup Inc.
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    investment_amount = 1_000_000 # $1,000,000 investment
    confidence_level = 0.99 # 99% confidence level

    # 2. Fetch data and calculate returns
    print(f"Fetching historical data for {stock_ticker}...")
    stock_data = fetch_stock_data(stock_ticker, start_date=start_date, end_date=end_date)

    if stock_data.empty:
        print("Could not fetch stock data. Exiting.")
        return

    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()

    # 3. Calculate VaR for different time horizons
    print(f"\n--- Value at Risk (VaR) Calculation ---")
    print(f"Investment: ${investment_amount:,.2f}")
    print(f"Confidence Level: {confidence_level:.0%}")

    # Calculate 1-day VaR
    var_1_day = calculate_var(
        investment=investment_amount,
        returns=log_returns,
        confidence_level=confidence_level,
        horizon_days=1
    )
    print(f"1-Day VaR: ${var_1_day:,.2f}")
    print(f"This means there is a {1-confidence_level:.0%} probability that the portfolio will lose at least ${var_1_day:,.2f} in the next day.")

    # Calculate 10-day VaR
    var_10_day = calculate_var(
        investment=investment_amount,
        returns=log_returns,
        confidence_level=confidence_level,
        horizon_days=10
    )
    print(f"\n10-Day VaR: ${var_10_day:,.2f}")
    print(f"This means there is a {1-confidence_level:.0%} probability that the portfolio will lose at least ${var_10_day:,.2f} in the next 10 days.")

if __name__ == '__main__':
    main()
