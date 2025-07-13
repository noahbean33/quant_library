import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.data_fetch import fetch_stock_data
from src.valueinvestpy.analysis.returns import calculate_log_returns
from src.valueinvestpy.analysis.risk import calculate_var, calculate_var_monte_carlo

def main():
    """Main function to run the Value at Risk (VaR) calculation example."""
    # 1. Define VaR parameters
    ticker = 'C'  # Citigroup
    period = '5y'
    initial_investment = 1_000_000
    confidence_level = 0.99
    horizon_days = 1
    num_simulations = 100000

    print(f"--- Value at Risk (VaR) Analysis for {ticker} ---")
    print(f"Investment: ${initial_investment:,.2f}, Confidence: {confidence_level:.0%}, Horizon: {horizon_days} day(s)")

    # 2. Fetch stock data and calculate returns
    print(f"Fetching {period} of historical data for {ticker}...")
    stock_data = fetch_stock_data(ticker, period=period)
    
    if stock_data.empty:
        print(f"Could not fetch data for {ticker}. Exiting.")
        return

    log_returns = calculate_log_returns(stock_data['Close'])

    # 3. Calculate VaR using the analytical (variance-covariance) method
    var_analytical = calculate_var(
        investment=initial_investment, 
        returns=log_returns, 
        confidence_level=confidence_level, 
        horizon_days=horizon_days
    )
    print(f"\nAnalytical VaR:           ${var_analytical:,.2f}")

    # 4. Calculate VaR using Monte Carlo simulation
    print(f"Running Monte Carlo simulation with {num_simulations} paths...")
    var_mc = calculate_var_monte_carlo(
        initial_investment=initial_investment,
        returns=log_returns,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
        num_simulations=num_simulations
    )
    print(f"Monte Carlo VaR:          ${var_mc:,.2f}")

    print(f"\nThis means we are {confidence_level:.0%} confident that we will not lose more than ~${var_mc:,.2f} in the next day.")

if __name__ == '__main__':
    main()
