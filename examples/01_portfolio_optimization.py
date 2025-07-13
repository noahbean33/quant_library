import sys
import os
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.portfolio import Portfolio
from src.valueinvestpy.portfolio.optimizer import optimize_portfolio
from src.valueinvestpy.visuals.plotting import plot_efficient_frontier

def main():
    """Main function to run the portfolio optimization example."""
    # 1. Define the portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    portfolio = Portfolio(tickers, start_date='2018-01-01', end_date='2023-01-01')

    if portfolio.history.empty:
        print("Could not fetch data for the portfolio. Exiting.")
        return

    # 2. Run the portfolio optimization
    print("Running Markowitz Portfolio Optimization...")
    simulated_portfolios, max_sharpe_portfolio, min_vol_portfolio = optimize_portfolio(portfolio)

    # 3. Print the optimal portfolios
    print("\n--- Maximum Sharpe Ratio Portfolio ---")
    print(f"Return: {max_sharpe_portfolio['return']:.2%}")
    print(f"Volatility: {max_sharpe_portfolio['volatility']:.2%}")
    print(f"Sharpe Ratio: {max_sharpe_portfolio['sharpe_ratio']:.2f}")
    print("Optimal Weights:")
    for ticker, weight in max_sharpe_portfolio['weights'].items():
        print(f"  {ticker}: {weight:.2%}")

    print("\n--- Minimum Volatility Portfolio ---")
    print(f"Return: {min_vol_portfolio['return']:.2%}")
    print(f"Volatility: {min_vol_portfolio['volatility']:.2%}")
    print(f"Sharpe Ratio: {min_vol_portfolio['sharpe_ratio']:.2f}")
    print("Optimal Weights:")
    for ticker, weight in min_vol_portfolio['weights'].items():
        print(f"  {ticker}: {weight:.2%}")

    # 4. Visualize the efficient frontier
    print("\nDisplaying the Efficient Frontier plot...")
    plot_efficient_frontier(simulated_portfolios, max_sharpe_portfolio, min_vol_portfolio)

if __name__ == '__main__':
    main()
