# examples/03_backtesting_a_strategy.py

"""
This example demonstrates how to use the ``run_backtest`` helper to test a
simple trading strategy.
It uses a basic moving average crossover strategy implemented with Backtrader
and runs it on historical data fetched with ``yfinance``.
"""

import yfinance as yf
from valueinvestpy.backtesting.engine import run_backtest
from valueinvestpy.backtesting.strategies.moving_average import SmaCross

def run_example():
    """Runs the backtest using a simple moving average crossover strategy."""
    ticker = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2022-01-01'
    initial_capital = 10000.0

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        print(f"No data found for ticker {ticker}. Exiting.")
        return

    print("Running backtest...")
    final_value = run_backtest(SmaCross, data, cash=initial_capital)

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    
if __name__ == '__main__':
    run_example()
