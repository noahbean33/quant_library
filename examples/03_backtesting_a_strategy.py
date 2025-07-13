# examples/03_backtesting_a_strategy.py

"""
This example demonstrates how to use the native backtesting engine to test
a simple trading strategy.

We will define a simple Moving Average Crossover strategy and run it against
historical data for a stock.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from valueinvestpy.backtesting import BacktestEngine

# For this self-contained example, we will define the strategy and data handling
# directly in this file. In a real application, these would be in their own modules.

class MovingAverageCrossoverStrategy:
    """
    A simple strategy that buys when the short-term moving average crosses
    above the long-term moving average, and sells when it crosses below.
    """
    def __init__(self, data, short_window=40, long_window=100):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signals = self._generate_signals()

    def _generate_signals(self):
        """Generates the trading signals based on moving average crossover."""
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0

        # Create short and long simple moving averages
        signals['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        # Create signals
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] 
                                                        > signals['long_mavg'][self.short_window:], 1.0, 0.0)   

        # Generate trading orders
        signals['positions'] = signals['signal'].diff()
        return signals

def run_example():
    """Runs the backtest for the moving average crossover strategy."""
    # Define the stock and the time period
    ticker = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2022-01-01'
    initial_capital = 10000.0

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        print(f"No data found for ticker {ticker}. Exiting.")
        return

    print("Initializing strategy and backtesting engine...")
    # 1. Initialize the Strategy
    strategy = MovingAverageCrossoverStrategy(data, short_window=40, long_window=100)

    # 2. Initialize the Backtest Engine
    engine = BacktestEngine(strategy.signals, data, initial_capital=initial_capital)

    print("Running backtest...")
    # 3. Run the Backtest
    portfolio = engine.run()

    print("\n--- Backtest Results ---")
    # 4. Get Performance Metrics
    stats = portfolio.get_performance_summary()
    print(f"Final Portfolio Value: ${stats['Final Value']:.2f}")
    print(f"Total Return: {stats['Total Return']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max Drawdown']:.2f}%")

    # 5. Plot the Equity Curve and Signals
    print("\nGenerating performance plots...")
    portfolio.plot_equity_curve()
    portfolio.plot_signals()
    print("Example finished. Check the plots.")

if __name__ == '__main__':
    run_example()
