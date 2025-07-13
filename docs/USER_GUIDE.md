# User Guide

Welcome to the user guide for the Financial Analysis and Trading Platform. This document will walk you through the essential features of the platform and provide examples of how to use them for your own financial analysis and trading activities.

## Table of Contents

1.  [Installation](#installation)
2.  [Running a Backtest](#running-a-backtest)
3.  [Using Technical Indicators](#using-technical-indicators)
4.  [Visualizing Data](#visualizing-data)
5.  [Connecting to a Broker](#connecting-to-a-broker)

---

## Installation

If you haven't already, please follow the installation steps in the main [README.md](../README.md) file.

---

## Running a Backtest

The platform includes a native backtesting engine to test your trading strategies. Here is a conceptual example of how you might run a backtest for a pairs trading strategy.

*Note: This is a simplified example. You will need to adapt it to the specific strategy and data source you are using.*

```python
# main_backtest.py
from valueinvestpy.backtesting import BacktestEngine
from valueinvestpy.strategies.pairs_trading import PairsTradingStrategy
from valueinvestpy.data import Ticker

# 1. Define the stocks for the strategy
stock1 = Ticker('ADBE')
stock2 = Ticker('NVDA')

# 2. Initialize the strategy
strategy = PairsTradingStrategy(stock1, stock2, window=20, z_threshold=2.0)

# 3. Initialize the backtesting engine
engine = BacktestEngine(strategy, start_date='2020-01-01', end_date='2022-01-01')

# 4. Run the simulation
results = engine.run()

# 5. Print the performance summary
print(results.summary())

# 6. Plot the equity curve
results.plot_equity()
```

---

## Using Technical Indicators

You can easily calculate various technical indicators on your financial data.

```python
import yfinance as yf
from valueinvestpy.technical_analysis.volatility import calculate_bollinger_bands
from valueinvestpy.technical_analysis.momentum import calculate_rsi

# Fetch data for a stock
data = yf.download('AAPL', start='2021-01-01', end='2022-01-01')

# Calculate Bollinger Bands
data_with_bbands = calculate_bollinger_bands(data, window=20, num_std_dev=2)
print(data_with_bbands.tail())

# Calculate RSI
data_with_rsi = calculate_rsi(data, window=14)
print(data_with_rsi.tail())
```

---

## Visualizing Data

The platform provides standardized functions for creating interactive charts.

```python
import yfinance as yf
from valueinvestpy.visuals.charts import plot_bollinger_bands

# Fetch data
data = yf.download('MSFT', start='2021-01-01', end='2022-01-01')

# Plot Bollinger Bands
plot_bollinger_bands(data, window=20, num_std_dev=2)
```

This will generate and display an interactive Plotly chart.

---

## Connecting to a Broker

To use the live trading features, you need to connect to a supported broker. Here is an example of how to connect to Interactive Brokers.

**Prerequisites:**
- You must have the Interactive Brokers Trader Workstation (TWS) or Gateway application running and logged in.
- Ensure the API settings in TWS/Gateway are configured to allow connections.

```python
from valueinvestpy.trading.broker import InteractiveBrokersBroker

# 1. Initialize the broker
# The default connection parameters are for a local TWS instance.
ib_broker = InteractiveBrokersBroker(host='127.0.0.1', port=7497, client_id=1)

# 2. Connect to the broker
ib_broker.connect()

# 3. Check if connected
if ib_broker.is_connected():
    print("Successfully connected to Interactive Brokers!")

    # 4. Fetch your current positions
    positions = ib_broker.get_positions()
    for pos in positions:
        print(f"Symbol: {pos['symbol']}, Quantity: {pos['quantity']}, Average Cost: {pos['average_cost']}")

    # 5. Disconnect when done
    ib_broker.disconnect()
```
