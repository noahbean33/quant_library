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

The platform includes a helper function, `run_backtest`, to test your trading strategies. Here is a conceptual example of how you might run a backtest for a pairs trading strategy.

*Note: This is a simplified example. You will need to adapt it to the specific strategy and data source you are using.*

```python
# main_backtest.py
from valueinvestpy.backtesting.engine import run_backtest
from valueinvestpy.strategies.pairs_trading import PairsTradingStrategy
from valueinvestpy.data_fetch import fetch_stock_data

# 1. Fetch historical data for each asset
stock1 = fetch_stock_data('ADBE', start_date='2020-01-01', end_date='2022-01-01')
stock2 = fetch_stock_data('NVDA', start_date='2020-01-01', end_date='2022-01-01')

# 2. Prepare the data dictionary
data = {
    'ADBE': stock1,
    'NVDA': stock2,
}

# 3. Run the backtest
final_value = run_backtest(PairsTradingStrategy, data)

# 4. Display the final portfolio value
print(f"Final portfolio value: ${final_value:.2f}")
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
