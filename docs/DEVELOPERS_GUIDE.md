# Developer's Guide

Welcome to the developer's guide for the Financial Analysis and Trading Platform. This document provides a deep dive into the platform's architecture, design principles, and instructions for extending its functionality.

## Architecture Overview

The platform is designed to be modular and extensible. The core components are organized into the following directories within the `src/valueinvestpy` package:

- **`trading`**: Contains the broker integration logic, including the abstract `Broker` class and concrete implementations like `InteractiveBrokersBroker`.
- **`strategies`**: Houses the trading strategy logic. Each strategy should be implemented as a class that can be consumed by the backtesting engine or a live trading application.
- **`technical_analysis`**: A library of functions for calculating technical indicators and financial metrics.
- **`statistics`**: Includes modules for statistical analysis and modeling, such as stochastic processes.
- **`visuals`**: Contains standardized functions for creating interactive plots and charts using Plotly.
- **`dash_apps`**: Standalone Dash applications for advanced analysis, such as the PCA analyzer.

## Contributing to the Platform

### Adding a New Technical Indicator

1.  **Add the function** to the relevant file in `src/valueinvestpy/technical_analysis` (e.g., `volatility.py`, `momentum.py`).
2.  **Write a corresponding unit test** in the `tests/technical_analysis` directory. Ensure the test covers edge cases and provides clear assertions.
3.  **(Optional)** Add a new plotting function to `src/valueinvestpy/visuals/charts.py` if the indicator requires a specific type of visualization.

### Adding a New Trading Strategy

1.  **Create a new file** in `src/valueinvestpy/strategies` for your strategy.
2.  **Implement the strategy** as a class. The design should allow it to be used by both the backtesting engine and a live trading application.
3.  **Add unit tests** for the strategy's logic in the `tests/strategies` directory.

### Adding a New Broker

1.  **Create a new class** in `src/valueinvestpy/trading` that inherits from the abstract `Broker` class.
2.  **Implement all the abstract methods** defined in the `Broker` interface (`connect`, `get_positions`, `stream_prices`, `place_order`, etc.).
3.  **Add integration tests** to ensure the connection and data flow with the new broker's API are working correctly.

## Running the Test Suite

To ensure the integrity of the codebase, please run the full test suite before submitting any changes.

```bash
pytest
```
