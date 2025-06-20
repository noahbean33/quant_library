import yfinance as yf
from src.financial_analysis_platform.data.data_preprocessing import (
    calculate_bollinger_bands,
    calculate_rsi,
)
from src.financial_analysis_platform.visualization.plotting import (
    plot_bollinger_bands,
    plot_rsi,
)

# 1. Fetch data
ticker = "TSLA"
data = yf.download(ticker, start="2022-01-01", end="2023-01-01")

# 2. Calculate technical indicators
data = calculate_bollinger_bands(data)
data = calculate_rsi(data)

# 3. Generate and show plots
bollinger_fig = plot_bollinger_bands(data, ticker)
bollinger_fig.show()

rsi_fig = plot_rsi(data, ticker)
rsi_fig.show()
