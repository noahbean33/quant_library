# main.py

from . import config
from .data import fetch_financial_data

def run_data_fetching():
    """Main function to demonstrate data fetching capabilities."""
    # Load API keys from config
    NASDAQ_API_KEY = config.NASDAQ_API_KEY
    INTRINIO_API_KEY = config.INTRINIO_API_KEY
    ALPHA_VANTAGE_API_KEY = config.ALPHA_VANTAGE_API_KEY

    # Define the parameters
    ticker = "AAPL"
    start_date = "2011-01-01"
    end_date = "2021-12-31"

    # Yahoo Finance
    print("Fetching data from Yahoo Finance...")
    df_yahoo = fetch_financial_data.get_data_yahoo_finance(ticker, start_date, end_date)
    print(f"Yahoo Finance data:\n{df_yahoo.head()}\n")

    # Nasdaq Data Link
    if NASDAQ_API_KEY and NASDAQ_API_KEY != "YOUR_NASDAQ_API_KEY":
        print("Fetching data from Nasdaq Data Link...")
        df_nasdaq = fetch_financial_data.get_data_nasdaq_datalink(ticker, start_date, end_date, NASDAQ_API_KEY)
        print(f"Nasdaq Data Link data:\n{df_nasdaq.head()}\n")
    else:
        print("Skipping Nasdaq Data Link: API key not found or is placeholder.")

    # Intrinio
    if INTRINIO_API_KEY and INTRINIO_API_KEY != "YOUR_INTRINIO_API_KEY":
        print("Fetching data from Intrinio...")
        df_intrinio = fetch_financial_data.get_data_intrinio(ticker, start_date, end_date, INTRINIO_API_KEY)
        print(f"Intrinio data:\n{df_intrinio.head()}\n")
    else:
        print("Skipping Intrinio: API key not found or is placeholder.")

    # Alpha Vantage Stock Data
    if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        print("Fetching stock data from Alpha Vantage...")
        df_alpha_stock = fetch_financial_data.get_data_alpha_vantage_stock(ticker, ALPHA_VANTAGE_API_KEY)
        print(f"Alpha Vantage Stock data:\n{df_alpha_stock.head()}\n")

        # Alpha Vantage Crypto Data
        print("Fetching cryptocurrency data from Alpha Vantage...")
        crypto_symbol = "BTC"
        market = "USD"
        df_alpha_crypto = fetch_financial_data.get_data_alpha_vantage_crypto(crypto_symbol, market, ALPHA_VANTAGE_API_KEY)
        print(f"Alpha Vantage Crypto data:\n{df_alpha_crypto.head()}\n")
    else:
        print("Skipping Alpha Vantage: API key not found or is placeholder.")

    # CoinGecko
    print("Fetching data from CoinGecko...")
    coin_id = "bitcoin"
    vs_currency = "usd"
    days = 14
    df_coingecko = fetch_financial_data.get_data_coingecko(coin_id, vs_currency, days)
    print(f"CoinGecko data:\n{df_coingecko.head()}\n")

if __name__ == "__main__":
    run_data_fetching()
