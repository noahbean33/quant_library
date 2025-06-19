# fetch_financial_data.py

import pandas as pd
import yfinance as yf
import nasdaqdatalink
import intrinio_sdk as intrinio
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from pycoingecko import CoinGeckoAPI
from datetime import datetime

from .. import config


def get_data_yahoo_finance(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    return df

def get_data_nasdaq_datalink(ticker, start_date, end_date, api_key):
    """
    Fetch historical stock data from Nasdaq Data Link.
    """
    nasdaqdatalink.ApiConfig.api_key = api_key
    df = nasdaqdatalink.get(dataset=f"WIKI/{ticker}", start_date=start_date, end_date=end_date)
    return df

def get_data_intrinio(ticker, start_date, end_date, api_key):
    """
    Fetch historical stock data from Intrinio.
    """
    intrinio.ApiClient().set_api_key(api_key)
    security_api = intrinio.SecurityApi()
    r = security_api.get_security_stock_prices(
        identifier=ticker,
        start_date=start_date,
        end_date=end_date,
        frequency="daily",
        page_size=10000
    )
    df = pd.DataFrame(r.stock_prices_dict).sort_values("date").set_index("date")
    return df

def get_data_alpha_vantage_stock(ticker, api_key):
    """
    Fetch daily stock data from Alpha Vantage.
    """
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='full')
    data.sort_index(inplace=True)
    return data

def get_data_alpha_vantage_crypto(symbol, market, api_key):
    """
    Fetch daily cryptocurrency data from Alpha Vantage.
    """
    cc = CryptoCurrencies(key=api_key, output_format='pandas')
    data, _ = cc.get_digital_currency_daily(symbol=symbol, market=market)
    data.sort_index(inplace=True)
    return data

def get_data_coingecko(coin_id, vs_currency, days):
    """
    Fetch OHLC cryptocurrency data from CoinGecko.
    """
    cg = CoinGeckoAPI()
    ohlc = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    ohlc_df = pd.DataFrame(ohlc)
    ohlc_df.columns = ["date", "open", "high", "low", "close"]
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"], unit="ms")
    return ohlc_df


