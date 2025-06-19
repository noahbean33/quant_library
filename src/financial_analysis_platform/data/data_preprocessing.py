
# data_preprocessing.py

# pip install pandas numpy yfinance matplotlib seaborn forex-python nasdaq-data-link

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def convert_prices_to_returns(ticker, start_date, end_date):
    """
    Converts adjusted close prices to simple and log returns.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL')
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    - DataFrame: Contains 'Adj Close', 'simple_rtn', and 'log_rtn'
    """
    #df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = get_data_yahoo_finance(ticker, start_date, end_date)

    df = df.loc[:, ["Adj Close"]]
    df["simple_rtn"] = df["Adj Close"].pct_change()
    df["log_rtn"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    return df
'''
def adjust_returns_for_inflation(ticker, start_date, end_date, nasdaq_api_key):
    """
    Adjusts stock returns for inflation using CPI data from Nasdaq Data Link.

    Parameters:
    - ticker (str): Stock ticker symbol
    - start_date (str): Start date for data
    - end_date (str): End date for data
    - nasdaq_api_key (str): Your Nasdaq Data Link API key

    Returns:
    - DataFrame: Contains 'Adj Close', 'cpi', 'simple_rtn', 'inflation_rate', 'real_rtn'
    """
    # Fetch stock data and resample to monthly
    #df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = get_data_yahoo_finance(ticker, start_date, end_date)
    df = df.loc[:, ["Adj Close"]]
    df = df.resample("M").last()

    # Fetch CPI data
    import nasdaqdatalink
    nasdaqdatalink.ApiConfig.api_key = nasdaq_api_key
    df_cpi = nasdaqdatalink.get(dataset="RATEINF/CPI_USA",
                                start_date=start_date,
                                end_date=end_date).rename(columns={"Value": "cpi"})

    # Join CPI data with stock data
    df = df.join(df_cpi, how="left")

    # Calculate returns and inflation rate
    df["simple_rtn"] = df["Adj Close"].pct_change()
    df["inflation_rate"] = df["cpi"].pct_change()

    # Calculate real returns
    df["real_rtn"] = ((df["simple_rtn"] + 1) / (df["inflation_rate"] + 1)) - 1
    return df
'''
def calculate_realized_volatility(ticker, start_date, end_date):
    """
    Calculates the annualized realized volatility of a stock.

    Parameters:
    - ticker (str): Stock ticker symbol
    - start_date (str): Start date for data
    - end_date (str): End date for data

    Returns:
    - DataFrame: Contains 'rv' (realized volatility)
    """
    # Fetch stock data
    #df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = get_data_yahoo_finance(ticker, start_date, end_date)

    df = df.loc[:, ["Adj Close"]].rename(columns={"Adj Close": "adj_close"})

    # Calculate log returns
    df["log_rtn"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df = df.dropna()

    # Define realized volatility function
    def realized_volatility(x):
        return np.sqrt(np.sum(x**2))

    # Calculate monthly realized volatility
    df_rv = df.groupby(pd.Grouper(freq="M")).apply(realized_volatility)
    df_rv = df_rv.rename(columns={"log_rtn": "rv"})

    # Annualize volatility
    df_rv["rv"] = df_rv["rv"] * np.sqrt(12)
    return df_rv

def impute_missing_values(df, method='interpolate'):
    """
    Imputes missing values in a DataFrame.

    Parameters:
    - df (DataFrame): DataFrame with missing values
    - method (str): Method of imputation ('ffill', 'bfill', 'interpolate')

    Returns:
    - DataFrame: DataFrame with imputed values
    """
    if method == 'interpolate':
        df_filled = df.interpolate()
    else:
        df_filled = df.fillna(method=method)
    return df_filled
'''
def convert_currency(df, from_currency, to_currency):
    """
    Converts currency of price data in a DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing price data with DateTime index
    - from_currency (str): Original currency code (e.g., 'USD')
    - to_currency (str): Target currency code (e.g., 'EUR')

    Returns:
    - DataFrame: DataFrame with converted currency columns
    """
    from forex_python.converter import CurrencyRates
    c = CurrencyRates()

    # Get exchange rates for each date
    df[f"{from_currency}_{to_currency}"] = [c.get_rate(from_currency, to_currency, date) for date in df.index]

    # Convert prices
    for column in df.columns[:-1]:  # Exclude the exchange rate column
        df[f"{column}_{to_currency}"] = df[column] * df[f"{from_currency}_{to_currency}"]
    return df
'''
def aggregate_trade_data(trades_df, bar_type='time', bar_size='1Min'):
    """
    Aggregates trade data into bars (time, tick, volume, or dollar bars).

    Parameters:
    - trades_df (DataFrame): DataFrame containing 'price', 'qty', 'quoteQty', and 'time'
    - bar_type (str): Type of bar to aggregate ('time', 'tick', 'volume', 'dollar')
    - bar_size (str or int): Size of the bar (e.g., '1Min' for time bars, integer for other types)

    Returns:
    - DataFrame: Aggregated bars
    """
    def get_bars(grouped):
        ohlc = grouped['price'].ohlc()
        vwap = (grouped.apply(lambda x: np.average(x['price'], weights=x['qty']))).to_frame('vwap')
        vol = grouped['qty'].sum().to_frame('vol')
        cnt = grouped['qty'].count().to_frame('cnt')
        res = pd.concat([ohlc, vwap, vol, cnt], axis=1)
        return res

    if bar_type == 'time':
        trades_df.set_index('time', inplace=True)
        df_grouped = trades_df.groupby(pd.Grouper(freq=bar_size))
        bars = get_bars(df_grouped)
    elif bar_type == 'tick':
        bar_size = int(bar_size)
        trades_df['tick_group'] = (np.arange(len(trades_df)) // bar_size)
        df_grouped = trades_df.groupby('tick_group')
        bars = get_bars(df_grouped)
    elif bar_type == 'volume':
        bar_size = int(bar_size)
        trades_df['cum_qty'] = trades_df['qty'].cumsum()
        trades_df['vol_group'] = (trades_df['cum_qty'] // bar_size)
        df_grouped = trades_df.groupby('vol_group')
        bars = get_bars(df_grouped)
    elif bar_type == 'dollar':
        bar_size = int(bar_size)
        trades_df['cum_value'] = trades_df['quoteQty'].cumsum()
        trades_df['value_group'] = (trades_df['cum_value'] // bar_size)
        df_grouped = trades_df.groupby('value_group')
        bars = get_bars(df_grouped)
    else:
        raise ValueError("Invalid bar_type. Choose from 'time', 'tick', 'volume', 'dollar'.")
    return bars

def main():
    # Suppress warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)

    # Set plotting style
    sns.set_theme(context="talk", style="whitegrid",
                  palette="colorblind", color_codes=True,
                  rc={"figure.figsize": [12, 8]})

    # Test 1: Convert prices to returns
    print("Testing convert_prices_to_returns...")
    df_returns = convert_prices_to_returns("AAPL", "2010-01-01", "2020-12-31")
    print(df_returns.head())
    '''
    # Test 2: Adjust returns for inflation
    print("\nTesting adjust_returns_for_inflation...")
    NASDAQ_API_KEY = nasdaq_api_key  # Replace with your actual API key
    df_inflation = adjust_returns_for_inflation("AAPL", "2010-01-01", "2020-12-31", NASDAQ_API_KEY)
    print(df_inflation.head())
    '''
    # Test 3: Calculate realized volatility
    print("\nTesting calculate_realized_volatility...")
    df_volatility = calculate_realized_volatility("AAPL", "2000-01-01", "2010-12-31")
    print(df_volatility.head())

    # Test 4: Impute missing values
    print("\nTesting impute_missing_values...")
    # Create sample data with missing values
    dates = pd.date_range('2020-01-01', periods=10)
    data = {'value': [np.nan if i % 3 == 0 else i for i in range(10)]}
    df_missing = pd.DataFrame(data, index=dates)
    df_filled = impute_missing_values(df_missing, method='interpolate')
    print(df_filled)
    '''
    # Test 5: Convert currency
    print("\nTesting convert_currency...")
    df_prices = yf.download("AAPL", start="2020-01-01", end="2020-01-31", progress=False)
    df_prices = df_prices.drop(columns=["Adj Close", "Volume"])
    df_converted = convert_currency(df_prices, "USD", "EUR")
    print(df_converted.head())
    '''
    # Test 6: Aggregate trade data
    print("\nTesting aggregate_trade_data...")
    # Sample trade data
    trade_data = {
        'price': np.random.uniform(100, 110, 500),
        'qty': np.random.uniform(1, 10, 500),
        'quoteQty': np.random.uniform(100, 1000, 500),
        'time': pd.date_range('2020-01-01', periods=500, freq='T')
    }
    trades_df = pd.DataFrame(trade_data)
    bars = aggregate_trade_data(trades_df, bar_type='time', bar_size='5Min')
    print(bars.head())

if __name__ == "__main__":
    main()
