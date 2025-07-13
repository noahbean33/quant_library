import yfinance as yf
import pandas as pd
from typing import Optional

def fetch_stock_data(ticker: str, period: str = '5y', start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieves historical stock data for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.
        period (str, optional): The period for which to fetch data. 
                                Defaults to '5y'. Is ignored if start_date 
                                and end_date are provided.
        start_date (Optional[str], optional): The start date for data retrieval (YYYY-MM-DD).
        end_date (Optional[str], optional): The end date for data retrieval (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame containing historical stock data,
                      or an empty DataFrame if the ticker is invalid.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, start=start_date, end=end_date)
    if hist.empty:
        print(f"Warning: No data found for ticker '{ticker}'. It may be delisted or invalid.")
    return hist

def fetch_financials(ticker: str) -> dict:
    """
    Retrieves financial statements for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        dict: A dictionary containing 'income_statement', 'balance_sheet',
              and 'cash_flow' as pandas DataFrames. Returns an empty
              dictionary if the ticker is invalid or data is not available.
    """
    stock = yf.Ticker(ticker)
    financials = {}
    try:
        financials['income_statement'] = stock.income_stmt
        financials['balance_sheet'] = stock.balance_sheet
        financials['cash_flow'] = stock.cashflow
        if all(df.empty for df in financials.values()):
            print(f"Warning: No financial data found for ticker '{ticker}'.")
            return {}
    except Exception as e:
        print(f"An error occurred while fetching financials for {ticker}: {e}")
        return {}
    return financials
