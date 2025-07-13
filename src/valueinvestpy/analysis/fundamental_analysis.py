import yfinance as yf
import pandas as pd

def calculate_ratios(ticker: str) -> dict:
    """
    Calculates key financial ratios for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        dict: A dictionary containing key financial ratios.
              Returns an empty dictionary if data cannot be retrieved.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    ratios = {}

    # Valuation Ratios
    ratios['pe_ratio'] = info.get('trailingPE')
    ratios['pb_ratio'] = info.get('priceToBook')
    ratios['ps_ratio'] = info.get('priceToSalesTrailing12Months')
    ratios['earnings_yield'] = 1 / info.get('trailingPE') if info.get('trailingPE') else None

    # Profitability Ratios
    ratios['return_on_equity'] = info.get('returnOnEquity')
    ratios['return_on_assets'] = info.get('returnOnAssets')
    ratios['profit_margin'] = info.get('profitMargins')

    # Financial Health Ratios
    ratios['debt_to_equity'] = info.get('debtToEquity')
    ratios['current_ratio'] = info.get('currentRatio')

    if not any(ratios.values()):
        print(f"Warning: Could not calculate ratios for ticker '{ticker}'. Check if the ticker is valid.")
        return {}

    return ratios

def calculate_graham_number(ticker: str) -> float:
    """
    Calculates the Graham Number for a given stock ticker.
    Formula: sqrt(22.5 * EPS * Book Value per Share)

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        float: The calculated Graham Number, or None if data is unavailable.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    eps = info.get('trailingEps')
    bvps = info.get('bookValue')

    if eps is None or bvps is None:
        print(f"Warning: Could not retrieve EPS or Book Value for ticker '{ticker}'.")
        return None
    
    if eps <= 0 or bvps <= 0:
        return 0.0 # Graham number is not meaningful for negative earnings/book value

    graham_number = (22.5 * eps * bvps) ** 0.5
    return graham_number
