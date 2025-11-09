import pandas as pd
from typing import List, Dict, Optional
from ..data_fetch import fetch_stock_data

class Portfolio:
    """
    Represents a stock portfolio, holding tickers, weights, and historical data.
    """
    def __init__(self, tickers: List[str], weights: Optional[Dict[str, float]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initializes the Portfolio object.

        Args:
            tickers (List[str]): A list of stock ticker symbols in the portfolio.
            weights (Optional[Dict[str, float]], optional): 
                A dictionary mapping tickers to weights. If None, equal weights 
                are assumed. Defaults to None.

        Raises:
            ValueError: If the ticker list is empty, or if weights do not match
                        tickers or do not sum to 1.
        """
        if not tickers:
            raise ValueError("Initial ticker list cannot be empty.")

        self.history = self._load_data(tickers, start_date, end_date)
        
        # After loading, the real tickers are the columns of the history df
        self.tickers = list(self.history.columns)

        if not self.tickers:
            print("Warning: Portfolio created with no valid tickers.")
            self.weights = {}
            return

        if weights:
            # Filter weights to only include valid tickers
            valid_weights = {t: w for t, w in weights.items() if t in self.tickers}
            if not valid_weights:
                 # If no provided weights are valid, fall back to equal weighting
                self.weights = {ticker: 1/len(self.tickers) for ticker in self.tickers}
            else:
                # Re-normalize the valid weights to sum to 1
                total_weight = sum(valid_weights.values())
                self.weights = {t: w / total_weight for t, w in valid_weights.items()}
        else:
            # Assign equal weights if none are provided
            self.weights = {ticker: 1/len(self.tickers) for ticker in self.tickers}

    def _load_data(self, tickers: List[str], start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Loads historical closing prices for all tickers in the portfolio.
        """
        all_data = {}
        # Default to '5y' only if no date range is provided
        period = '5y' if start_date is None and end_date is None else None
        for ticker in tickers:
            hist = fetch_stock_data(ticker, period=period, start_date=start_date, end_date=end_date)
            if not hist.empty:
                all_data[ticker] = hist['Close']
        
        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        # Drop rows with missing data for any ticker to align dates
        df.dropna(inplace=True)
        return df

def build_portfolio(tickers: List[str], weights: Optional[Dict[str, float]] = None) -> Portfolio:
    """
    Factory function to create a Portfolio instance.

    Args:
        tickers (List[str]): A list of stock ticker symbols.
        weights (Optional[Dict[str, float]], optional): 
            A dictionary of weights for each ticker. Defaults to None (equal weights).

    Returns:
        Portfolio: An instance of the Portfolio class.
    """
    return Portfolio(tickers, weights)
