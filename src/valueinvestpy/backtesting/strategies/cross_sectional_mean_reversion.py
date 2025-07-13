import backtrader as bt
import numpy as np

class CrossSectionalMeanReversion(bt.Strategy):
    """
    A portfolio-level, market-neutral strategy that implements cross-sectional
    mean reversion. It shorts recent winners and buys recent losers based on
    their daily returns relative to the portfolio's average return.
    """

    def __init__(self):
        """Initializes the strategy."""
        self.stock_data = self.datas

    def prenext(self):
        """Ensures next() is called even with missing data for some tickers."""
        self.next()

    def next(self):
        """Executes the rebalancing logic on each bar."""
        stock_returns = np.zeros(len(self.stock_data))

        # Calculate the last daily return for each stock
        for index, stock in enumerate(self.stock_data):
            stock_returns[index] = (stock.close[0] - stock.close[-1]) / stock.close[-1]

        # Calculate the average return of the stock universe
        market_return = np.mean(stock_returns)
        
        # Calculate weights: negative of the excess return
        weights = -(stock_returns - market_return)
        
        # Normalize weights to create a dollar-neutral portfolio
        weights = weights / np.sum(np.abs(weights))

        # Rebalance the portfolio to the new target weights
        for index, stock in enumerate(self.stock_data):
            self.order_target_percent(stock, target=weights[index])
