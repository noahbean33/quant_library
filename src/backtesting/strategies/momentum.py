import backtrader as bt
from src.backtesting.indicators.momentum import Momentum

class MomentumStrategy(bt.Strategy):
    """
    A portfolio-level momentum strategy that ranks stocks and trades the top
    quintile, using a market regime filter (S&P 500 vs. its 200-day SMA).
    """

    def __init__(self):
        self.counter = 0
        self.indicators = {}
        self.sorted_data = []
        # The first data feed is assumed to be the market index (e.g., SPY)
        self.spy = self.datas[0]
        # All subsequent data feeds are the stocks in the universe
        self.stocks = self.datas[1:]

        for stock in self.stocks:
            self.indicators[stock] = {}
            self.indicators[stock]['momentum'] = Momentum(stock.close, period=90)
            self.indicators[stock]['sma100'] = bt.indicators.SimpleMovingAverage(stock.close, period=100)
            self.indicators[stock]['atr20'] = bt.indicators.ATR(stock, period=20)

        # Market regime filter: 200-day SMA on the index
        self.sma200 = bt.indicators.MovingAverageSimple(self.spy.close, period=200)

    def prenext(self):
        # Call next() even when data is not available for all tickers
        self.next()

    def next(self):
        # Rebalance weekly
        if self.counter % 5 == 0:
            self.rebalance_portfolio()
        # Update position sizes bi-weekly
        if self.counter % 10 == 0:
            self.update_positions()

        self.counter += 1

    def rebalance_portfolio(self):
        """Ranks stocks by momentum and closes/opens positions."""
        # Filter out stocks with insufficient data
        self.sorted_data = list(filter(lambda stock_data: len(stock_data) > 100, self.stocks))
        # Sort stocks by their momentum value in descending order
        self.sorted_data.sort(key=lambda stock: self.indicators[stock]['momentum'][0], reverse=True)
        num_stocks = len(self.sorted_data)

        # Close positions for stocks that are no longer in the top 20% 
        # or have fallen below their 100-day SMA.
        for i, stock in enumerate(self.sorted_data):
            if self.getposition(stock).size:
                if i > 0.2 * num_stocks or stock.close[0] < self.indicators[stock]['sma100'][0]:
                    self.close(stock)

        # Do not open new positions if the market is in a downtrend (bear market)
        if self.spy.close[0] < self.sma200[0]:
            return

        # Open long positions in the top 20% of stocks
        for stock in self.sorted_data[:int(0.2 * num_stocks)]:
            if self.broker.get_cash() > 0 and not self.getposition(stock).size:
                # Size position based on volatility (ATR)
                size = self.broker.get_value() * 0.001 / self.indicators[stock]["atr20"][0]
                self.buy(stock, size=size)

    def update_positions(self):
        """Updates the size of existing positions."""
        num_stocks = len(self.sorted_data)
        # Iterate through the top 20% of stocks
        for stock in self.sorted_data[:int(0.2 * num_stocks)]:
            if self.broker.get_cash() > 0:
                size = self.broker.get_value() * 0.001 / self.indicators[stock]["atr20"][0]
                self.order_target_size(stock, size)
