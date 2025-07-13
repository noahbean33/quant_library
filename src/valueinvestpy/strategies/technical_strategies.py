import backtrader as bt
import numpy as np
from src.valueinvestpy.technical_analysis.momentum import Momentum

class EMACrossoverStrategy(bt.Strategy):
    """
    A classic Exponential Moving Average (EMA) Crossover strategy.
    Buys when the short-term EMA crosses above the long-term EMA.
    Sells when the short-term EMA crosses below the long-term EMA.
    """
    params = (
        ('short_period', 30),
        ('long_period', 100),
    )

    def __init__(self):
        """Initialize the strategy and its indicators."""
        ema_short = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.short_period
        )
        ema_long = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.long_period
        )
        
        # The CrossOver indicator returns 1.0 on an upward cross and -1.0 on a downward cross.
        self.crossover = bt.indicators.CrossOver(ema_short, ema_long)

    def next(self):
        """Define the trading logic based on the crossover signal."""
        if not self.position:  # Not in the market
            if self.crossover[0] > 0:  # Upward crossover
                self.buy()
        elif self.crossover[0] < 0:  # Downward crossover
            self.close()


class MovingAverageRSIStrategy(bt.Strategy):
    """
    A strategy that combines Moving Average Crossover with the Relative Strength Index (RSI).
    It enters a long position when the short-term moving average crosses above the
    long-term moving average and the RSI is below a certain threshold (indicating
    an oversold condition). It exits the position when the short-term MA crosses
    below the long-term MA.
    """
    params = (
        ('short_period', 40),
        ('long_period', 150),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
    )

    def __init__(self):
        self.short_ma = bt.indicators.EMA(self.data.close, period=self.params.short_period)
        self.long_ma = bt.indicators.EMA(self.data.close, period=self.params.long_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if not self.position:  # Not in the market
            if self.crossover > 0 and self.rsi < self.params.rsi_oversold:
                self.buy()
        elif self.crossover < 0:  # In the market and short MA crosses below long MA
            self.close()


class BollingerBandStrategy(bt.Strategy):
    """
    A mean-reversion Bollinger Band trading strategy.
    Enters a short position if the price crosses above the upper band.
    Enters a long position if the price crosses below the lower band.
    Exits the position when the price reverts to the middle band.
    """
    params = (
        ('period', 20),
        ('std', 2),
        ('size', 20) 
    )

    def __init__(self):
        """Initialize the strategy and its indicators."""
        self.bollinger = bt.indicators.BollingerBands(
            period=self.p.period, devfactor=self.p.std
        )

    def next(self):
        """Define the trading logic based on Bollinger Bands."""
        if not self.position:  # Not in the market
            if self.data.close > self.bollinger.top:
                self.sell()
            elif self.data.close < self.bollinger.bot:
                self.buy(size=self.p.size)   # Enter long
        else:  # In the market
            if self.position.size > 0:  # We are long
                if self.data.close >= self.bollinger.mid:
                    self.close()  # Close long position
            else:  # We are short
                if self.data.close <= self.bollinger.mid:
                    self.close()  # Close short position


class MomentumStrategy(bt.Strategy):
    """
    A momentum-based strategy that ranks stocks and rebalances the portfolio
    periodically. It uses the first data feed as a market filter (e.g., SPY).
    """
    def __init__(self):
        self.counter = 0
        self.indicators = {}
        self.sorted_data = []
        # The first data feed is the market index (e.g., SPY)
        self.market_index = self.datas[0]
        # The rest of the data feeds are the stocks to trade
        self.stocks = self.datas[1:]

        for stock in self.stocks:
            self.indicators[stock] = {
                'momentum': Momentum(stock.close, period=90),
                'sma100': bt.indicators.SimpleMovingAverage(stock.close, period=100),
                'atr20': bt.indicators.ATR(stock, period=20)
            }

        # Market filter: 200-day SMA on the market index
        self.market_sma = bt.indicators.SimpleMovingAverage(self.market_index.close, period=200)

    def next(self):
        # Rebalance weekly
        if self.counter % 5 == 0:
            self.rebalance_portfolio()
        # Update positions bi-weekly (adjust size)
        if self.counter % 10 == 0:
            self.update_positions()
        self.counter += 1

    def rebalance_portfolio(self):
        # Filter out stocks with insufficient data
        self.sorted_data = [s for s in self.stocks if len(s) > 100]
        # Sort stocks by momentum score in descending order
        self.sorted_data.sort(key=lambda s: self.indicators[s]['momentum'][0], reverse=True)
        num_stocks = len(self.sorted_data)
        
        # Define top quintile for buying
        top_quintile = int(0.2 * num_stocks)

        # Sell logic: close positions for stocks that are no longer top momentum
        # or have fallen below their 100-day SMA.
        for i, stock in enumerate(self.sorted_data):
            if self.getposition(stock).size:
                if i > top_quintile or stock.close[0] < self.indicators[stock]['sma100'][0]:
                    self.close(stock)

        # Buy logic: only open long positions if the market is bullish
        if self.market_index.close[0] < self.market_sma[0]:
            return

        # Buy top 20% momentum stocks
        for stock in self.sorted_data[:top_quintile]:
            if not self.getposition(stock).size:
                # Position sizing based on ATR
                size = (0.01 * self.broker.getvalue()) / self.indicators[stock]['atr20'][0]
                self.buy(stock, size=size)

    def update_positions(self):
        top_quintile = int(0.2 * len(self.sorted_data))
        for stock in self.sorted_data[:top_quintile]:
            if self.getposition(stock).size:
                # Adjust position size based on current portfolio value and ATR
                size = (0.01 * self.broker.getvalue()) / self.indicators[stock]['atr20'][0]
                self.order_target_size(stock, size)
