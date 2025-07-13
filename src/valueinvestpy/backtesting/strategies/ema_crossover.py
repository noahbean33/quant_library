import backtrader as bt

class EMACrossoverStrategy(bt.Strategy):
    """
    A classic strategy that buys when a short-term Exponential Moving Average (EMA)
    crosses above a long-term EMA, and sells when it crosses below.
    """
    params = (
        ('short_period', 30),
        ('long_period', 100),
        ('printlog', False),
    )

    def __init__(self):
        """Initializes the strategy indicators."""
        self.dataclose = self.datas[0].close

        # Initialize the EMAs
        self.short_ema = bt.indicators.ExponentialMovingAverage(
            self.dataclose, period=self.p.short_period
        )
        self.long_ema = bt.indicators.ExponentialMovingAverage(
            self.dataclose, period=self.p.long_period
        )

        # Use the Crossover indicator to generate signals
        self.crossover = bt.indicators.CrossOver(self.short_ema, self.long_ema)

    def log(self, txt, dt=None, doprint=False):
        """Logging function for this strategy."""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def next(self):
        """Executes on each bar of data to check for crossover signals."""
        # If we are not in the market and a bullish crossover occurs
        if not self.position:
            if self.crossover > 0:  # short_ema crosses above long_ema
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()
        # If we are in the market and a bearish crossover occurs
        elif self.crossover < 0:  # short_ema crosses below long_ema
            self.log('SELL CREATE, %.2f' % self.dataclose[0])
            self.close()

    def stop(self):
        """Called when the backtest is complete."""
        self.log(f'(Short Period {self.p.short_period}, Long Period {self.p.long_period}) Ending Value {self.broker.getvalue():.2f}', doprint=True)
