import backtrader as bt

class EMARSICrossoverStrategy(bt.Strategy):
    """
    A strategy that combines an EMA crossover with an RSI filter.
    It buys when a short-term EMA crosses above a long-term EMA, but only if
    the RSI is in an oversold region. It sells when the short-term EMA
    crosses back below the long-term EMA.
    """
    params = (
        ('short_period', 40),
        ('long_period', 150),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
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

        # Initialize the RSI
        self.rsi = bt.indicators.RSI(
            self.dataclose, period=self.p.rsi_period
        )

        # Use the Crossover indicator to generate signals
        self.crossover = bt.indicators.CrossOver(self.short_ema, self.long_ema)

    def log(self, txt, dt=None, doprint=False):
        """Logging function for this strategy."""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def next(self):
        """Executes on each bar of data to check for signals."""
        # If we are not in the market, check for a buy signal
        if not self.position:
            # Bullish crossover AND oversold RSI
            if self.crossover > 0 and self.rsi < self.p.rsi_oversold:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()
        # If we are in the market, check for a sell signal
        elif self.crossover < 0:  # Bearish crossover
            self.log('SELL CREATE, %.2f' % self.dataclose[0])
            self.close()

    def stop(self):
        """Called when the backtest is complete."""
        self.log(f'(Short {self.p.short_period}, Long {self.p.long_period}, RSI {self.p.rsi_period}) Ending Value {self.broker.getvalue():.2f}', doprint=True)
