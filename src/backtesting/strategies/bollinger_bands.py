import backtrader as bt

class BollingerBandStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('std', 2),
        ('size', 20)
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(period=self.p.period,
                                                      devfactor=self.p.std)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.bollinger.lines.top:
                self.sell(size=self.p.size)
            if self.data.close[0] < self.bollinger.lines.bot:
                self.buy(size=self.p.size)
        else:
            if self.position.size > 0:
                self.sell(exectype=bt.Order.Limit, price=self.bollinger.lines.mid[0],
                          size=self.p.size)
            else:
                self.buy(exectype=bt.Order.Limit, price=self.bollinger.lines.mid[0],
                         size=self.p.size)
