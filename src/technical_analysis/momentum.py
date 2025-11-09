import backtrader as bt
import numpy as np
from scipy.stats import linregress


class Momentum(bt.Indicator):
    """
    Calculates momentum based on the annualized slope of a log-linear regression,
    adjusted by the R-squared value.
    """
    lines = ('momentum_trend',)
    params = (('period', 90),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        """Calculates the momentum value for the current bar."""
        returns = np.log(self.data.get(size=self.params.period))
        x = np.arange(len(returns))
        beta, _, rvalue, _, _ = linregress(x, returns)
        annualized = (1 + beta) ** 252
        self.lines.momentum_trend[0] = annualized * (rvalue ** 2)
