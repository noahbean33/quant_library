import backtrader as bt
import numpy as np
import pandas as pd
import statsmodels.api as sm

class PairsTradingStrategy(bt.Strategy):
    """
    A pairs trading strategy that identifies cointegrated assets and trades on the
    reversion of their spread to the mean.
    """
    params = (
        ('lookback', 252),  # Lookback period for regression and z-score
        ('entry_threshold', 2.0),  # Z-score threshold for entering a trade
        ('exit_threshold', 0.5),   # Z-score threshold for exiting a trade
    )

    def __init__(self):
        if len(self.datas) != 2:
            raise ValueError("PairsTradingStrategy requires exactly two data feeds.")

        self.data1 = self.datas[0]
        self.data2 = self.datas[1]
        self.hedge_ratio = 0

    def start(self):
        # Ensure there's enough data to calculate the initial hedge ratio
        if len(self.data1) < self.p.lookback or len(self.data2) < self.p.lookback:
            return

        # Perform initial regression to find the hedge ratio
        self.update_hedge_ratio()

    def next(self):
        if len(self.data1) < self.p.lookback or len(self.data2) < self.p.lookback:
            return  # Not enough data yet

        # Recalculate hedge ratio periodically (e.g., every 20 bars)
        if len(self.data1) % 20 == 0:
            self.update_hedge_ratio()

        # Calculate the current spread
        spread = self.data1.close[0] - self.hedge_ratio * self.data2.close[0]

        # Calculate the z-score of the spread
        spread_series = pd.Series(self.data1.close.get(size=self.p.lookback)) - \
                        self.hedge_ratio * pd.Series(self.data2.close.get(size=self.p.lookback))
        mean = spread_series.mean()
        std = spread_series.std()
        z_score = (spread - mean) / std

        # Trading logic based on z-score
        if not self.position:
            if z_score > self.p.entry_threshold:
                # Spread is too high, short the spread (sell data1, buy data2)
                self.sell(data=self.data1)
                self.buy(data=self.data2)
            elif z_score < -self.p.entry_threshold:
                # Spread is too low, long the spread (buy data1, sell data2)
                self.buy(data=self.data1)
                self.sell(data=self.data2)
        else:
            # Exit logic: close position when spread reverts to the mean
            if abs(z_score) < self.p.exit_threshold:
                self.close(data=self.data1)
                self.close(data=self.data2)

    def update_hedge_ratio(self):
        """Recalculates the hedge ratio using linear regression."""
        y = np.array(self.data1.close.get(size=self.p.lookback))
        x = np.array(self.data2.close.get(size=self.p.lookback))
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        self.hedge_ratio = model.params[1]
