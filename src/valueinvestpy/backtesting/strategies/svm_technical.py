import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from valueinvestpy.machine_learning.feature_engineering import construct_technical_indicator_features
from valueinvestpy.technical_analysis.oscillators import rsi
from valueinvestpy.technical_analysis.moving_averages import sma

class SVMTechnicalStrategy(bt.Strategy):
    """
    A backtrader strategy that uses an SVM model trained on technical indicator
    features (SMA trend and RSI) to predict price direction.
    """
    params = (
        ('ma_period', 60),
        ('rsi_period', 14),
        ('C', 1.0),
        ('gamma', 'scale'),
    )

    def __init__(self):
        """
        Initializes the strategy by training an SVM model on the historical data.
        """
        df = self._convert_data_to_dataframe()
        df_features = construct_technical_indicator_features(
            df, 
            ma_period=self.p.ma_period, 
            rsi_period=self.p.rsi_period
        )
        df_features.dropna(inplace=True)

        X = df_features[['trend', 'RSI']]
        y = df_features['direction']

        self.model = SVC(C=self.p.C, gamma=self.p.gamma)
        self.model.fit(X, y)

        self.sma = sma(self.datas[0].close, window=self.p.ma_period)
        self.rsi = rsi(self.datas[0].close, window=self.p.rsi_period)

    def next(self):
        """
        Executes on each bar, generates features and a prediction, and places an order.
        """
        if self.order:
            return

        # Ensure we have enough data to calculate indicators
        if len(self) > self.p.ma_period:
            # Generate features for the current time step
            current_trend = (self.datas[0].open[0] - self.sma[0]) * 100
            current_rsi = self.rsi[0] / 100
            features_array = np.array([current_trend, current_rsi]).reshape(1, -1)

            prediction = self.model.predict(features_array)

            if self.position.size == 0:
                if prediction == 1:
                    self.log('BUY CREATE, %.2f' % self.datas[0].close[0])
                    self.order = self.buy()
            else:
                if prediction == -1:
                    self.log('SELL CREATE, %.2f' % self.datas[0].close[0])
                    self.order = self.sell()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def stop(self):
        self.log('Ending Value %.2f' % (self.broker.getvalue()))

    def _convert_data_to_dataframe(self):
        data = self.datas[0]
        dates = [bt.num2date(x) for x in data.datetime.array]
        df = pd.DataFrame(index=dates)
        df['Open'] = data.open.array
        df['Close'] = data.close.array
        return df
