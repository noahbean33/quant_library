import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from valueinvestpy.machine_learning.feature_engineering import construct_lagged_return_features

class LogisticRegressionStrategy(bt.Strategy):
    """
    A backtrader strategy that uses a logistic regression model to predict
    the direction of the next day's price change.
    """
    params = (
        ('lags', 2),
    )

    def __init__(self):
        """
        Initializes the strategy by training a logistic regression model on the
        entire historical dataset provided to the backtest.
        """
        # Convert the backtrader data feed to a pandas DataFrame for feature engineering
        df = self._convert_data_to_dataframe()

        # Construct features using the imported utility function
        df_features = construct_lagged_return_features(df, lags=self.p.lags)
        df_features.dropna(inplace=True)

        # Prepare the data for training the model
        feature_columns = [f'Lag{i+1}' for i in range(self.p.lags)]
        X = df_features[feature_columns]
        y = df_features['Direction']

        # Train the logistic regression model
        self.model = LogisticRegression()
        self.model.fit(X, y)

        # Keep a reference to the close price data feed
        self.dataclose = self.datas[0].close

        # Track pending orders to avoid double submissions
        self.order = None

    def next(self):
        """
        Executes on each bar of data, generates a prediction, and places an order.
        """
        # Check if an order is pending
        if self.order:
            return

        # We need enough data to construct the features
        if len(self) > self.p.lags:
            # Construct the feature set for the current bar
            # The features are based on lagged percentage changes
            current_features = []
            for i in range(1, self.p.lags + 1):
                lag_pct_change = (self.dataclose[-i] - self.dataclose[-(i+1)]) / self.dataclose[-(i+1)] * 100
                current_features.append(lag_pct_change)
            
            features_array = np.array(current_features).reshape(1, -1)

            # Predict the direction for the next bar
            prediction = self.model.predict(features_array)

            # Trading logic based on the prediction
            if self.position.size == 0:  # No position open
                if prediction == 1:
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    self.order = self.buy()
            else:  # In a long position
                if prediction == -1:
                    self.log('SELL CREATE, %.2f' % self.dataclose[0])
                    self.order = self.sell()

    def log(self, txt, dt=None):
        """ Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def stop(self):
        """ Called when the backtest is complete """
        self.log('Ending Value %.2f' % (self.broker.getvalue()))

    def _convert_data_to_dataframe(self):
        """
        Helper to convert the backtrader data feed to a pandas DataFrame.
        """
        data = self.datas[0]
        dates = [bt.num2date(x) for x in data.datetime.array]
        df = pd.DataFrame(index=dates)
        df['Close'] = data.close.array
        return df
