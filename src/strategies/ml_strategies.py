import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from scipy.stats import linregress

def _construct_features(data: pd.DataFrame, lags: int = 2) -> (pd.DataFrame, pd.Series):
    """
    Constructs features and target variable from historical price data.
    Returns X (features) and y (target).
    """
    df = data.copy()
    
    # Create lagged returns as features
    for i in range(1, lags + 1):
        df[f'Lag_{i}'] = df['Close'].pct_change(i)

    # Create the target variable: +1 for up day, -1 for down day
    df['Direction'] = np.where(df['Close'].diff() > 0, 1, -1)
    
    df = df.dropna()
    
    X = df[[f'Lag_{i}' for i in range(1, lags + 1)]]
    y = df['Direction']
    
    return X, y

class LogisticRegressionStrategy(bt.Strategy):
    """
    A trading strategy that uses a Logistic Regression model to predict
    the direction of the next day's price movement.
    """
    params = (('lags', 2), ('min_data_points', 50),)

    def __init__(self):
        """
        Initialize the strategy. We will train the model in the start() method.
        """
        self.model = None
        # Keep track of the current features to make predictions
        # We must use backtrader's indicators, not pandas functions, on data lines.
        self.lags = [bt.indicators.PercentChange(period=i) for i in range(1, self.p.lags + 1)]

    def start(self):
        """
        Called once all data has been loaded. This is where we'll train the model.
        """
        if len(self.data) < self.p.min_data_points:
            # Not enough data to train the model, do nothing.
            return

        # Convert the backtrader data feed to a pandas DataFrame
        data_df = pd.DataFrame({
            'Open': self.data0.open.get(size=len(self.data0)),
            'High': self.data0.high.get(size=len(self.data0)),
            'Low': self.data0.low.get(size=len(self.data0)),
            'Close': self.data0.close.get(size=len(self.data0)),
            'Volume': self.data0.volume.get(size=len(self.data0)),
        }, index=pd.to_datetime([bt.num2date(x) for x in self.data0.datetime.get(size=len(self.data0))]))

        # Construct features and train the model
        X, y = _construct_features(data_df, lags=self.p.lags)
        
        if len(X) > 0:
            self.model = LogisticRegression()
            self.model.fit(X, y)

    def next(self):
        """
        Execute the strategy on each new bar of data.
        """
        # If the model hasn't been trained, we cannot trade.
        if not self.model:
            return

        # Check if we have an open position
        if self.position:
            return

        try:
            # Get the feature values for the current bar
            current_features = [lag[0] for lag in self.lags]
            
            # Reshape for a single prediction
            features_for_prediction = np.array(current_features).reshape(1, -1)

            # Predict the direction for the *next* bar
            prediction = self.model.predict(features_for_prediction)

            if prediction == 1:
                self.buy()
            elif prediction == -1:
                self.sell()

        except (IndexError, ValueError):
            # Not enough data to create features yet, or features contain NaNs
            pass


class SVMCombinedStrategy(bt.Strategy):
    """
    An SVM-based strategy that uses a combination of technical indicators (SMA, RSI)
    as features. It performs a grid search to find the optimal hyperparameters
    for the SVM model before trading.
    """
    params = (
        ('ma_period', 60),
        ('rsi_period', 14),
        ('min_data_points', 100), # Needs enough data for training and indicators
    )

    def __init__(self):
        self.model = None
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.ma_period)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)

    def start(self):
        if len(self.data) < self.p.min_data_points:
            return

        # Convert data to pandas DataFrame for feature engineering and training
        data_df = pd.DataFrame({
            'Open': self.data.open.get(size=len(self.data)),
            'Close': self.data.close.get(size=len(self.data)),
        }, index=pd.to_datetime([bt.num2date(x) for x in self.data.datetime.get(size=len(self.data))]))

        # Feature Engineering
        sma = data_df['Close'].rolling(window=self.p.ma_period).mean()
        data_df['trend'] = (data_df['Open'] - sma)
        
        # RSI calculation
        move = data_df['Close'] - data_df['Close'].shift(1)
        up = np.where(move > 0, move, 0)
        down = np.where(move < 0, move, 0)
        avg_gain = pd.Series(up).rolling(self.p.rsi_period).mean()
        avg_loss = pd.Series(down).abs().rolling(self.p.rsi_period).mean()
        rs = avg_gain / avg_loss
        data_df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        # Target variable
        data_df['direction'] = np.where(data_df['Close'] - data_df['Open'] > 0, 1, -1)
        data_df.dropna(inplace=True)

        if len(data_df) < 20: # Ensure we have enough data after dropping NaNs
            return

        X = data_df[['trend', 'RSI']]
        y = data_df['direction']

        # Hyperparameter tuning with Grid Search
        parameters = {'gamma': [1, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]}
        grid = list(ParameterGrid(parameters))
        best_accuracy = 0
        best_params = None

        for p in grid:
            svm = SVC(C=p['C'], gamma=p['gamma'])
            svm.fit(X, y) # In a real scenario, you'd use a train/validation split
            accuracy = svm.score(X, y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = p

        # Train final model with best parameters
        if best_params:
            self.model = SVC(C=best_params['C'], gamma=best_params['gamma'])
            self.model.fit(X, y)

    def next(self):
        # Do not proceed if the model is not trained
        if not self.model:
            return

        try:
            # Get features for the current bar
            trend_feature = self.data.open[0] - self.sma[0]
            rsi_feature = self.rsi[0]
            features = np.array([trend_feature, rsi_feature]).reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(features)

            # Trading logic
            if not self.position:  # Not in the market
                if prediction == 1:
                    self.buy()
            elif prediction == -1:  # In the market
                self.close()

        except (IndexError, ValueError):
            # Not enough data for indicators yet, just pass
            pass

class SVMTuningStrategy(bt.Strategy):
    """
    An SVM-based strategy that uses lagged returns as features and performs
    hyperparameter tuning for the SVM model using a grid search.
    """
    params = (
        ('lags', 2),
        ('min_data_points', 50),
    )

    def __init__(self):
        self.model = None
        self.lags = [bt.indicators.PercentChange(period=i) for i in range(1, self.p.lags + 1)]

    def start(self):
        if len(self.data) < self.p.min_data_points:
            return

        # Prepare data using the helper function
        data_df = pd.DataFrame({
            'Open': self.data.open.get(size=len(self.data)),
            'High': self.data.high.get(size=len(self.data)),
            'Low': self.data.low.get(size=len(self.data)),
            'Close': self.data.close.get(size=len(self.data)),
            'Volume': self.data.volume.get(size=len(self.data)),
        }, index=pd.to_datetime([bt.num2date(x) for x in self.data.datetime.get(size=len(self.data))]))
        
        X, y = _construct_features(data_df, lags=self.p.lags)
        
        if len(X) < 20: # Need enough data for tuning
            return

        # Hyperparameter tuning with Grid Search
        parameters = {'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}
        grid = list(ParameterGrid(parameters))
        best_accuracy = 0
        best_params = None

        # For simplicity in backtesting, we train on the whole available history.
        for p in grid:
            svm = SVC(C=p['C'], gamma=p['gamma'])
            svm.fit(X, y)
            accuracy = svm.score(X, y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = p
        
        # Train final model with best parameters on the full dataset
        if best_params:
            self.model = SVC(C=best_params['C'], gamma=best_params['gamma'], probability=True)
            self.model.fit(X, y)

    def next(self):
        if not self.model or self.position:
            return

        try:
            current_features = [lag[0] for lag in self.lags]
            features_for_prediction = np.array(current_features).reshape(1, -1)
            prediction = self.model.predict(features_for_prediction)
            
            if prediction == 1:
                self.buy()
        except (IndexError, ValueError):
            # Not enough data for features
            pass


class SVMStrategy(bt.Strategy):
    """
    A trading strategy that uses a Support Vector Machine (SVM) to predict
    the direction of the next day's price movement.
    """
    params = (('lags', 2), ('min_data_points', 50),)

    def __init__(self):
        """
        Initialize the strategy. We will train the model in the start() method.
        """
        self.model = None
        self.lags = [bt.indicators.PercentChange(period=i) for i in range(1, self.p.lags + 1)]

    def start(self):
        """
        Called once all data has been loaded. This is where we'll train the model.
        """
        if len(self.data) < self.p.min_data_points:
            return

        data_df = pd.DataFrame({
            'Open': self.data0.open.get(size=len(self.data0)),
            'High': self.data0.high.get(size=len(self.data0)),
            'Low': self.data0.low.get(size=len(self.data0)),
            'Close': self.data0.close.get(size=len(self.data0)),
            'Volume': self.data0.volume.get(size=len(self.data0)),
        }, index=pd.to_datetime([bt.num2date(x) for x in self.data0.datetime.get(size=len(self.data0))]))

        X, y = _construct_features(data_df, lags=self.p.lags)
        
        if len(X) > 0:
            self.model = SVC()
            self.model.fit(X, y)

    def next(self):
        """
        Execute the strategy on each new bar of data.
        """
        if not self.model or self.position:
            return

        try:
            current_features = [lag[0] for lag in self.lags]
            features_for_prediction = np.array(current_features).reshape(1, -1)
            prediction = self.model.predict(features_for_prediction)

            if prediction == 1:
                self.buy()
            elif prediction == -1:
                self.sell()

        except (IndexError, ValueError):
            pass

def _calculate_regression_momentum(price_series: pd.Series) -> float:
    """
    Calculates the momentum of a stock based on the slope of the
    log-price regression, annualized and weighted by R-squared.
    """
    if len(price_series) < 10 or price_series.isnull().any():
        return -np.inf

    log_data = np.log(price_series)
    x_data = np.arange(len(log_data))
    
    try:
        beta, _, rvalue, _, _ = linregress(x_data, log_data)
    except ValueError:
        return -np.inf

    return ((1 + beta) ** 252) * (rvalue ** 2)

class RegressionMomentumStrategy(bt.Strategy):
    """
    A cross-sectional momentum strategy that ranks stocks and rebalances
    the portfolio periodically to hold the top performers.
    """
    params = (
        ('lookback', 90),
        ('rebalance_days', 21),
        ('num_positions', 5),
    )

    def __init__(self):
        """Initialize the strategy."""
        self.rebalance_day_counter = 0

    def next(self):
        """
        The main trading logic, tied to a rebalancing schedule.
        """
        self.rebalance_day_counter += 1
        if self.rebalance_day_counter >= self.p.rebalance_days:
            self.rebalance()
            self.rebalance_day_counter = 0

    def rebalance(self):
        """
        Calculates momentum for all assets, ranks them, and adjusts the portfolio.
        """
        momentums = {}
        for data in self.datas:
            if len(data) < self.p.lookback:
                continue

            close_prices = pd.Series(data.close.get(size=self.p.lookback))
            momentums[data._name] = _calculate_regression_momentum(close_prices)

        if not momentums:
            return

        ranked_stocks = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
        top_stocks = {item[0] for item in ranked_stocks[:self.p.num_positions]}
        
        # Sell stocks no longer in the top tier
        for data in self.datas:
            if data._name not in top_stocks and self.getposition(data).size > 0:
                self.close(data=data)
        
        # Buy stocks in the top tier
        if top_stocks:
            target_weight = 1.0 / len(top_stocks)
            for data in self.datas:
                if data._name in top_stocks:
                    self.order_target_percent(data=data, target=target_weight)

class CrossSectionalMeanReversion(bt.Strategy):

    def __init__(self):
        self.stock_data = self.datas

    def prenext(self):
        self.next()

    def next(self):

        stock_returns = np.zeros(len(self.stock_data))

        # calculate the last daily returns
        for index, stock in enumerate(self.stock_data):
            stock_returns[index] = (stock.close[0] - stock.close[-1]) / stock.close[-1]

        # average return of the market (SP500)
        market_return = np.mean(stock_returns)
        # weights
        weights = -(stock_returns-market_return)
        weights = weights / np.sum(np.abs(weights))

        # we can update our positions based on w weights
        for index, stock in enumerate(self.stock_data):
            self.order_target_percent(stock, target=weights[index])

class RegressionMomentumIndicator(bt.Indicator):
    """
    Calculates the momentum based on the annualized slope of a log-price
    regression, weighted by the R-squared value.
    """
    lines = ('momentum_trend',)
    params = (('period', 90),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        returns = np.log(self.data.get(size=self.params.period))
        x = np.arange(len(returns))
        beta, _, rvalue, _, _ = linregress(x, returns)
        annualized = (1 + beta) ** 252
        self.lines.momentum_trend[0] = annualized * rvalue ** 2

class TimeRebalancedMomentumStrategy(bt.Strategy):
    """
    A time-rebalanced momentum strategy that uses a market filter and
    volatility-based position sizing.
    """

    def __init__(self):
        self.counter = 0
        self.indicators = {}
        self.sorted_data = []
        self.market_index = self.datas[0]
        self.stocks = self.datas[1:]

        for stock in self.stocks:
            self.indicators[stock] = {
                'momentum': RegressionMomentumIndicator(stock.close, period=90),
                'sma100': bt.indicators.SimpleMovingAverage(stock.close, period=100),
                'atr20': bt.indicators.ATR(stock, period=20)
            }

        self.market_sma200 = bt.indicators.MovingAverageSimple(self.market_index.close, period=200)

    def prenext(self):
        self.next()

    def next(self):
        if self.counter % 5 == 0:
            self.rebalance_portfolio()
        if self.counter % 10 == 0:
            self.update_positions()
        self.counter += 1

    def rebalance_portfolio(self):
        self.sorted_data = list(filter(lambda d: len(d) > 100, self.stocks))
        self.sorted_data.sort(key=lambda d: self.indicators[d]['momentum'][0], reverse=True)
        num_stocks = len(self.sorted_data)
        top_quintile_cutoff = int(0.2 * num_stocks)

        for i, stock in enumerate(self.sorted_data):
            if self.getposition(stock).size:
                if i > top_quintile_cutoff or stock.close[0] < self.indicators[stock]['sma100'][0]:
                    self.close(stock)

        if self.market_index.close[0] < self.market_sma200[0]:
            return

        for stock in self.sorted_data[:top_quintile_cutoff]:
            if not self.getposition(stock).size:
                value = self.broker.get_value()
                size = (value * 0.001) / self.indicators[stock]['atr20'][0]
                self.buy(stock, size=size)

    def update_positions(self):
        num_stocks = len(self.sorted_data)
        top_quintile_cutoff = int(0.2 * num_stocks)

        for stock in self.sorted_data[:top_quintile_cutoff]:
            if self.getposition(stock).size:
                value = self.broker.get_value()
                size = (value * 0.001) / self.indicators[stock]['atr20'][0]
                self.order_target_size(stock, size)


