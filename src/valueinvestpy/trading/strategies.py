import datetime as dt
import pandas as pd

class MeanReversionTrader(object):
    def __init__(
        self, broker, symbol=None, units=1,
        resample_interval='60s', mean_periods=5
    ):
        """
        A trading platform that trades on one side
            based on a mean-reverting algorithm.

        :param broker: Broker object
        :param symbol: A str object recognized by the broker for trading
        :param units: Number of units to trade
        :param resample_interval:
            Frequency for resampling price time series
        :param mean_periods: Number of resampled intervals
            for calculating the average price
        """
        self.broker = self.setup_broker(broker)

        self.resample_interval = resample_interval
        self.mean_periods = mean_periods
        self.symbol = symbol
        self.units = units

        self.df_prices = pd.DataFrame(columns=[symbol])
        self.pnl, self.upnl = 0, 0

        self.bid_price, self.ask_price = 0, 0
        self.position = 0
        self.is_order_pending = False
        self.is_next_signal_cycle = True

    def setup_broker(self, broker):
        broker.on_price_event = self.on_price_event
        broker.on_order_event = self.on_order_event
        broker.on_position_event = self.on_position_event
        return broker

    def on_price_event(self, symbol, bid, ask):
        print(dt.datetime.now(), '[PRICE]', symbol, 'bid:', bid, 'ask:', ask)

        self.bid_price = bid
        self.ask_price = ask
        self.df_prices.loc[pd.Timestamp.now(), symbol] = (bid + ask) / 2.

        self.get_positions()
        self.generate_signals_and_think()

        self.print_state()

    def get_positions(self):
        try:
            self.broker.get_positions()
        except Exception as ex:
            print('get_positions error:', ex)

    def on_order_event(self, symbol, quantity, is_buy, transaction_id, status):
        print(
            dt.datetime.now(), '[ORDER]',
            'transaction_id:', transaction_id,
            'status:', status,
            'symbol:', symbol,
            'quantity:', quantity,
            'is_buy:', is_buy,
        )
        if status == 'FILLED':
            self.is_order_pending = False
            self.is_next_signal_cycle = False

            self.get_positions()  # Update positions before thinking
            self.generate_signals_and_think()

    def on_position_event(self, symbol, is_long, units, upnl, pnl):
        if symbol == self.symbol:
            self.position = abs(units) * (1 if is_long else -1)
            self.pnl = pnl
            self.upnl = upnl
            self.print_state()

    def print_state(self):
        print(
            dt.datetime.now(), self.symbol, self.position_state,
            abs(self.position), 'upnl:', self.upnl, 'pnl:', self.pnl
        )

    @property
    def position_state(self):
        if self.position == 0:
            return 'FLAT'
        if self.position > 0:
            return 'LONG'
        if self.position < 0:
            return 'SHORT'

    def generate_signals_and_think(self):
        df_resampled = self.df_prices.resample(self.resample_interval).ffill().dropna()
        resampled_len = len(df_resampled.index)

        if resampled_len < self.mean_periods:
            print(
                'Insufficient data size to calculate logic. Need',
                self.mean_periods - resampled_len, 'more.'
            )
            return

        mean = df_resampled.tail(self.mean_periods).mean()[self.symbol]

        # Signal flag calculation
        is_signal_buy = mean > self.ask_price
        is_signal_sell = mean < self.bid_price

        print(
            'is_signal_buy:', is_signal_buy,
            'is_signal_sell:', is_signal_sell,
            'average_price: %.5f' % mean,
            'bid:', self.bid_price,
            'ask:', self.ask_price
        )

        self.think(is_signal_buy, is_signal_sell)

    def think(self, is_signal_buy, is_signal_sell):
        if self.is_order_pending:
            return

        if self.position == 0:
            self.think_when_position_flat(is_signal_buy, is_signal_sell)
        elif self.position > 0:
            self.think_when_position_long(is_signal_sell)
        elif self.position < 0: 
            self.think_when_position_short(is_signal_buy)        

    def think_when_position_flat(self, is_signal_buy, is_signal_sell):
        if is_signal_buy and self.is_next_signal_cycle:
            print('Opening position, BUY', 
                  self.symbol, self.units, 'units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, True)
            return

        if is_signal_sell and self.is_next_signal_cycle:
            print('Opening position, SELL', 
                  self.symbol, self.units, 'units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, False)
            return

        if not is_signal_buy and not is_signal_sell:
            self.is_next_signal_cycle = True

    def think_when_position_long(self, is_signal_sell):
        if is_signal_sell:
            print('Closing position, SELL', 
                  self.symbol, self.units, 'units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, False)

    def think_when_position_short(self, is_signal_buy):
        if is_signal_buy:
            print('Closing position, BUY', 
                  self.symbol, self.units, 'units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, True)

    def send_market_order(self, symbol, quantity, is_buy):
        self.broker.send_market_order(symbol, quantity, is_buy)

    def run(self):
        self.broker.stream_prices(symbols=[self.symbol])

class TrendFollowingTrader(MeanReversionTrader):
    def __init__(
        self, *args, long_mean_periods=10,
        buy_threshold=1.0, sell_threshold=1.0, **kwargs
    ):
        super(TrendFollowingTrader, self).__init__(*args, **kwargs)

        self.long_mean_periods = long_mean_periods
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals_and_think(self):
        df_resampled = self.df_prices.resample(self.resample_interval).ffill().dropna()
        resampled_len = len(df_resampled.index)

        if resampled_len < self.long_mean_periods:
            print(
                'Insufficient data size to calculate logic. Need',
                self.mean_periods - resampled_len, 'more.'
            )
            return

        mean_short = df_resampled.tail(self.mean_periods).mean()[self.symbol]
        mean_long = df_resampled.tail(self.long_mean_periods).mean()[self.symbol]
        beta = mean_short / mean_long

        is_signal_buy = beta > self.buy_threshold
        is_signal_sell = beta < self.sell_threshold

        print(
            'is_signal_buy:', is_signal_buy,
            'is_signal_sell:', is_signal_sell,
            'beta: %.2f' % beta,
            'buy_threshold:', self.buy_threshold,
            'sell_threshold:', self.sell_threshold
        )

        self.think(is_signal_buy, is_signal_sell)
