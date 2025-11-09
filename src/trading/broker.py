from abc import abstractmethod
import threading
import time

import v20
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

class Broker(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.__price_event_handler = None
        self.__order_event_handler = None
        self.__position_event_handler = None

    @property
    def on_price_event(self):
        """
        Listeners will receive:
        symbol, bid, ask
        """
        return self.__price_event_handler

    @on_price_event.setter
    def on_price_event(self, event_handler):
        self.__price_event_handler = event_handler

    @property
    def on_order_event(self):
        """
        Listeners will receive:
        transaction_id
        """
        return self.__order_event_handler

    @on_order_event.setter
    def on_order_event(self, event_handler):
        self.__order_event_handler = event_handler

    @property
    def on_position_event(self):
        """
        Listeners will receive:
        symbol, is_long, units, unrealized_pnl, pnl
        """
        return self.__position_event_handler

    @on_position_event.setter
    def on_position_event(self, event_handler):
        self.__position_event_handler = event_handler

    @abstractmethod
    def get_prices(self, symbols=[]):
        """
        Query market prices from a broker
        :param symbols: list of symbols recognized by your broker
        """
        raise NotImplementedError('Method is required!')

    @abstractmethod
    def stream_prices(self, symbols=[]):
        """
        Continuously stream prices from a broker.
        :param symbols: list of symbols recognized by your broker
        """
        raise NotImplementedError('Method is required!')

    @abstractmethod
    def send_market_order(self, symbol, quantity, is_buy):
        raise NotImplementedError('Method is required!')

    @abstractmethod
    def get_positions(self):
        """
        Query account positions from a broker
        """
        raise NotImplementedError('Method is required!')

class InteractiveBrokersBroker(Broker, EWrapper, EClient):
    def __init__(self, host, port, client_id, account):
        Broker.__init__(self, host, port)
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.account = account
        self.next_order_id = None
        self.req_id_to_symbol = {}
        self.prices = {}
        self.positions = {}
        self.orders = {}
        self.account_values = {}

        self.connect(host, port, client_id)

        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()

        time.sleep(1) # Allow time for connection to be established

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_order_id = orderId

    def get_prices(self, symbols=[]):
        # IB's get_prices equivalent is a snapshot from stream_prices
        # This will be handled by stream_prices and a mechanism to stop it
        pass

    def stream_prices(self, symbols=[]):
        for symbol in symbols:
            req_id = self.next_order_id
            self.req_id_to_symbol[req_id] = symbol
            self.prices[symbol] = {'bid': None, 'ask': None}
            contract = self._create_stock_contract(symbol)
            self.reqMktData(req_id, contract, "", False, False, [])
            self.next_order_id += 1

    def tickPrice(self, reqId, tickType, price, attrib):
        super().tickPrice(reqId, tickType, price, attrib)
        if tickType not in [1, 2] or price <= 0:
            return

        symbol = self.req_id_to_symbol.get(reqId)
        if not symbol:
            return

        if tickType == 1:  # Bid
            self.prices[symbol]['bid'] = price
        elif tickType == 2:  # Ask
            self.prices[symbol]['ask'] = price

        bid = self.prices[symbol]['bid']
        ask = self.prices[symbol]['ask']

        if bid and ask and self.on_price_event:
            self.on_price_event(symbol, bid, ask)

    def send_market_order(self, symbol, quantity, is_buy):
        contract = self._create_stock_contract(symbol)
        order = Order()
        order.action = "BUY" if is_buy else "SELL"
        order.orderType = "MKT"
        order.totalQuantity = abs(quantity)

        order_id = self.next_order_id
        self.orders[order_id] = {
            "symbol": symbol,
            "quantity": quantity,
            "is_buy": is_buy,
            "status": "Submitted"
        }
        self.placeOrder(order_id, contract, order)
        self.next_order_id += 1

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        
        order_details = self.orders.get(orderId)
        if not order_details:
            return

        order_details['status'] = status
        if self.on_order_event:
            # The base on_order_event needs to be updated to handle these params
            self.on_order_event(orderId, status)

    def get_positions(self):
        self.reqAccountUpdates(True, self.account)

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        super().updateAccountValue(key, val, currency, accountName)
        self.account_values[key] = val

    def updatePortfolio(self, contract: Contract, position: float, marketPrice: float, marketValue: float, averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str):
        super().updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName)
        
        symbol = contract.symbol
        self.positions[symbol] = {
            'position': position,
            'marketPrice': marketPrice,
            'marketValue': marketValue,
            'averageCost': averageCost,
            'unrealizedPNL': unrealizedPNL,
            'realizedPNL': realizedPNL
        }

        if self.on_position_event:
            is_long = position > 0 if position != 0 else None
            self.on_position_event(symbol, is_long, abs(position), unrealizedPNL, realizedPNL)

    def _create_stock_contract(self, symbol, sec_type='STK', currency='USD', exchange='SMART'):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.currency = currency
        contract.exchange = exchange
        return contract


class OandaBroker(Broker):
    PRACTICE_API_HOST = 'api-fxpractice.oanda.com'
    PRACTICE_STREAM_HOST = 'stream-fxpractice.oanda.com'

    LIVE_API_HOST = 'api-fxtrade.oanda.com'
    LIVE_STREAM_HOST = 'stream-fxtrade.oanda.com'

    PORT = '443'

    def __init__(self, accountid, token, is_live=False):
        if is_live:
            host = self.LIVE_API_HOST
            stream_host = self.LIVE_STREAM_HOST
        else:
            host = self.PRACTICE_API_HOST
            stream_host = self.PRACTICE_STREAM_HOST

        super(OandaBroker, self).__init__(host, self.PORT)

        self.accountid = accountid
        self.token = token

        self.api = v20.Context(host, self.port, token=token)
        self.stream_api = v20.Context(stream_host, self.port, token=token)

    def get_prices(self, symbols=[]):
        response = self.api.pricing.get(
            self.accountid,
            instruments=",".join(symbols),
            snapshot=True,
            includeUnitsAvailable=False
        )
        body = response.body
        prices = body.get('prices', [])
        for price in prices:
            self.process_price(price)

    def process_price(self, price):
        symbol = price.instrument

        if not symbol:
            print('Price symbol is empty!')
            return

        bids = price.bids or []
        price_bucket_bid = bids[0] if bids and len(bids) > 0 else None
        bid = price_bucket_bid.price if price_bucket_bid else 0

        asks = price.asks or []
        price_bucket_ask = asks[0] if asks and len(asks) > 0 else None
        ask = price_bucket_ask.price if price_bucket_ask else 0

        self.on_price_event(symbol, bid, ask)

    def stream_prices(self, symbols=[]):
        response = self.stream_api.pricing.stream(
            self.accountid,
            instruments=",".join(symbols),
            snapshot=True
        )

        for msg_type, msg in response.parts():
            if msg_type == "pricing.Heartbeat":
                continue
            elif msg_type == "pricing.ClientPrice":
                self.process_price(msg)

    def send_market_order(self, symbol, quantity, is_buy):
        response = self.api.order.market(
            self.accountid,
            units=abs(quantity) * (1 if is_buy else -1),
            instrument=symbol,
            type='MARKET',
        )
        if response.status != 201:
            self.on_order_event(symbol, quantity, is_buy, None, 'NOT_FILLED')
            return

        body = response.body
        if 'orderCancelTransaction' in body:
            self.on_order_event(symbol, quantity, is_buy, None, 'NOT_FILLED')
            return

        transaction_id = body.get('lastTransactionID', None)
        self.on_order_event(symbol, quantity, is_buy, transaction_id, 'FILLED')

    def get_positions(self):
        response = self.api.position.list(self.accountid)
        body = response.body
        positions = body.get('positions', [])
        for position in positions:
            symbol = position.instrument
            unrealized_pnl = position.unrealizedPL
            pnl = position.pl
            long = position.long
            short = position.short

            if short.units:
                self.on_position_event(
                    symbol, False, short.units, unrealized_pnl, pnl)
            elif long.units:
                self.on_position_event(
                    symbol, True, long.units, unrealized_pnl, pnl)
            else:
                self.on_position_event(
                    symbol, None, 0, unrealized_pnl, pnl)
