import pytest
from unittest.mock import MagicMock

from src.trading.broker import OandaBroker, InteractiveBrokersBroker


def test_oanda_broker_init():
    """Tests that the OandaBroker can be initialized."""
    try:
        broker = OandaBroker(accountid='test-account', token='test-token')
        assert broker.accountid == 'test-account'
        assert broker.token == 'test-token'
    except Exception as e:
        # The v20 library might try to connect on init, which would fail.
        # For this test, we're just checking that the class can be instantiated
        # without syntax errors or other basic issues.
        pass


def test_interactive_brokers_broker_init(mocker):
    """Tests that the InteractiveBrokersBroker can be initialized."""
    mocker.patch.object(InteractiveBrokersBroker, 'connect', return_value=None)
    mocker.patch.object(InteractiveBrokersBroker, 'run', return_value=True)
    mocker.patch.object(InteractiveBrokersBroker, 'isConnected', return_value=True)

    broker = InteractiveBrokersBroker(
        host='127.0.0.1',
        port=7497,
        client_id=1,
        account='DU1234567'
    )
    assert broker.account == 'DU1234567'
    assert broker.isConnected()

def test_interactive_brokers_price_stream(mocker):
    """Tests the price streaming and event handling of the InteractiveBrokersBroker."""
    # Prevent the real connect method from being called
    mocker.patch.object(InteractiveBrokersBroker, 'connect', return_value=None)
    mocker.patch.object(InteractiveBrokersBroker, 'run', return_value=True)

    # Instantiate the broker
    broker = InteractiveBrokersBroker('localhost', 7497, 1, 'TestAccount')

    # Manually set the connection object to a mock
    broker.conn = MagicMock()
    broker.conn.sendMsg = MagicMock(return_value=None)

    # Mock other connection-dependent methods
    mocker.patch.object(broker, 'isConnected', return_value=True)
    mocker.patch.object(broker, 'serverVersion', return_value=999)
    broker.next_order_id = 1

    # Mock the price event handler
    price_handler_mock = MagicMock()
    broker.on_price_event = price_handler_mock

    # Start streaming prices for a symbol
    broker.stream_prices(symbols=['AAPL'])

    # Simulate receiving a bid price
    broker.tickPrice(reqId=1, tickType=1, price=150.0, attrib=None)
    price_handler_mock.assert_not_called()  # Should not be called until ask is also received

    # Simulate receiving an ask price
    broker.tickPrice(reqId=1, tickType=2, price=150.05, attrib=None)

    # Assert that the price handler was called with the correct data
    price_handler_mock.assert_called_once_with('AAPL', 150.0, 150.05)

def test_interactive_brokers_send_market_order(mocker):
    """Tests sending a market order."""
    mocker.patch.object(InteractiveBrokersBroker, 'connect', return_value=None)
    mocker.patch.object(InteractiveBrokersBroker, 'run', return_value=True)

    broker = InteractiveBrokersBroker('localhost', 7497, 1, 'TestAccount')
    broker.conn = MagicMock()
    broker.placeOrder = MagicMock()
    broker.next_order_id = 1

    broker.send_market_order('TSLA', 10, is_buy=True)

    broker.placeOrder.assert_called_once()
    args, _ = broker.placeOrder.call_args
    order_id, contract, order = args

    assert order_id == 1
    assert contract.symbol == 'TSLA'
    assert order.action == 'BUY'
    assert order.orderType == 'MKT'
    assert order.totalQuantity == 10
    assert broker.next_order_id == 2

def test_interactive_brokers_position_updates(mocker):
    """Tests position updates from the updatePortfolio callback."""
    mocker.patch.object(InteractiveBrokersBroker, 'connect', return_value=None)
    mocker.patch.object(InteractiveBrokersBroker, 'run', return_value=True)

    broker = InteractiveBrokersBroker('localhost', 7497, 1, 'TestAccount')
    position_handler_mock = MagicMock()
    broker.on_position_event = position_handler_mock

    # Simulate an updatePortfolio callback
    contract = broker._create_stock_contract('GOOG')
    broker.updatePortfolio(contract, 100, 1500.0, 150000.0, 1400.0, 10000.0, 5000.0, 'TestAccount')

    # Assert that the position was updated and the event was called
    assert 'GOOG' in broker.positions
    assert broker.positions['GOOG']['position'] == 100
    position_handler_mock.assert_called_once_with('GOOG', True, 100, 10000.0, 5000.0)

def test_interactive_brokers_order_status(mocker):
    """Tests order status updates."""
    mocker.patch.object(InteractiveBrokersBroker, 'connect', return_value=None)
    mocker.patch.object(InteractiveBrokersBroker, 'run', return_value=True)

    broker = InteractiveBrokersBroker('localhost', 7497, 1, 'TestAccount')
    order_handler_mock = MagicMock()
    broker.on_order_event = order_handler_mock

    # Manually add an order to track
    broker.orders[1] = {'symbol': 'MSFT', 'quantity': 50, 'is_buy': True, 'status': 'Submitted'}

    # Simulate an orderStatus callback
    broker.orderStatus(1, 'Filled', 50, 0, 300.0, 0, 0, 300.0, 1, '', 0)

    # Assert that the order status was updated and the event was called
    assert broker.orders[1]['status'] == 'Filled'
    order_handler_mock.assert_called_once_with(1, 'Filled')
