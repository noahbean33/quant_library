import backtrader as bt
import pandas as pd


from typing import Union

def run_backtest(strategy, data: Union[pd.DataFrame, dict[str, pd.DataFrame]], cash: float = 100000.0, commission: float = 0.001):
    """
    Runs an event-driven backtest using backtrader.

    Args:
        strategy: The backtrader strategy class to test.
        data (Union[pd.DataFrame, dict[str, pd.DataFrame]]): 
            For single-asset backtests, a pandas DataFrame with OHLCV columns.
            For multi-asset backtests, a dictionary where keys are asset names
            and values are their corresponding OHLCV DataFrames.
        cash (float, optional): Initial cash for the portfolio. Defaults to 100000.0.
        commission (float, optional): Broker commission fee. Defaults to 0.001.

    Returns:
        float: The final value of the portfolio.
    """
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(strategy)

    # Add data feeds
    if isinstance(data, pd.DataFrame):
        # Single data feed for backward compatibility
        data_feed = bt.feeds.PandasData(dataname=data, name='single_asset')
        cerebro.adddata(data_feed)
    else:
        # Multiple data feeds for cross-sectional strategies
        for name, df in data.items():
            data_feed = bt.feeds.PandasData(dataname=df, name=name)
            cerebro.adddata(data_feed)

    # Set starting cash
    cerebro.broker.setcash(cash)

    # Set commission
    cerebro.broker.setcommission(commission=commission)

    # Run the backtest
    cerebro.run()

    # Get final portfolio value
    final_value = cerebro.broker.getvalue()
    
    return final_value
