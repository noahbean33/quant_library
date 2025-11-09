import pandas as pd
import numpy as np
from ..portfolio import Portfolio

def run_backtest(portfolio: Portfolio):
    """
    Runs a buy-and-hold backtest for a given portfolio.

    Args:
        portfolio (Portfolio): An initialized Portfolio object.

    Returns:
        dict: A dictionary containing key performance metrics.
    """
    if portfolio.history.empty:
        print("Warning: Cannot run backtest on an empty portfolio or with no data.")
        return {}

    # Calculate daily returns for each asset
    daily_returns = portfolio.history.pct_change().dropna()

    # Calculate weighted portfolio returns
    weights = np.array([portfolio.weights[ticker] for ticker in daily_returns.columns])
    portfolio_returns = daily_returns.dot(weights)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # --- Performance Metrics ---
    total_return = (cumulative_returns.iloc[-1] - 1)
    
    # Annualized metrics (assuming 252 trading days)
    trading_days = 252
    # Ensure we don't divide by zero if there's only one return
    if len(portfolio_returns) > 1:
        annualized_return = ((1 + total_return) ** (trading_days / len(portfolio_returns))) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(trading_days)
    else:
        annualized_return = 0.0
        annualized_volatility = 0.0

    # Sharpe Ratio (assuming risk-free rate of 0)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0
    
    # Max Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    performance_summary = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cumulative_returns': cumulative_returns
    }
    
    return performance_summary
