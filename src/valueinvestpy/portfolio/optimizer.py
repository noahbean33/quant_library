import numpy as np
import pandas as pd
import scipy.optimize as optimization
from .portfolio import Portfolio

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

def _calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates the log returns of the historical data."""
    return np.log(data / data.shift(1)).dropna()

def _portfolio_statistics(weights: np.ndarray, log_returns: pd.DataFrame):
    """Calculates the annualized return, volatility, and Sharpe ratio for a given portfolio."""
    annual_return = np.sum(log_returns.mean() * weights) * NUM_TRADING_DAYS
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * NUM_TRADING_DAYS, weights)))
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
    return np.array([annual_return, annual_volatility, sharpe_ratio])

def _minimize_sharpe(weights: np.ndarray, log_returns: pd.DataFrame) -> float:
    """Objective function to minimize (negative Sharpe ratio)."""
    return -_portfolio_statistics(weights, log_returns)[2]

def optimize_portfolio(portfolio: Portfolio):
    """
    Performs Markowitz portfolio optimization.

    Args:
        portfolio (Portfolio): An initialized Portfolio object.

    Returns:
        Tuple containing:
        - pd.DataFrame: A DataFrame with weights, returns, risks, and Sharpe ratios for simulated portfolios.
        - Dict: The optimal portfolio with the maximum Sharpe ratio.
        - Dict: The optimal portfolio with the minimum volatility.
    """
    log_returns = _calculate_log_returns(portfolio.history)
    num_assets = len(portfolio.tickers)

    # --- 1. Monte Carlo Simulation for random portfolios ---
    simulated_weights = []
    simulated_stats = []
    for _ in range(NUM_PORTFOLIOS):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        simulated_weights.append(weights)
        simulated_stats.append(_portfolio_statistics(weights, log_returns))

    simulated_portfolios_df = pd.DataFrame(
        data=np.array(simulated_stats),
        columns=['return', 'volatility', 'sharpe_ratio']
    )
    simulated_portfolios_df['weights'] = simulated_weights

    # --- 2. Optimization to find the best portfolios ---
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    # Find Max Sharpe Ratio portfolio
    max_sharpe_optimizer = optimization.minimize(
        fun=_minimize_sharpe,
        x0=initial_weights,
        args=(log_returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    max_sharpe_stats = _portfolio_statistics(max_sharpe_optimizer.x, log_returns)
    max_sharpe_portfolio = {
        'return': max_sharpe_stats[0],
        'volatility': max_sharpe_stats[1],
        'sharpe_ratio': max_sharpe_stats[2],
        'weights': dict(zip(portfolio.tickers, max_sharpe_optimizer.x))
    }

    # Find Min Volatility portfolio
    min_vol_optimizer = optimization.minimize(
        fun=lambda w, r: _portfolio_statistics(w, r)[1], # Minimize volatility
        x0=initial_weights,
        args=(log_returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    min_vol_stats = _portfolio_statistics(min_vol_optimizer.x, log_returns)
    min_vol_portfolio = {
        'return': min_vol_stats[0],
        'volatility': min_vol_stats[1],
        'sharpe_ratio': min_vol_stats[2],
        'weights': dict(zip(portfolio.tickers, min_vol_optimizer.x))
    }

    return simulated_portfolios_df, max_sharpe_portfolio, min_vol_portfolio
