import numpy as np
import pandas as pd
import scipy.optimize as sco

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    """Calculates portfolio annualized return and volatility."""
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calculates the negative Sharpe ratio."""
    p_std, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """Finds the portfolio with the maximum Sharpe ratio."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Calculates portfolio volatility."""
    return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    """Finds the portfolio with the minimum variance."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    """Calculates the efficient frontier."""
    efficients = []
    for ret in returns_range:
        constraints = ({'type': 'eq', 'fun': lambda x: portfolio_annualized_performance(x, mean_returns, cov_matrix)[1] - ret},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for asset in range(len(mean_returns)))
        result = sco.minimize(portfolio_volatility, len(mean_returns)*[1./len(mean_returns),], args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        efficients.append(result)
    return efficients
