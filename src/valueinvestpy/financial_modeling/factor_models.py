import pandas as pd
import statsmodels.api as sm

def run_factor_model(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Runs a factor model regression using Ordinary Least Squares (OLS).

    Args:
        portfolio_returns (pd.Series): A series of portfolio excess returns.
        factor_returns (pd.DataFrame): A DataFrame of factor returns (e.g., Mkt-RF, SMB, HML).

    Returns:
        sm.regression.linear_model.RegressionResultsWrapper: The results of the OLS regression.
    """
    # The dependent variable (y) is the portfolio's excess return.
    y = portfolio_returns

    # The independent variables (X) are the factors. Add a constant for the intercept (alpha).
    X = sm.add_constant(factor_returns)

    # Fit the OLS model
    model = sm.OLS(y, X).fit()

    return model
