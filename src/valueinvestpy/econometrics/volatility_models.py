from arch import arch_model
import pandas as pd

def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    """Fits a GARCH(p, q) model to a series of returns.

    Args:
        returns (pd.Series): A series of asset returns.
        p (int, optional): The order of the ARCH term. Defaults to 1.
        q (int, optional): The order of the GARCH term. Defaults to 1.

    Returns:
        ARCHModelResult: The fitted GARCH model results object.
    """
    # The arch_model works best with returns scaled by 100
    model = arch_model(returns * 100, vol='Garch', p=p, q=q, dist='Normal')
    results = model.fit(update_freq=5, disp='off')
    return results
