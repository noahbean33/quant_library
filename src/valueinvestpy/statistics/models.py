import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

def fit_arima(series: pd.Series, order: tuple = (1, 0, 1)) -> ARIMAResults:
    """
    Fits an ARIMA (AutoRegressive Integrated Moving Average) model to a time series.

    ARIMA models are a class of statistical models for analyzing and forecasting
    time series data. They are specified by three order parameters: (p, d, q).

    - p: The number of lag observations included in the model (AR part).
    - d: The number of times the raw observations are differenced (I part).
    - q: The size of the moving average window (MA part).

    Args:
        series (pd.Series): The time series data to model.
        order (tuple, optional): The (p, d, q) order of the model. 
                                 Defaults to (1, 0, 1) for a simple ARMA model.

    Returns:
        ARIMAResults: The fitted ARIMA model result object, which contains
                      information like model summary, parameters, and residuals.
    """
    # The series must have a frequency set for the ARIMA model to converge
    # If no frequency is present, we infer it or set it to a business day frequency
    if series.index.freq is None:
        series = series.asfreq(pd.infer_freq(series.index) or 'B')

    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit
