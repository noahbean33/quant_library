import pandas as pd
from src.technical_analysis.moving_averages import ema

def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Calculate the Moving Average Convergence Divergence (MACD)."""
    ema_fast = ema(data, window=fast_period)
    ema_slow = ema(data, window=slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, window=signal_period)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })
