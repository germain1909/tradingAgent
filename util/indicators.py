# utils/indicators.py
import pandas as pd

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Compute the classic RSI on a price series.
    Returns a pandas Series of the same length.
    """
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up   = up.rolling(length).mean()
    ma_down = down.rolling(length).mean()
    rs      = ma_up / ma_down
    return 100 - (100 / (1 + rs))