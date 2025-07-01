from collections import deque
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import linregress
from typing import Optional, Dict



 # 2. Define slope calculation function using linear regression
 #TODO Make these static
def linear_slope(series):
    if series.isnull().any():
        return np.nan
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

def ema_slope_angle(series):
    x = np.arange(len(series))
    y = series.values
    slope, _, _, _, _ = linregress(x, y)     # Get linear regression slope
    angle_rad = np.arctan(slope)            # Convert slope to angle (radians)
    angle_deg = np.degrees(angle_rad)       # Convert to degrees
    return angle_deg

class BarHistory:
    """
    Stores completed 1-minute bars for your trading system.
    Can return raw or indicator-enriched DataFrames.
    """

    def __init__(self, maxlen=5000):
        """
        :param maxlen: Max number of bars to store (rolling window).
        """
        self.bars = deque(maxlen=maxlen)

    def add_bar(self, bar: dict):
        """
        Add a completed bar.
        :param bar: dict with keys 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        """
        self.bars.append(bar)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get stored bars as pandas DataFrame.
        """
        if not self.bars:
            return pd.DataFrame()

        df = pd.DataFrame(self.bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        self.set_timezone_EST(df)    

        return df
    
    def set_timezone_EST(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        print("Index type:", type(df.index), "| sample index:", df.index[:3])
        return df

    def get_enriched_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with only the essential indicators needed for live decision-making.
        """
        df = self.get_dataframe()
        if df.empty:
            return df
        self.set_timezone_EST(df)    
        # Handle timezone: localize if naive, otherwise just convert
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        print("Index type:", type(df.index), "| sample index:", df.index[:3])



        # 1-Minute EMA(50) and its slope/angle
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_50_slope"] = df["ema_50"].rolling(window=5).apply(linear_slope, raw=False)
        df["ema_50_angle_1m"] = df["ema_50"].rolling(window=5).apply(ema_slope_angle, raw=False)

        # 1-Minute ATR(14)
        df["true_range_1m"] = (df["high"] - df["low"]).clip(lower=0)
        df["atr_14_1m"] = df["true_range_1m"].rolling(14).mean()

        # 5-Minute Resample for MACD and EMA7/17/33
        df5 = df[["open", "high", "low", "close", "volume"]].resample("5T").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        ema12_5 = df5["close"].ewm(span=12, adjust=False).mean()
        ema26_5 = df5["close"].ewm(span=26, adjust=False).mean()
        macd_5 = ema12_5 - ema26_5
        macd_signal_5 = macd_5.ewm(span=9, adjust=False).mean()

        df5["macd_5m"] = macd_5
        df5["macd_signal_5m"] = macd_signal_5
        df5["macd_cross_bullish_5m"] = ((macd_5.shift(1) <= macd_signal_5.shift(1)) & (macd_5 > macd_signal_5)).astype(int)
        df5["macd_cross_bearish_5m"] = ((macd_5.shift(1) >= macd_signal_5.shift(1)) & (macd_5 < macd_signal_5)).astype(int)

        df5["ema_7_5m"] = df5["close"].ewm(span=7, adjust=False).mean()
        df5["ema_17_5m"] = df5["close"].ewm(span=17, adjust=False).mean()
        df5["ema_33_5m"] = df5["close"].ewm(span=33, adjust=False).mean()
        df5["ema_7_angle_5m"] = df5["ema_7_5m"].rolling(window=5).apply(ema_slope_angle, raw=False)

        # 60-Minute Resample for EMA(9) and its slope/angle
        df60 = df[["open", "high", "low", "close", "volume"]].resample("60T").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        df60["ema_9_60m"] = df60["close"].ewm(span=9, adjust=False).mean()
        df60["ema_9_slope_60m"] = df60["ema_9_60m"].rolling(window=5).apply(linear_slope, raw=False)
        df60["ema_9_angle_60m"] = df60["ema_9_60m"].rolling(window=5).apply(ema_slope_angle, raw=False)

       # Forward fill relevant indicators back to 1-minute bars

        # 1) Define feature lists for clarity
        five_min_feats = [
            "macd_5m", "macd_signal_5m",
            "macd_cross_bullish_5m", "macd_cross_bearish_5m",
            "ema_7_5m", "ema_17_5m", "ema_33_5m", "ema_7_angle_5m"
        ]

        sixty_min_feats = [
            "ema_9_60m", "ema_9_slope_60m", "ema_9_angle_60m"
        ]

        # 2) Join 5-minute features back to 1-minute bars
        df = df.join(df5[five_min_feats], how="left")

        # 3) Join 60-minute features back to 1-minute bars
        df = df.join(df60[sixty_min_feats], how="left")

        # 4) Forward-fill higher timeframe indicators
        df.fillna(method="ffill", inplace=True)

        return df

    def get_latest_bar(self) -> Optional[Dict]:
        """
        Get the most recent completed bar as a dict.
        """
        return self.bars[-1] if self.bars else None

    def get_recent_bars(self, n=5) -> list:
        """
        Return the last n bars as list of dicts.
        """
        return list(self.bars)[-n:]
