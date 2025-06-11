import os
import json
import glob
import pandas as pd
import random
import numpy as np
import torch
from util.indicators import compute_rsi
import traceback
import pickle
from sklearn.linear_model import LinearRegression
from datetime import datetime
from scipy.stats import linregress


# ─── Set Seed for Reproducibility ─────
SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 2. Define slope calculation function using linear regression
def linear_slope(series):
    if series.isnull().any():
        return np.nan
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

def save_dataframes(dataframes: dict, filetype="xlsx", folder="logs"):
    """
    Save multiple DataFrames to disk with timestamped filenames.

    Args:
        dataframes (dict): Dictionary where keys are names and values are DataFrames.
        filetype (str): "csv" or "xlsx".
        folder (str): Destination folder.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for name, df in dataframes.items():
        filename = f"{folder}/{name}_{timestamp}.{filetype}"

        if filetype == "csv":
            df.to_csv(filename, index=False)
        elif filetype == "xlsx":
            df.to_excel(filename, index=False)
        else:
            raise ValueError("Unsupported filetype. Use 'csv' or 'xlsx'.")

        print(f"Saved: {filename}")

        
# Function to calculate angle of the EMA slope in degrees
def ema_slope_angle(series):
    x = np.arange(len(series))
    y = series.values
    slope, _, _, _, _ = linregress(x, y)     # Get linear regression slope
    angle_rad = np.arctan(slope)            # Convert slope to angle (radians)
    angle_deg = np.degrees(angle_rad)       # Convert to degrees
    return angle_deg


def train_and_save_model():
    try:
        print("Starting backtesting script...")
        # Load Single File In case we need it 
        # json_path = "sample_data/sample_1min_5_19.json"

        # with open(json_path, "r") as f:
        #     raw = json.load(f)
        #     # reverse bars because topstep returns data last first
        # bars = raw["bars"][::-1]
        # df = pd.DataFrame(bars).rename(columns={
        #     "t": "datetime",  # original timestamp field
        #     "o": "open",
        #     "h": "high",
        #     "l": "low",
        #     "c": "price",
        #     "v": "volume",
        # })

        

        # ─── A. Data Prep (multi-day JSON) ──────────────────────
        all_bars = []
        # adjust the path to where you dumped your daily files
        for json_file in sorted(glob.glob("data/month_json/*.json")):
                with open(json_file, "r") as f:
                     raw = json.load(f)
                # raw["bars"] is in reverse chronological order; reverse it
                bars = raw["bars"][::-1]
                all_bars.extend(bars)

        # now build one big DataFrame
        df = (pd.DataFrame(all_bars).rename(columns={
            "t": "datetime",  # original timestamp field
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "price",
            "v": "volume",
        })
        )
        df.index = pd.to_datetime(df.index)  # ensure datetime dtype
        df.index = df.index.tz_localize('UTC')  # localize naive timestamps to UTC
        # Now convert to US/Eastern time
        df.index = df.index.tz_convert('US/Eastern')





        num_steps = len(df)
        df_price = df.copy()

        print("Columns after rename:", df.columns.tolist())

        # ─── Step 2: Parse & set index, then reset so datetime is a column ─────
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        print("Index type:", type(df.index), "| sample index:", df.index[:3])

         
        #B. Indicators
        # 1) 1-min indicators
        df["ema_50"]  = df["price"].ewm(span=50,  adjust=False).mean()
        df["ema_50_slope"] = df["ema_50"].rolling(window=5).apply(linear_slope, raw=False)
        # df["ema_200"] = df["price"].ewm(span=200, adjust=False).mean()
        typ = (df["high"] + df["low"] + df["price"])/3
        df["vwap"]    = (typ * df["volume"]).cumsum() / df["volume"].cumsum()

        # 1-min ATR 
        df["true_range_1m"]    = (df["high"] - df["low"]).clip(lower=0)
        df["atr_14_1m"]     = df["true_range_1m"].rolling(14).mean()
        
        # 2) 5-min bars + indicators
        df5 = df[["open","high","low","price","volume"]].resample("5T").agg({
            "open":"first","high":"max","low":"min","price":"last","volume":"sum"
        })
        # for span in (7,17,33):
        #  df5[f"ema_{span}_5m"] = df5["price"].ewm(span=span, adjust=False).mean()
        ema12_5 = df5["price"].ewm(span=12, adjust=False).mean()
        ema26_5 = df5["price"].ewm(span=26, adjust=False).mean()
        macd_5  = ema12_5 - ema26_5
        sig_5   = macd_5.ewm(span=9, adjust=False).mean()
        df5["macd_5m"]        = macd_5
        df5["macd_signal_5m"] = sig_5

        # 2) Flag the cross of MACD line over/under its signal line
        df5["macd_cross_bullish_5m"] = 0
        df5["macd_cross_bearish_5m"] = 0

        # bullish cross: MACD went from ≤signal → >signal
        df5.loc[(macd_5.shift(1) <= sig_5.shift(1)) & (macd_5 > sig_5),"macd_cross_bullish_5m"] = 1
        # bearish cross: MACD went from ≥signal → <signal
        df5.loc[(macd_5.shift(1) >= sig_5.shift(1)) & (macd_5 < sig_5),"macd_cross_bearish_5m"] = 1

        df5["rsi_5m"]    = compute_rsi(df5["price"], length=10)
        df5["rsi_ma_5m"] = df5["rsi_5m"].rolling(3).mean()

        # On your 5-min resampled df5:
        df5["true_range_5m"]   = (df5["high"] - df5["low"]).clip(lower=0)
        df5["atr_14_5m"]    = df5["true_range_5m"].rolling(14).mean()
        df5["bb_mid_5m"]    = df5["price"].rolling(20).mean()
        df5["bb_std_5m"]    = df5["price"].rolling(20).std()
        df5["bb_upper_5m"]  = df5["bb_mid_5m"] + 2 * df5["bb_std_5m"]
        df5["bb_lower_5m"]  = df5["bb_mid_5m"] - 2 * df5["bb_std_5m"]
        df5["ema_7_5m"]  = df5["price"].ewm(span=7, adjust=False).mean()
        df5["ema_17_5m"] = df5["price"].ewm(span=17, adjust=False).mean()
        df5["ema_33_5m"] = df5["price"].ewm(span=33, adjust=False).mean()
        df5["ema_7_slope_5m"] = df5["ema_7_5m"].rolling(window=5).apply(linear_slope, raw=False)
        df5["ema_7_angle_5m"] = df5["ema_7_5m"].rolling(window=5).apply(ema_slope_angle, raw=False)



        # 3) 15-min bars + MACD
        df15 = df[["open","high","low","price","volume"]].resample("15T").agg({
            "open":"first","high":"max","low":"min","price":"last","volume":"sum"
        })
        ema12_15 = df15["price"].ewm(span=12, adjust=False).mean()
        ema26_15 = df15["price"].ewm(span=26, adjust=False).mean()
        macd_15  = ema12_15 - ema26_15
        sig_15   = macd_15.ewm(span=9, adjust=False).mean()
        df15["macd_15m"]        = macd_15
        df15["macd_signal_15m"] = sig_15
        df15["macd_cross_15m"] = 0
        df15["macd_slope_15m"] = df15["macd_15m"].diff()
        # bullish cross: MACD went from ≤ signal → > signal
        df15.loc[
            (macd_15.shift(1) <= sig_15.shift(1)) &
            (macd_15 > sig_15),
            "macd_cross_15m"
        ] = +1
        # bearish cross: MACD went from ≥ signal → < signal
        df15.loc[
            (macd_15.shift(1) >= sig_15.shift(1)) &
            (macd_15 < sig_15),
            "macd_cross_15m"
        ] = -1



        # 4) merge back & reset index → datetime column returns
        five_min_feats = [
             "macd_5m","macd_signal_5m",
             "rsi_5m","rsi_ma_5m","macd_cross_bullish_5m","macd_cross_bearish_5m"
             ,"true_range_5m","atr_14_5m","bb_mid_5m",
             "bb_std_5m","bb_upper_5m","bb_lower_5m","ema_7_5m","ema_17_5m","ema_33_5m","ema_7_angle_5m",           # ← add it here
        ]

        fifteen_min_feats = [
            "macd_15m", "macd_signal_15m", "macd_cross_15m","macd_slope_15m",  # example features
         # add more 15m feature columns here as needed
        ]


        # Forward-fill only selected 5-minute features
        for col in five_min_feats: df[col] = df5[col].reindex(df.index, method="ffill")
        # Forward-fill only selected 15-minute features
        for col in fifteen_min_feats: df[col] = df15[col].reindex(df.index).ffill()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)


        # Step 1: Define columns to skip
        exclude_cols = ["index","datetime", "balance", "position","macd_cross_5m"]

        # Add Symbol
        df["symbol"] = "GC2"
        print(f"DataFrame ready: {len(df)} rows, head:\n", df.head())

        # print(f"Loaded futures data with shape: {all_bars}")
        # print(f"Loaded {len(all_bars)} bars")
        pd.set_option('display.width', None)           # Don't wrap columns
        pd.set_option('display.max_colwidth', None)    # Show full column content
        print(f"Loaded futures data with shape: {bars}")
        print(f"Loaded {len(bars)} bars")


        #Save dataframes to csv
        save_dataframes({
            "df": df,
            "df_price": df_price
        }, filetype="csv")
    
                # ─── D. Clean one‐pass evaluation ─────────────────────────
        print("\nStarting clean one-pass evaluation…")
        print("\nStarting clean one-pass evaluation…")
        # for counting and printing signals
        signal_count = 0  
        #add equity curve
        done  = False
        step_idx = 0
        trades = []
        trade = {}
        unrealized_pnl = 0
        balance  = 0
        while not done:
            #if at the end of data set
            # look at the bar we’re about to act on
            bar      = df.iloc[step_idx]
            # pick whichever cross you’re using:
            print("macd_cross_bullish_5m:",bar["macd_cross_bullish_5m"],"macd_cross_bearish_5m",bar["macd_cross_bearish_5m"],"ema_50_slope:",bar["ema_50_slope"],"ema_7_5m:",bar["ema_7_5m"],"ema_17_5m:",bar["ema_17_5m"],"ema_33_5m:",bar["ema_33_5m"],
                  "ema_7_angle_5m",bar["ema_7_angle_5m"]
                  )
            angle = bar["ema_7_angle_5m"]
            price = bar["price"]
            # or, if you moved to 15m:
            # cross15 = bar["macd_cross_15m"]
            # Check angle threshold
            if not trade:
                if angle >= 20:
                        # Enter trade
                        entry_price = price
                        trade = {
                            "entry_time": df.iloc[step_idx],
                            "entry_price": entry_price,
                        }
                        print(f"Entered trade at {entry_price}")
                        trades.append(trade)
            else:
                unrealized_pnl = (price - trade["entry_price"]) *100
                if unrealized_pnl >= 1500 or unrealized_pnl <= -500:
                    trade["exit_time"] = df.index[step_idx]
                    trade["exit_price"] = price
                    trade["trade_pnl"] = unrealized_pnl
                    balance += unrealized_pnl
                    trade["balance"] = balance
                    trade = {}
                    trades.append(trade)
                    unrealized_pnl = 0


            step_idx += 1
            if step_idx >= len(df):
                with open("trades_log.json", "w") as f:
                    json.dump(trades, f, indent=4, default=str)  # default=str handles datetime objects
                done = True
            




    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc() 

if __name__ == "__main__":
    train_and_save_model()


