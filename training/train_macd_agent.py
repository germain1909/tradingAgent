import os
import json
import glob
import pandas as pd
import random
import numpy as np
import torch
from envs.futures_trading_env import FuturesTradingEnv  # Your custom gym environment
from agents.drl_agent import PPOTradingAgent  # Your PPO agent class
from callbacks.trade_logging import TradeLoggingJSONCallback #Logging callback
from util.indicators import compute_rsi
import traceback
import pickle
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.linear_model import LinearRegression
from datetime import datetime

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


def train_and_save_model():
    try:
        print("Starting training script...")
        #Load Single File In case we need it 
        json_path = "sample_data/sample_1min_5_19.json"

        with open(json_path, "r") as f:
            raw = json.load(f)
            # reverse bars because topstep returns data last first
        bars = raw["bars"][::-1]
        df = pd.DataFrame(bars).rename(columns={
            "t": "datetime",  # original timestamp field
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "price",
            "v": "volume",
        })

        df_price = df.copy()

        # # ─── A. Data Prep (multi-day JSON) ──────────────────────
        # all_bars = []
        # # adjust the path to where you dumped your daily files
        # for json_file in sorted(glob.glob("data/month_json/*.json")):
        #         with open(json_file, "r") as f:
        #              raw = json.load(f)
        #         # raw["bars"] is in reverse chronological order; reverse it
        #         day_bars = raw["bars"][::-1]
        #         all_bars.extend(day_bars)

        # # now build one big DataFrame
        # df = (pd.DataFrame(all_bars).rename(columns={
        #     "t": "datetime",  # original timestamp field
        #     "o": "open",
        #     "h": "high",
        #     "l": "low",
        #     "c": "price",
        #     "v": "volume",
        # })
        # )
        num_steps = len(df)

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
        df5["macd_cross_5m"] = 0
        # bullish cross: MACD went from ≤signal → >signal
        df5.loc[(macd_5.shift(1) <= sig_5.shift(1)) & (macd_5 > sig_5),"macd_cross_5m"] = +1
        # bearish cross: MACD went from ≥signal → <signal
        df5.loc[(macd_5.shift(1) >= sig_5.shift(1)) & (macd_5 < sig_5),"macd_cross_5m"] = -1

        df5["rsi_5m"]    = compute_rsi(df5["price"], length=10)
        df5["rsi_ma_5m"] = df5["rsi_5m"].rolling(3).mean()

        # On your 5-min resampled df5:
        df5["true_range_5m"]   = (df5["high"] - df5["low"]).clip(lower=0)
        df5["atr_14_5m"]    = df5["true_range_5m"].rolling(14).mean()
        df5["bb_mid_5m"]    = df5["price"].rolling(20).mean()
        df5["bb_std_5m"]    = df5["price"].rolling(20).std()
        df5["bb_upper_5m"]  = df5["bb_mid_5m"] + 2 * df5["bb_std_5m"]
        df5["bb_lower_5m"]  = df5["bb_mid_5m"] - 2 * df5["bb_std_5m"]

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
             "rsi_5m","rsi_ma_5m",
             "macd_cross_5m","true_range_5m","atr_14_5m","bb_mid_5m",
             "bb_std_5m","bb_upper_5m","bb_lower_5m"           # ← add it here
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

        #Begin Normalization
        # Step 2: Identify numeric columns not in the exclusion list
        cols_to_normalize = [
            col for col in df.select_dtypes(include=["number"]).columns
            if col not in exclude_cols
        ]

        # Step 3: Normalize only selected columns
        df_norm = df.copy()
        for col in cols_to_normalize:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                df_norm[col] = 0  # or np.nan if you want to flag it
            else:
                df_norm[col] = (df[col] - mean) / std

        # print(f"Loaded futures data with shape: {all_bars}")
        # print(f"Loaded {len(all_bars)} bars")
        pd.set_option('display.width', None)           # Don't wrap columns
        pd.set_option('display.max_colwidth', None)    # Show full column content
        print(f"Loaded futures data with shape: {bars}")
        print(f"Loaded {len(bars)} bars")
        print(df_norm.tail(5))


        #Save dataframes to csv
        save_dataframes({
            "df": df,
            "df_norm": df_norm,
            "df_price": df_price
        }, filetype="csv")
        
        # D. Prepare environment kwargs, adjust depending on your env's __init__ signature
        env_kwargs = {"data": df, "normalized_data": df_norm}
        print ('kwargs added')

        # Initialize the PPO agent with your custom environment class and kwargs
        agent = PPOTradingAgent(
            env_class=FuturesTradingEnv,
            env_kwargs=env_kwargs,
            verbose=1,
            n_envs=4,        # single–env backtest
            n_steps=2048,     # collect 900 bars before each update
            batch_size=32,   # mini-batch size of 10
            learning_rate=3e-5,
            gamma=0.95,
            gae_lambda=0.92,
            clip_range=0.2,
            ent_coef=0.005,
            seed=SEED,
            tensorboard_log="models/tb_logs",    # ← new: where SB3 will write event files
        )

        # Instantiate with verbose=1 for real-time prints
        trade_cb = TradeLoggingJSONCallback(
        log_path="models/trades.json",
        verbose=1
        )

        # Evaluation environemt and callback
        eval_env = DummyVecEnv([lambda: FuturesTradingEnv(**env_kwargs)])
        eval_cb  = EvalCallback(
            eval_env,
            best_model_save_path="models/best_model/",
            log_path="models/eval_logs/",
            eval_freq= num_steps,      # evaluate once per pass
            n_eval_episodes=1,
            deterministic=True,
        )

        # Train the agent
        agent.train(total_timesteps=num_steps*20,callback=[trade_cb,eval_cb])
        print("Training completed.")

        # Save the trained model
        os.makedirs("models", exist_ok=True)
        agent.save("models/ppo_macd_futures")
        print("Model saved to models/ppo_macd_futures")

        print("Training complete and model saved!")

                # ─── D. Clean one‐pass evaluation ─────────────────────────
        print("\nStarting clean one-pass evaluation…")
        print("\nStarting clean one-pass evaluation…")
        eval_env    = FuturesTradingEnv(**env_kwargs)
        obs         = eval_env.reset()
        done        = False

        # for counting and printing signals
        signal_count = 0  

        # start equity curve at initial cash
        equity_curve = [eval_env.balance]

        while not done:
            # look at the bar we’re about to act on
            step_idx = eval_env.current_step
            bar      = eval_env.data.iloc[step_idx]
            # pick whichever cross you’re using:
            cross5   = bar["macd_cross_5m"]
            # or, if you moved to 15m:
            # cross15 = bar["macd_cross_15m"]

            if cross5 != 0:
                signal_count += 1
                print(f"  → MACD 5m cross {cross5:+d} at {bar['datetime']} (step {step_idx})")

            action, _ = agent.model.predict(obs, deterministic=True)
            obs, _, done, info = eval_env.step(action)

            bal   = info["balance"]
            pos   = info["position"]
            price = info["current_price"]
            equity_curve.append(bal + pos * price * 100)

        # after the loop:
        print(f"\nTotal MACD-5m crosses seen during eval: {signal_count}")

        # dump all closed trades & equity curve
        with open("models/eval_trades.json", "w") as f:
            json.dump(eval_env.trades, f, default=str, indent=2)
        with open("models/equity_curve.pkl", "wb") as f:
            pickle.dump(equity_curve, f)

        final_pl = equity_curve[-1] - equity_curve[0]
        print(f"\nFinal evaluation P/L = {final_pl:.2f}")
        print(f"Saved {len(eval_env.trades)} trades → models/eval_trades.json")
        print(f"Equity curve → models/equity_curve.pkl")

        




    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc() 

if __name__ == "__main__":
    train_and_save_model()


