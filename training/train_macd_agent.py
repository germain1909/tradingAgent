import os
import json
import pandas as pd
from envs.futures_trading_env import FuturesTradingEnv  # Your custom gym environment
from agents.drl_agent import PPOTradingAgent  # Your PPO agent class
from callbacks.trade_logging import TradeLoggingJSONCallback #Logging callback
from util.indicators import compute_rsi


def train_and_save_model():
    try:
        print("Starting training script...")


        #A.Data Prep
        # Load or prepare your futures data here (Pandas DataFrame)
        #data = pd.read_csv('data/futures_data.csv')  # Replace with your actual data path
        json_path = "sample_data/sample_1min.json"

        with open(json_path, "r") as f:
            raw = json.load(f)
        bars = raw["bars"]
        df = pd.DataFrame(bars).rename(columns={
            "t": "datetime",  # original timestamp field
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "price",
            "v": "volume",
        })
        print("Columns after rename:", df.columns.tolist())

        # ─── Step 2: Parse & set index, then reset so datetime is a column ─────
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        print("Index type:", type(df.index), "| sample index:", df.index[:3])

         
        #B. Indicators
        # 1) 1-min indicators
        df["ema_50"]  = df["price"].ewm(span=50,  adjust=False).mean()
        df["ema_200"] = df["price"].ewm(span=200, adjust=False).mean()
        typ = (df["high"] + df["low"] + df["price"])/3
        df["vwap"]    = (typ * df["volume"]).cumsum() / df["volume"].cumsum()
        df["rsi_1m"]  = compute_rsi(df["price"], length=14)

        # 2) 5-min bars + indicators
        df5 = df[["open","high","low","price","volume"]].resample("5T").agg({
            "open":"first","high":"max","low":"min","price":"last","volume":"sum"
        })
        for span in (7,17,33):
            df5[f"ema_{span}_5m"] = df5["price"].ewm(span=span, adjust=False).mean()
        ema12_5 = df5["price"].ewm(span=12, adjust=False).mean()
        ema26_5 = df5["price"].ewm(span=26, adjust=False).mean()
        macd_5  = ema12_5 - ema26_5
        sig_5   = macd_5.ewm(span=9, adjust=False).mean()
        df5["macd_5m"]        = macd_5
        df5["macd_signal_5m"] = sig_5
        df5["macd_hist_5m"]   = macd_5 - sig_5
        df5["rsi_5m"]    = compute_rsi(df5["price"], length=10)
        df5["rsi_ma_5m"] = df5["rsi_5m"].rolling(3).mean()

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
        df15["macd_hist_15m"]   = macd_15 - sig_15

        # 4) merge back & reset index → datetime column returns
        for col in df5.columns:   df[col]  = df5[col].reindex(df.index, method="ffill")
        for col in df15.columns:  df[col]  = df15[col].reindex(df.index, method="ffill")
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        # C. Reset Index and ADD SYMBOL
        # bring datetime back as a column for the env
        df.reset_index(inplace=True)
        print("Columns after reset_index:", df.columns.tolist())

        # Add Symbol
        df["symbol"] = "GC2"
        print(f"DataFrame ready: {len(df)} rows, head:\n", df.head())

        print(f"Loaded futures data with shape: {bars}")
        print(f"Loaded {len(bars)} bars")
        print(df)


        
        
        # D. Prepare environment kwargs, adjust depending on your env's __init__ signature
        env_kwargs = {"data": df}
        print ('kwargs added')

        # Initialize the PPO agent with your custom environment class and kwargs
        agent = PPOTradingAgent(env_class=FuturesTradingEnv, env_kwargs=env_kwargs, verbose=1)

        # Instantiate with verbose=1 for real-time prints
        trade_cb = TradeLoggingJSONCallback(
        log_path="models/trades.json",
        verbose=1
        )

        # Train the agent
        agent.train(total_timesteps=10000,callback=trade_cb)
        print("Training completed.")

        # Save the trained model
        os.makedirs("models", exist_ok=True)
        agent.save("models/ppo_macd_futures")
        print("Model saved to models/ppo_macd_futures")

        print("Training complete and model saved!")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_and_save_model()
