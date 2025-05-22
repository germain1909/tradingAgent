import os
import json
import pandas as pd
from envs.futures_trading_env import FuturesTradingEnv  # Your custom gym environment
from agents.drl_agent import PPOTradingAgent  # Your PPO agent class
from callbacks.trade_logging import TradeLoggingJSONCallback #Logging callback


def train_and_save_model():
    try:
        print("Starting training script...")

        # Load or prepare your futures data here (Pandas DataFrame)
        #data = pd.read_csv('data/futures_data.csv')  # Replace with your actual data path
        json_path = "sample_data/sample_1min.json"

        with open(json_path, "r") as f:
            raw = json.load(f)
        bars = raw["bars"]
        df = pd.DataFrame(bars)
        #Rename the close price column to what your env expects
        df = df.rename(columns={"c": "price"})
        df["t"] = pd.to_datetime(df["t"])
        df["symbol"] = "GC2"
        print(f"Loaded futures data with shape: {bars}")
        print(f"Loaded {len(bars)} bars")
        print(df)

        # Prepare environment kwargs, adjust depending on your env's __init__ signature
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
