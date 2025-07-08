# trader.py
from signalrcore.hub_connection_builder import HubConnectionBuilder
import logging
import time
from util.bar_aggregator import BarAggregator
import random
from datetime import datetime
from util.BarFileWriter import BarFileWriter
from util.BarHistory import BarHistory
from util.Strategy import Strategy
from functools import partial
from util.HistoricalFetcher import HistoricalFetcher
from util.TopStepExecutor import TopstepExecutor
import os
import pandas as pd
import json
import glob

# SSL Cert
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()


#TOPSTEP Configuration
ASSET_ID = "CON.F.US.GCE.Q25"
OUTPUT_DIR = "data/warmup"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjIyNzQ3NiIsImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL3NpZCI6ImZiN2E0NWY4LWM3MDctNGFjYi04Yzk4LTYwMThlNTcxYWY1NSIsImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL25hbWUiOiJzYWludGdlcm1haW4iLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJ1c2VyIiwibXNkIjoiQ01FX1RPQiIsIm1mYSI6InZlcmlmaWVkIiwiZXhwIjoxNzUyNTI3OTA2fQ.iDjja4w8-I7WtjDZ2hNhGfHzIY2yO9ST5tCNrA7sMmk"
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

LIVE_MODE = False  # Set to True when market is open
BACKTEST = "BACKTEST"
LIVE = "LIVE"
SIM = "SIM"
MODE = LIVE # BACKTEST or SIM or LIVE
CONTRACT_ID = "CON.F.US.GCE.Q25"
ACCOUNT_ID = "7882531"


# Instantiate the aggregator once
aggregator = BarAggregator()
bar_history = BarHistory()
fetcher = HistoricalFetcher(ASSET_ID, OUTPUT_DIR, HEADERS)
topstep_executor  =  TopstepExecutor(ACCOUNT_ID,CONTRACT_ID,HEADERS)




def trade():
    print("traded")

def bar_to_json(bar):
    return {
        "timestamp": bar["timestamp"].isoformat() if isinstance(bar["timestamp"], datetime) else bar["timestamp"],
        "open": bar["open"],
        "high": bar["high"],
        "low": bar["low"],
        "close": bar["close"],
        "volume": bar["volume"]
    }


def save_dataframes(dataframes: dict, filetype="xlsx", folder="logs", tag=""):
    """
    Save multiple DataFrames to disk with timestamped filenames.

    Args:
        dataframes (dict): Dictionary where keys are names and values are DataFrames.
        filetype (str): "csv" or "xlsx".
        folder (str): Destination folder.
        tag (str): Optional tag to include in the filename (e.g. 'warmup', 'backtest').
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for name, df in dataframes.items():
        filename = f"{folder}/{tag}_{name}_{timestamp}.{filetype}" if tag else f"{folder}/{name}_{timestamp}.{filetype}"

        if filetype == "csv":
            df.to_csv(filename, index=False)
        elif filetype == "xlsx":
            df.to_excel(filename, index=False)
        else:
            raise ValueError("Unsupported filetype. Use 'csv' or 'xlsx'.")

        print(f"Saved: {filename}")

def on_gateway_trade(args,writer=None,strategy=None):
    if MODE == LIVE:
        # In live mode, args = [contract_id, [list_of_trades]]
        trades = args[1] if (isinstance(args, list) and len(args) > 1) else []
        if not trades:
            print("[GatewayTrade] No trades found in LIVE mode.")
            return
        raw_trade = trades[-1]  # Get the last trade in the list
        print("[RawGatewayTrade_LIVE]", raw_trade)
    else:
        # In SIM or BACKTEST, args = [trade_dict]
        raw_trade = args[0]
        print("[RawGatewayTrade_SIM]", raw_trade)

    trade = {
        "price": raw_trade["price"],
        "volume": raw_trade["volume"],
        "timestamp": raw_trade["timestamp"]
    }
    print("[GatewayTrade]", trade)
    aggregator.on_trade(trade)

     # Try to get a completed bar (1-min)
    # will either return a bar on None which is falsy 
    new_bar = aggregator.get_latest_completed_bar()  

    if new_bar:
        #TODO we need to update to handle TOPSTEP data structure
        bar_json = bar_to_json(new_bar)
        bar_history.add_bar(bar_json)
        print(bar_json)
        bar_history.add_bar(bar_json)
        if writer:
            writer.append_bar(bar_json)
        # Enrich and run strategy after each new bar in LIVE/SIM
        enriched_df = bar_history.get_enriched_dataframe()
        if len(enriched_df) > 20:  
            current_bar = enriched_df.iloc[-1]
            current_time = enriched_df.index[-1]
            if strategy:
                strategy.should_execute(current_bar, enriched_df, current_time)
    else:
        print("No completed bars yet.")



def setup_signalr_connection():

    def on_open():
        print("✅ Connection established!")
        # Now it's safe to subscribe:
        hub_connection.send("SubscribeContractTrades", [CONTRACT_ID])
        # Add more subscriptions if you want:
        # hub_connection.send("SubscribeContractQuotes", [CONTRACT_ID])
        # hub_connection.send("SubscribeContractMarketDepth", [CONTRACT_ID])


    writer = BarFileWriter("bars.json", max_lines=1000)
    strategy = Strategy(mode=LIVE,executor=topstep_executor)
    hub_connection = HubConnectionBuilder()\
        .with_url(
            f"https://rtc.topstepx.com/hubs/market?access_token={TOKEN}"
            )\
        .configure_logging(logging.INFO)\
        .with_automatic_reconnect({
            "type": "raw",
            "keep_alive_interval": 10,
            "reconnect_interval": 5,
            "max_attempts": 5
        })\
        .build()

    # Register callbacks
    hub_connection.on_open(on_open)  # This is the key line!
    hub_connection.on("GatewayTrade", partial(on_gateway_trade, writer=writer, strategy=strategy))
    # hub_connection.on("GatewayQuote", on_gateway_quote)
    # hub_connection.on("GatewayDepth", on_gateway_depth)

    # Start the connection
    hub_connection.start()

    # Subscribe to the contract's data
    # hub_connection.send("SubscribeContractQuotes", [CONTRACT_ID])
    # hub_connection.send("SubscribeContractTrades", [CONTRACT_ID])
    # hub_connection.send("SubscribeContractMarketDepth", [CONTRACT_ID])

    # print("SignalR connection established. Subscribed to market data.")

    #Keep the connection alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        # hub_connection.send("UnsubscribeContractQuotes", [CONTRACT_ID])
        hub_connection.send("UnsubscribeContractTrades", [CONTRACT_ID])
        # hub_connection.send("UnsubscribeContractMarketDepth", [CONTRACT_ID])
        hub_connection.stop()
        print("🛑 Live Connection interrupted.")



def simulate_trades():
    print("📈 Starting smart trade simulation...")
    writer = BarFileWriter("bars.json", max_lines=1000)
    strategy = Strategy(mode=SIM)
    

    price = 2350.0
    trend = "flat"
    ticks = 0

    try:
        while True:
            # Shift trend every 60 ticks (~1.5 minutes)
            if ticks % 60 == 0:
                trend = random.choice(["up", "down", "flat"])
                print(f"\n📊 Market trend changed to: {trend.upper()}")

            # Simulate price movement based on trend
            if trend == "up":
                price += random.uniform(0.1, 0.5)
            elif trend == "down":
                price -= random.uniform(0.1, 0.5)
            else:  # flat/consolidation
                price += random.uniform(-0.2, 0.2)

            # Keep price in reasonable range
            price = max(2300, min(price, 2450))

            # Simulate volume (with rare spikes)
            if random.random() < 0.05:
                volume = random.randint(10, 50)  # spike
            else:
                volume = random.randint(1, 5)

            mock_trade = {
                "contractId": CONTRACT_ID,
                "price": round(price, 2),
                "volume": volume,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            print("[MockTrade]", mock_trade)
            on_gateway_trade([mock_trade],writer=writer)

            ticks += 1
            time.sleep(1.5)  # 1.5 sec between trades
    except KeyboardInterrupt:
        print("🛑 Simulation interrupted.")


def run_backtest():
    print("🔍 Running backtest...")

    # ─── A. Load & Combine Multi-Day JSON Files ──────────────
    all_bars = []
    for json_file in sorted(glob.glob("data/month_json_6_15/*.json")):
        with open(json_file, "r") as f:
            raw = json.load(f)

        bars = raw["bars"][::-1]  # reverse to chronological order
        all_bars.extend(bars)

    # ─── B. Convert to DataFrame ──────────────────────────────
    df = pd.DataFrame(all_bars).rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "price",
        "v": "volume",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    # ─── C. Load into BarHistory & Enrich Once ───────────────
    bar_history = BarHistory()
    strategy = Strategy(mode=BACKTEST)

    # Add all bars at once
    for idx, row in df.iterrows():
        bar = {
            "timestamp": idx,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "price": row["price"],
            "volume": row["volume"]
        }
        bar_history.add_bar(bar)

    # Enrich once after all bars are loaded
    enriched_df = bar_history.get_enriched_dataframe()
    print(f"Total bars in enriched_df: {len(enriched_df)}")


    # Make a copy to avoid modifying original
    enriched_df_with_ts = enriched_df.copy()

    # Add timestamp column from index
    enriched_df_with_ts["timestamp"] = enriched_df_with_ts.index
    #Save dataframes to csv
    save_dataframes({
        "enriched_df_with_ts": enriched_df_with_ts
    }, filetype="csv", tag="backtest")


    # ─── D. Run Strategy Over Enriched Data ───────────────────
    for i in range(len(enriched_df)):
        if i < 20:
            continue  # skip warmup period

        current_bar = enriched_df.iloc[i]
        current_time = enriched_df.index[i]

        strategy.should_execute(current_bar, enriched_df, current_time)

    print("📉 Backtest complete.")






print("⏳ Fetching warm-up data...")
df = fetcher.fetch_past_hours(10)
print(df.head(1))

#When using iterrows(), the timestamp is in row.name, not row["timestamp"].
#so You could reset the index first:
#rename index to timestamp
if df.empty:
    print("⚠️ No warm-up data fetched. Trading logic may not behave as expected.")
else:
    print(f"✅ Warm-up data received: {len(df)} rows.")

    # Localize and convert timestamp to EST
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    df.index.name = "timestamp"
    df = df.reset_index()

    #Save dataframes to csv
    save_dataframes({
        "df": df
    }, filetype="csv", tag="warmup")

    for _, row in df.iterrows():
        bar = {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "price": row["price"],
            "volume": row["volume"]
        }
        bar_history.add_bar(bar)
        
    print(" Warm-up successful — data loaded and converted to EST.  Time to initialize the Strategy class...")


# Call the function to test
if __name__ == "__main__":
    if MODE == "LIVE":
        print("Live mode selected.")
        setup_signalr_connection()

    elif MODE == "SIM":
        print("Simulation mode selected.")
        simulate_trades()

    elif MODE == "BACKTEST":
        print("Backtest mode selected.")
        run_backtest()

    else:
        raise ValueError(f"Unsupported MODE '{MODE}'. Use BACKTEST, SIM, or LIVE.")



