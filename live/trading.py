# trader.py
from signalrcore.hub_connection_builder import HubConnectionBuilder
import logging
import time
from util.bar_aggregator import BarAggregator
import random
from datetime import datetime
from util.BarFileWriter import BarFileWriter
from util.BarHistory import BarHistory
from functools import partial
from util.HistoricalFetcher import HistoricalFetcher




ASSET_ID = "CON.F.US.GCE.Q25"
OUTPUT_DIR = "data/month_json"
TOKEN = "TOKEN"
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

LIVE_MODE = False  # Set to True when market is open
CONTRACT_ID = "CON.F.US.GCE.Q25"

# Instantiate the aggregator once
aggregator = BarAggregator()
bar_history = BarHistory()
fetcher = HistoricalFetcher(ASSET_ID, OUTPUT_DIR, HEADERS)




def trade():
    print("traded")

def bar_to_json(bar):
    return {
        "t": bar["timestamp"].isoformat(),
        "o": bar["open"],
        "h": bar["high"],
        "l": bar["low"],
        "c": bar["close"],
        "v": bar["volume"],
    }


def on_gateway_trade(args,writer=None):
    print("[RawGatewayTrade]", args[0])  # print raw trade data from the market
    raw_trade = args[0]  # Extract actual trade
    trade = {
        "price": raw_trade["Price"],
        "volume": raw_trade["Volume"],
        "timestamp": raw_trade["UtcTimestamp"]
    }
    print("[GatewayTrade]", trade)
    aggregator.on_trade(trade)

     # Try to get a completed bar (1-min)
    # will either return a bar on None which is falsy 
    new_bar = aggregator.get_latest_completed_bar()  

    if new_bar:
        #TODO we need to update to handle TOPSTEP data structure
        bar_json = bar_to_json(new_bar)
        print(bar_json)
        bar_history.add_bar(bar_json)
        if writer:
            writer.append_bar(bar_json)
    else:
        print("No completed bars yet.")



def setup_signalr_connection():
    writer = BarFileWriter("bars.json", max_lines=1000)
    hub_connection = HubConnectionBuilder()\
        .with_url(
            "https://rtc.topstepx.com/hubs/market",
            options={
                "access_token_factory": lambda: JWT_TOKEN
            })\
        .configure_logging(logging.INFO)\
        .with_automatic_reconnect({
            "type": "raw",
            "keep_alive_interval": 10,
            "reconnect_interval": 5,
            "max_attempts": 5
        })\
        .build()

    # Register callbacks
    hub_connection.on("GatewayTrade", partial(on_gateway_trade, writer=writer))
    # hub_connection.on("GatewayQuote", on_gateway_quote)
    # hub_connection.on("GatewayDepth", on_gateway_depth)

    # Start the connection
    hub_connection.start()

    # Subscribe to the contract's data
    # hub_connection.send("SubscribeContractQuotes", [CONTRACT_ID])
    hub_connection.send("SubscribeContractTrades", [CONTRACT_ID])
    # hub_connection.send("SubscribeContractMarketDepth", [CONTRACT_ID])

    print("SignalR connection established. Subscribed to market data.")

    # Keep the connection alive
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
                "ContractId": CONTRACT_ID,
                "Price": round(price, 2),
                "Volume": volume,
                "UtcTimestamp": datetime.utcnow().isoformat() + "Z"
            }

            print("[MockTrade]", mock_trade)
            on_gateway_trade([mock_trade],writer=writer)

            ticks += 1
            time.sleep(1.5)  # 1.5 sec between trades
    except KeyboardInterrupt:
        print("🛑 Simulation interrupted.")

# Call the function to test
if __name__ == "__main__":
    if LIVE_MODE:
        setup_signalr_connection()
    else:
        simulate_trades()



