import os
import json
import pandas as pd
from util.SlackNotifier import SlackNotifier


#Slack Configuration 
SLACK_URL = "https://hooks.slack.com/services/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"
slack_notifier = SlackNotifier(SLACK_URL)

class Strategy:
    def __init__(self, executor=None,mode="BACKTEST"):
        self.trade = {}
        self.balance = 0
        self.unrealized_pnl = 0
        self.last_trade_time = None
        self.break_even_trigger = 500
        self.tick_size = 1
        self.trades = []
        self.mode = mode
        self.swings = []             # Stores detected swings
        self.last_swing_checked = 0  # Index of last bar checked for swings
        # self.executor = executor  

    def recent_macd_cross(self, df, current_time, column, lookback=20):
        """
        Check if a MACD cross occurred within the last `lookback` bars **before** current_time.
        Compatible with both backtest and live.
        """
        # Filter to only rows before the current_time
        recent = df[df["timestamp"] < current_time].tail(lookback)
        return recent[column].eq(1).any()


    def update_swings(self, df, lookback=5):
        """
        Incrementally update swings by checking only new bars.
        """
        # Only check new bars since last update
        start = max(self.last_swing_checked, lookback)
        end = len(df) - lookback  # Do not check if not enough future bars

        for i in range(start, end):
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            # Local high
            if high == df["high"].iloc[i-lookback:i+lookback+1].max():
                self.swings.append({'index': df.index[i], 'type': 'High', 'price': high})
            # Local low
            if low == df["low"].iloc[i-lookback:i+lookback+1].min():
                self.swings.append({'index': df.index[i], 'type': 'Low', 'price': low})
        self.last_swing_checked = end

    def get_last_swing(self, swings, current_index):
        swings_before = [s for s in swings if s['index'] < current_index]
        if not swings_before:
            return None
        return swings_before[-1]

    def in_cooldown(self, current_time):
        return self.last_trade_time is not None and (current_time - self.last_trade_time) < pd.Timedelta(minutes=15)

    def should_execute(self, bar, bar_history_df, current_time):
        price = bar["price"]
        ema_9_60m = bar["ema_9_60m"]
        ema_9_angle_60m = bar["ema_9_angle_60m"]
        ema_50_angle = bar["ema_50"]
        ema_7 = bar["ema_7_5m"]
        ema_33 = bar["ema_33_5m"]

        #  #Bullish Rececent crosses & conditions
        # recent_crosses = bar_history_df["macd_cross_bullish_5m"].iloc[-10:]
        # macd_recently_crossed_bullish = (recent_crosses == 1).any()
        # price_above_ema_7 = price > ema_7
        # price_above_ema_33 = price > ema_33



        # #Bearish Recent crosses and conditions
        # recent_bearish_crosses = bar_history_df["macd_cross_bearish_5m"].iloc[-10]
        # macd_recently_crossed_bearish = (recent_bearish_crosses == 1).any()

        #MACD Crosses 
        macd_recent_bullish = self.recent_macd_cross(bar_history_df, current_time, "macd_cross_bullish_5m")
        macd_recent_bearish = self.recent_macd_cross(bar_history_df, current_time, "macd_cross_bearish_5m")

        #Look for a recent highs/lows
        lookback = 30  # You can tune this as needed
        self.update_swings(bar_history_df, lookback=lookback)
        last_swing = self.get_last_swing(self.swings, current_time)


        

        if self.trade:
            return self.check_exit_conditions(bar, current_time)

        if not self.in_cooldown(current_time):
            print(
    f"🔎 Price: {price:.2f} | EMA_9_60m: {ema_9_60m:.2f} | EMA_9_Angle_60m: {ema_9_angle_60m:.2f} | "
    f"EMA_50: {ema_50_angle:.2f} | MACD Bullish 5m: {macd_recent_bullish} | MACD Bearish 5m: {macd_recent_bearish}"
)
            # Long setup
            if price > ema_9_60m and ema_9_angle_60m > 3:
                if macd_recent_bullish:
                    if last_swing and last_swing['type'] == 'Low':
                        print("📈 Long entry confirmed")
                        return self.enter_trade("buy", price, current_time, bar)
                    
            # Short setup
            if price < ema_9_60m and ema_9_angle_60m < -3:
                if macd_recent_bearish:
                    if last_swing and last_swing['type'] == 'High':
                        print("📉 Short entry confirmed")
                        return self.enter_trade("sell", price, current_time, bar)

        return False

    def enter_trade(self, direction, price, current_time, bar):
        stop_amount = 3  # $4 move = $400 loss per your PnL calc

        if direction == "buy":
            stop_loss_price = price - stop_amount
        else:  # sell
            stop_loss_price = price + stop_amount

        self.trade = {
            "entry_step": current_time,
            "entry_price": price,
            "entry_time": bar["timestamp"],
            "stop_loss_price": stop_loss_price,
            "break_even_set": False,
            "trade_direction": direction
        }
        self.last_trade_time = current_time

        # Slack notification if available
        # slack_notifier.send(f"🚀 Entered {direction.upper()} at {price} | SL: {stop_loss_price}")

        #if self.mode == "LIVE":
        # TODO: Send exit order via executor (live trading)
        # if self.executor:
        #     symbol = "GC2"
        #     qty = 1
        #     side = "sell" if self.trade["trade_direction"] == "buy" else "buy"
        #     self.executor.send_order(symbol=symbol, side=side, qty=qty, order_type="market")


        
        print(f"🚀 Entered {direction.upper()} at {price} | SL: {stop_loss_price}")

        return True



    def check_exit_conditions(self, bar, current_time):
        price = bar["price"]
        direction = self.trade["trade_direction"]

        self.update_unrealized_pnl(price)
        exit_price = None
        reason = None

        # Move SL to breakeven if unrealized PnL exceeds trigger
        if not self.trade["break_even_set"] and self.unrealized_pnl >= self.break_even_trigger:
            self.trade["stop_loss_price"] = self.trade["entry_price"]
            self.trade["break_even_set"] = True
            print(f"🔁 SL moved to breakeven at {self.trade['stop_loss_price']}")

        # Exit on profit target
        if self.unrealized_pnl >= 700:
            exit_price = price
            reason = "target hit"

        # Exit if stopped out, but check for breakeven stop condition
        elif direction == "buy" and price <= self.trade["stop_loss_price"]:
            exit_price = self.trade["stop_loss_price"]
            reason = "stopped out"
            if exit_price >= self.trade["entry_price"]:
                reason = "breakeven stop"

        elif direction == "sell" and price >= self.trade["stop_loss_price"]:
            exit_price = self.trade["stop_loss_price"]
            reason = "stopped out"
            if exit_price <= self.trade["entry_price"]:
                reason = "breakeven stop"

        # Finalize trade exit
        if exit_price is not None:
            self.exit_trade(exit_price, current_time, bar, reason)
            return True

        return False


    def update_unrealized_pnl(self, price):
        direction = self.trade["trade_direction"]
        entry_price = self.trade["entry_price"]
        if direction == "buy":
            self.unrealized_pnl = (price - entry_price) * 100
        else:
            self.unrealized_pnl = (entry_price - price) * 100

    def exit_trade(self, exit_price, current_time, bar, reason):
        self.trade.update({
            "exit_step": current_time,
            "exit_time": current_time,  # ✅ safer for backtest
            "exit_price": exit_price,
            "close_reason": reason
        })

        pnl = (
            (exit_price - self.trade["entry_price"]) * 100
            if self.trade["trade_direction"] == "buy"
            else (self.trade["entry_price"] - exit_price) * 100
        )
        self.trade["trade_pnl"] = pnl
        self.balance += pnl
        self.trade["balance"] = self.balance

        print(f"💰 {self.trade['trade_direction'].upper()} exit at {exit_price} | PnL: {pnl:.2f} | Reason: {reason}")
        # slack_notifier.send((f"💰 {self.trade['trade_direction'].upper()} exit at {exit_price} | PnL: {pnl:.2f} | Reason: {reason}"))

        #if self.mode == "LIVE":
        # TODO: Send exit order via executor (live trading)
        # if self.executor:
        #     symbol = "GC2"
        #     qty = 1
        #     side = "sell" if self.trade["trade_direction"] == "buy" else "buy"
        #     self.executor.send_order(symbol=symbol, side=side, qty=qty, order_type="market")

        self.trades.append(self.trade)
        self.log_trade()  # 👈 log to file immediately
        self.trade = {}
        self.unrealized_pnl = 0


    def log_trade(self, filename="logs/live_backtest_trades_log.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Write full trade history to file
        with open(filename, "w") as f:
            json.dump(self.trades, f, indent=4, default=str)



def is_long_allowed(price, ema_9_60m, ema_9_angle_60m):
    return price > ema_9_60m and ema_9_angle_60m > 3

def is_short_allowed(price, ema_9_60m, ema_9_angle_60m):
    return price < ema_9_60m and ema_9_angle_60m < -3

