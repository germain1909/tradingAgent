# futures_trading_env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd

class FuturesTradingEnv(gym.Env):
    """
    A simple futures trading environment.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, initial_balance=100000,asset_name:str = "GC4",stop_loss: float    = 300.0,take_profit: float  = 600.0,):
        super(FuturesTradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.stop_loss       = stop_loss
        self.take_profit     = take_profit

        # Define action and observation space
        # Action space: Discrete number of contracts to buy (negative for sell)
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.int32)

        # ——————————————————————————————————————————————
        # Build your feature list (order must match exactly what you engineered)
        self.feature_cols = [
            "price",       # raw price
            "balance",     # cash on hand

            # 1-minute indicators
            "ema_50", "ema_200", "vwap", "rsi_1m",

            # 5-minute indicators
            "ema_7_5m", "ema_17_5m", "ema_33_5m",
            "macd_5m", "macd_signal_5m", "macd_hist_5m",
            "rsi_5m", "rsi_ma_5m",

            # 15-minute indicators
            "macd_15m", "macd_signal_15m", "macd_hist_15m",
        ]

        # Observation space: example - price and cash balance
        self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(self.feature_cols),),
        dtype=np.float32
        )

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # number of contracts held
        self.done = False

        # Trade bookkeeping
        if "symbol" not in data.columns:
            raise ValueError("DataFrame must have a 'symbol' column")
        self.asset = data["symbol"].iloc[0]
        self.balance = initial_balance
        self.trades      = []   # completed trades
        self._open_trade = None # currently open
        

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        # 3a) start each episode with fresh trade logs
        self.trades      = []
        self._open_trade = None
        return self._get_obs()

    def _get_obs(self):
        # pull the current row of data
        row = self.data.iloc[self.current_step]

        # start with price & cash balance
        obs = [row["price"], self.balance]

        # append each engineered feature in order
        for feat in self.feature_cols[2:]:
            obs.append(row[feat])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # a) if already done
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # b) get today's bar
        bar = self.data.iloc[self.current_step]
        price    = bar["price"]    # usually 'close' of previous bar
        prev_pos = self.position

        # c) apply the agent's action (open/close) as before
        trade_contracts = int(action[0])
        cost            = trade_contracts * price * 100
        self.position  += trade_contracts
        self.balance   -= cost

        # record new opens
        if prev_pos == 0 and self.position != 0:
            self._open_trade = {
                "asset":       self.asset,
                "contracts":   self.position,
                "entry_price": price,
                "entry_step":  self.current_step,
            }

        # d) advance to next bar
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_bar = self.data.iloc[self.current_step]
        high = next_bar["high"]
        low  = next_bar["low"]

        # e) intra-bar SL/TP check
        if prev_pos != 0 and self._open_trade is not None:
            contracts = self._open_trade["contracts"]
            entry     = self._open_trade["entry_price"]
            mul       = contracts * 100

            # compute the price at which our thresholds lie
            sl_price = entry - (self.stop_loss / mul)   # e.g. entry - 300/(contracts*100)
            tp_price = entry + (self.take_profit / mul)

            hit_sl = (contracts > 0 and low  <= sl_price) or \
                    (contracts < 0 and high >= tp_price)
            hit_tp = (contracts > 0 and high >= tp_price) or \
                    (contracts < 0 and low  <= sl_price)

            if hit_sl or hit_tp:
                # decide which threshold we hit first:                
                if hit_sl:
                    exit_price = sl_price
                    pnl        = -self.stop_loss
                else:
                    exit_price = tp_price
                    pnl        =  self.take_profit

                # settle and log
                self.balance += contracts * exit_price * 100
                closed = {
                    **self._open_trade,
                    "exit_price": exit_price,
                    "exit_step":  self.current_step,
                    "pl":         pnl,
                    "timestamp":  next_bar["datetime"].isoformat(),
                }
                self.trades.append(closed)

                # flatten
                self.position    = 0
                self._open_trade = None

                obs = self._get_obs()
                info = {
                    "balance":       self.balance,
                    "position":      self.position,
                    "current_price": exit_price,
                    "last_trade":    closed,
                    "stopped":       True,
                }
                return obs, pnl, self.done, info

        # f) normal reward on close-to-close
        next_price = next_bar["price"]
        reward     = self.position * (next_price - price) * 100

        obs = self._get_obs()
        info = {
            "balance":       self.balance,
            "position":      self.position,
            "current_price": next_price,
        }

        # g) log any agent-initiated close
        if prev_pos != 0 and self.position == 0 and self._open_trade is not None:
            tr = {
                **self._open_trade,
                "exit_price":  price,
                "exit_step":   self.current_step,
                "pl":          contracts * (price - entry) * 100,
                "timestamp":   next_bar["datetime"].isoformat(),
            }
            self.trades.append(tr)
            self._open_trade = None
            info["last_trade"] = tr

        return obs, reward, self.done, info





    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}"
        )
