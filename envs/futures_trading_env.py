# futures_trading_env.py
import random
import gym
from gym import spaces
import numpy as np
import pandas as pd

class FuturesTradingEnv(gym.Env):
    """
    A simple futures trading environment.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100_000,
        asset_name: str = "GC4",
        stop_loss: float = 500.0,
        take_profit: float = 1000.0,
    ):
        super(FuturesTradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance

        # 2) risk-management params
        self.stop_loss = stop_loss
        self.take_profit = take_profit

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

        # b) snapshot
        bar       = self.data.iloc[self.current_step]
        price     = bar["price"]
        prev_pos  = self.position
        cross5    = bar["macd_cross_5m"]

        # desired from the agent
        desired = int(action[0])

        # only allow new entries on a MACD cross
        if prev_pos == 0 and desired != 0:
            if desired > 0 and cross5 != +1:
                desired = 0
            elif desired < 0 and cross5 != -1:
                desired = 0

        # c) apply action (use filtered desired) trade_contracts = desired
        trade_contracts = desired
        cost            = trade_contracts * price * 100
        self.position  += trade_contracts
        self.balance   -= cost

        # d) record open (flat→nonzero)
        if prev_pos == 0 and self.position != 0:
            self._open_trade = {
                "asset":       self.asset,
                "contracts":   self.position,
                "entry_price": price,
                "entry_step":  self.current_step,
                "macd_cross":   cross5,
            }

        # e) advance time & only end at final bar
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # f) look ahead price for P/L
        next_bar    = self.data.iloc[self.current_step]
        next_price  = next_bar["price"]

        # g) compute clipped reward (but DON'T end episode here)
        unrealized_pnl = self.position * (next_price - price) * 100
        reward = np.clip(unrealized_pnl, -self.stop_loss, self.take_profit)

        # h) base info & obs
        obs = self._get_obs()
        info = {
            "balance":       self.balance,
            "position":      self.position,
            "current_price": next_price,
        }

        # i) record natural close (nonzero→flat)
        if prev_pos != 0 and self.position == 0 and self._open_trade is not None:
            tr = {
                **self._open_trade,
                "exit_price":  price,
                "exit_step":   self.current_step,
                "macd_cross_exit": bar["macd_cross_5m"],
                "pl":          self._open_trade["contracts"] 
                             * (price - self._open_trade["entry_price"]) * 100,
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


    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        """
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
