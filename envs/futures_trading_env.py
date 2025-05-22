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

    def __init__(self, data: pd.DataFrame, initial_balance=100000,asset_name:str = "GC4"):
        super(FuturesTradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Define action and observation space
        # Action space: Discrete number of contracts to buy (negative for sell)
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.int32)

        # Observation space: example - price and cash balance
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(2,), dtype=np.float32
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
        # Example observation: current price and cash balance
        price = self.data.loc[self.current_step, "price"]
        return np.array([price, self.balance], dtype=np.float32)

    def step(self, action):
        # a) if already done
        if self.done:
            return self._get_obs(), 0, True, {}

        # b) snapshot
        price    = self.data.loc[self.current_step, "price"]
        prev_pos = self.position

        # c) apply action
        trade_contracts = int(action[0])
        cost            = trade_contracts * price * 100
        self.position  += trade_contracts
        self.balance   -= cost

        # d) record open (flat→nonzero)
        if prev_pos == 0 and self.position != 0:
            self._open_trade = {
                "asset":        self.asset,
                "contracts":    self.position,
                "entry_price":  price,
                "entry_step":   self.current_step,
            }

        # e) advance time
        self.current_step += 1
        if self.current_step >= len(self.data)-1:
            self.done = True

        # f) compute reward
        next_price     = self.data.loc[self.current_step, "price"]
        reward         = self.position * (next_price - price) * 100

        # g) base info
        obs = self._get_obs()
        info = {
            "balance":       self.balance,
            "position":      self.position,
            "current_price": next_price,
        }

        # h) record close (nonzero→flat)
        if prev_pos != 0 and self.position == 0 and self._open_trade is not None:
            tr = {
                **self._open_trade,
                "exit_price": price,
                "exit_step":  self.current_step,
                "pl":         self._open_trade["contracts"] 
                            * (price - self._open_trade["entry_price"]) * 100,
                "timestamp":  self.data.loc[self.current_step, "t"],
            }
            self.trades.append(tr)
            self._open_trade = None
            info["last_trade"] = tr

        return obs, reward, self.done, info


    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}"
        )
