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

    def __init__(self, data: pd.DataFrame, initial_balance=100000):
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

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        # Example observation: current price and cash balance
        price = self.data.loc[self.current_step, "price"]
        return np.array([price, self.balance], dtype=np.float32)

    def step(self, action):
        """
        action: scalar indicating contracts to buy/sell (negative means sell)
        """

        if self.done:
            return self._get_obs(), 0, self.done, {}

        price = self.data.loc[self.current_step, "price"]

        # Apply action: buy/sell contracts
        trade_contracts = int(action[0])

        # Calculate cost and update position and balance
        cost = trade_contracts * price * 100  # assuming contract multiplier 100
        self.position += trade_contracts
        self.balance -= cost

        # Move to next time step
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Reward: unrealized PnL + balance change (simplified)
        next_price = self.data.loc[self.current_step, "price"]
        unrealized_pnl = self.position * (next_price - price) * 100

        reward = unrealized_pnl

        obs = self._get_obs()
        info = {
            "balance": self.balance,
            "position": self.position,
            "current_price": next_price,
        }

        return obs, reward, self.done, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}"
        )
