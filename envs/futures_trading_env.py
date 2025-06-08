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
        normalized_data: pd.DataFrame,
        initial_balance: float = 100_000,
        asset_name: str = "GC4",
        # stop_loss: float = 500.0,
        # take_profit: float = 1000.0,
    ):
        super(FuturesTradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.normalized_data = normalized_data.reset_index(drop=True)
        self.initial_balance = initial_balance

        # 2) risk-management params
        # self.stop_loss = stop_loss
        # self.take_profit = take_profit


        # Build your feature list (order must match exactly what you engineered)
        self.feature_cols = [
           
            # 1-minute indicators
            "open","high","low","price","volume","ema_50",
            "ema_50_slope","vwap","true_range_1m",
            "atr_14_1m",

            # 5-minute indicators
            "macd_5m", "macd_signal_5m",
            "rsi_5m", "rsi_ma_5m","true_range_5m",
            "atr_14_5m","bb_mid_5m","bb_std_5m","bb_upper_5m",
            "bb_lower_5m","macd_cross_5m",

            # 15-minute indicators
            "macd_15m", "macd_signal_15m","macd_cross_15m",
            "macd_slope_15m",
        ]


         # ←── HERE: declare per‐episode state (no values yet)
        self.current_step = None
        self.position     = None
        self.balance      = None
        self.trades       = None
        self._open_trade  = None
        self.done         = None

          # ─── initialize them once so _get_obs() works in __init__ ─────────
        initial_obs = self.reset()

         # Define action and observation space
        # Action space: Discrete number of contracts to buy (negative for sell)
        self.action_space = spaces.Discrete(3)

         # observation is whatever shape _get_obs() returned
        self.observation_space = spaces.Box(
        low=-np.inf,
        high= np.inf,
        shape=initial_obs.shape,
        dtype=np.float32
        )

         # observation is whatever shape _get_obs() returned
        self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=initial_obs.shape,
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
        row_normalized = self.normalized_data.iloc[self.current_step]

        # start with price & cash balance
        obs = [row["price"], self.balance]

        # append each engineered feature in order
        for feat in self.feature_cols:
            obs.append(row_normalized[feat])

        # DEBUG: print the observation
        obs_array = np.array(obs, dtype=np.float32)
        np.set_printoptions(suppress=True, precision=10)
        feature_names = ["price", "balance"] + self.feature_cols
        obs_named = [f"{name}: {val:.5f}" for name, val in zip(feature_names, obs_array)]
        print(f"[Step {self.current_step}] Observation:\n" + ", ".join(obs_named))
        # print(f"[Step {self.current_step}] Normalized Row:\n{row_normalized}")

        return np.array(obs, dtype=np.float32)



    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # --- Current bar data
        bar      = self.data.iloc[self.current_step]
        price    = bar["price"]
        prev_pos = self.position
        cross15  = bar["macd_cross_15m"]

        # --- Action normalization
        arr = np.array(action)
        raw = int(np.ravel(arr)[0])
        desired = {0: -1, 1: 0, 2: 1}[raw]

        # --- Entry filter
        if prev_pos == 0 and desired != 0:
            if (desired > 0 and cross15 != +1) or (desired < 0 and cross15 != -1):
                desired = 0

        # --- MACD exit
        if prev_pos > 0 and cross15 == -1:
            desired = 0
        elif prev_pos < 0 and cross15 == +1:
            desired = 0

        # --- Execute trade
        trade_contracts = desired - self.position
        cost            = trade_contracts * price * 100
        self.position  += trade_contracts
        self.balance   -= cost

        if prev_pos == 0 and self.position != 0:
            self._open_trade = {
                "asset":       self.asset,
                "contracts":   self.position,
                "entry_price": price,
                "entry_step":  self.current_step,
                "macd_cross":  cross15,
            }

        # --- Advance time
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_bar   = self.data.iloc[self.current_step]
        next_price = next_bar["price"]

        # --- Observation
        obs = self._get_obs()
        info = {
            "balance":       self.balance,
            "position":      self.position,
            "current_price": next_price,
        }

        # --- Stop-loss logic
        was_stopped_out = False
        if self._open_trade is not None and self.current_step > self._open_trade["entry_step"]:
            entry_price   = self._open_trade["entry_price"]
            contracts     = self._open_trade["contracts"]
            direction     = np.sign(contracts)
            price_diff    = (next_price - entry_price) * direction
            unrealized_pl = abs(contracts) * price_diff * 100

            if unrealized_pl <= -1000:
                self.balance += -contracts * next_price * 100
                self.position = 0
                was_stopped_out = True

                tr = {
                    **self._open_trade,
                    "exit_price":      next_price,
                    "exit_step":       self.current_step,
                    "macd_cross_exit": "stop",
                    "pl":              unrealized_pl,
                    "timestamp":       next_bar["datetime"].isoformat(),
                }
                self.trades.append(tr)
                self._open_trade = None
                info["last_trade"] = tr

        # --- Natural exit (skip if stopped)
        if not was_stopped_out and prev_pos != 0 and self.position == 0 and self._open_trade is not None:
            tr = {
                **self._open_trade,
                "exit_price":      price,
                "exit_step":       self.current_step,
                "macd_cross_exit": cross15,
                "pl":               self._open_trade["contracts"]
                                    * (price - self._open_trade["entry_price"]) * 100,
                "timestamp":       next_bar["datetime"].isoformat(),
            }
            self.trades.append(tr)
            self._open_trade = None
            info["last_trade"] = tr

        # --- Final forced exit
        if self.done and self._open_trade is not None:
            tr = {
                **self._open_trade,
                "exit_price":  price,
                "exit_step":   self.current_step,
                "pl":          self._open_trade["contracts"]
                            * (price - self._open_trade["entry_price"]) * 100,
                "timestamp":   next_bar["datetime"].isoformat(),
            }
            self.trades.append(tr)
            self._open_trade = None
            info["last_trade"] = tr

        # --- Reward
        if "last_trade" in info:
            pl = info["last_trade"]["pl"]
            reward = 1.0 if pl > 0 else -1.0
            if pl >= 1500:
                reward += 1.5
            elif pl >= 1000:
                reward += 1.0
            elif pl >= 500:
                reward += 0.5
            if pl <= -500:
                reward -= 1.0
            elif pl <= -300:
                reward -= 0.5
        else:
            reward = -0.5  # idle penalty

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
