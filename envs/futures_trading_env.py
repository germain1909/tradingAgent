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
        unrealized_pnl = 0,
        stop_loss: float = 500.0,
        take_profit: float = 1000.0,
        contract_size = 100,
    ):
        super(FuturesTradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.normalized_data = normalized_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.unrealized_pnl = unrealized_pnl
        self.contract_size = contract_size

        # 2) risk-management params
        self.stop_loss = stop_loss
        self.take_profit = take_profit


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
        obs = [row["price"], self.balance, self.unrealized_pnl,self.position]

        # append each engineered feature in order
        for feat in self.feature_cols:
            obs.append(row_normalized[feat])

        # DEBUG: print the observation
        obs_array = np.array(obs, dtype=np.float32)
        np.set_printoptions(suppress=True, precision=10)
        feature_names = ["price", "balance","unrealized_pnl","position"] + self.feature_cols
        obs_named = [f"{name}: {val:.5f}" for name, val in zip(feature_names, obs_array)]
        # print(f"[Step {self.current_step}] Observation:\n" + ", ".join(obs_named))
        # print(f"[Step {self.current_step}] Normalized Row:\n{row_normalized}")

        return np.array(obs, dtype=np.float32)



    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # --- Current bar data
        bar      = self.data.iloc[self.current_step]
        normalized_bar = self.normalized_data.iloc[self.current_step]
        price    = bar["price"]
        prev_pos = self.position
        cross15  = bar["macd_cross_15m"]
        info = {}
     # Map the action directly
        action_map = {
            0: "hold",
            1: "open_long",
            2: "close_long",
            # 3: "open_short",
            # 4: "close_short"
        }
        action_name = action_map[action]

        if self._open_trade is not None:
            entry_price = self._open_trade["entry_price"]
            position = self.position
            self.unrealized_pnl =(price - entry_price) * position * self.contract_size
        else:
            self.unrealized_pnl = 0

        #OPEN TRADES

        # Prevents position flipping
        if action_name == "open_long":
            if self.position == 0:
                self.position = 1
                self._open_trade = {
                    "entry_price": price,
                    "entry_step": self.current_step,
                    "side": "long"
                }
       
        if action_name == "close_long":
            if self.position == 1:
                info["last_trade"] = self._close_trade(price, "AI")
                self.position = 0
                self._open_trade = None
            else:
                print(f"[Step {self.current_step}] ⚠️ Ignored invalid action: close_long with no long position")
                unavailable_reward = -20
                obs = self._get_obs()
                self.current_step += 1  # advance time to avoid infinite loop
                return obs, unavailable_reward, self.done, info  # 🛑 return early without progressing


        # if action_name == "open_short":
        #     if self.position == 0:
        #         self.position = -1
        #         self._open_trade = {
        #             "entry_price": price,
        #             "entry_step": self.current_step,
        #             "side": "short"
        #         }
        # if action_name == "close_short":
        #     if self.position == -1:
        #         info["last_trade"] = self._close_trade(price,"AI")
        #         self.position = 0
        #         self._open_trade = None
        #     else:
        #         print(f"[Step {self.current_step}] ⚠️ Ignored invalid action: close_short with no short position")
        #         obs = self._get_obs()
        #         unavailable_reward = -20
        #         self.current_step += 1  # advance time to avoid infinite loop
        #         return obs, unavailable_reward, self.done, info  # 🛑 return early without progressing

        
        print(f"[Step {self.current_step}] Action: {action_name} | Position: {self.position} | PnL: {self.unrealized_pnl:.2f} | Balance: {self.balance:.2f}")


        # CLOSE TRADES 

        if self._open_trade is not None:
            side         = self._open_trade["side"]
            if side == "long":
                if self.unrealized_pnl <= -self.stop_loss:
                    info["last_trade"] = self._close_trade(price,"stop loss")
                    self.position = 0
                    self._open_trade = None
                    print(f"Stop loss hit on LONG at step {self.current_step}")
                
                elif self.unrealized_pnl >= self.take_profit:
                    info["last_trade"] = self._close_trade(price,"take profit")
                    self.position = 0
                    self._open_trade = None
                    print(f"Take profit hit on LONG at step {self.current_step}")

            # elif side == "short":

            #     if self.unrealized_pnl <= -self.stop_loss:
            #         info["last_trade"] = self._close_trade(price,"stop loss")
            #         self.position = 0
            #         self._open_trade = None
            #         print(f"Stop loss hit on SHORT at step {self.current_step}")
                
            #     elif self.unrealized_pnl >= self.take_profit:
            #         info["last_trade"] = self._close_trade(price,"take profit")
            #         self.position = 0
            #         self._open_trade = None
            #         print(f"Take profit hit on SHORT at step {self.current_step}")


         # --- Advance time
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_bar   = self.data.iloc[self.current_step]
        next_price = next_bar["price"]

        # --- Observation
        obs = self._get_obs()
        info["balance"] = self.balance
        info["position"] = self.position
        info["current_price"] = next_price 
        # --- Reward
        # Default reward
        reward = -1.0  # mild penalty for doing nothing (hold)

        # Penalty for invalid action
        if action_name == "close_long" and self.position != 1:
            reward = -10.0  # trying to close with no open long
            return self._get_obs(), reward, self.done, info

        # If a trade was closed, calculate profit/loss
        if "last_trade" in info:
            pl = info["last_trade"]["pl"]

            # Scale profit/loss for strong signal
            reward = pl / 100.0

            # Bonus for hitting big profits
            if pl >= 1000:
                reward += 10.0
            elif pl >= 500:
                reward += 5.0
            elif pl > 0:
                reward += 1.0

            # Penalty for losses
            if pl <= -1000:
                reward -= 10.0
            elif pl < 0:
                reward -= 5.0

        # Optional: small penalty every step with open position (to discourage holding too long)
        elif self.position == 1:
            reward = -0.1
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

    def _close_trade(self, price: float, closeType: str):
        #"""Close the current open trade and log the result."""
        if not self._open_trade:
            return

        entry_price = self._open_trade["entry_price"]
        position = self.position
        pl = (price - entry_price) * position * self.contract_size

        self.unrealized_pnl = pl
        self.balance += pl

        closed_trade = {
            "asset": self.asset,
            "clos_type": closeType,
            "contracts": abs(position),
            "side": self._open_trade["side"],
            "entry_price": entry_price,
            "exit_price": price,
            "pl": pl,
            "entry_step": self._open_trade["entry_step"],
            "exit_step": self.current_step,
            "timestamp": self.data.iloc[self.current_step]["datetime"],
            "macd_cross_15m": self.data.iloc[self._open_trade["entry_step"]]["macd_cross_15m"],
            "macd_cross_exit": self.data.iloc[self.current_step]["macd_cross_15m"]
        }

        self.trades.append(closed_trade)
        self._open_trade = None
        self.position = 0
        self.unrealized_pnl = 0
        return closed_trade