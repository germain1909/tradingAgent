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

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float    = 100_000,
        asset_name: str           = "GC4",
        stop_loss: float          = 300.0,
        take_profit: float        = 600.0,
        max_daily_trades: int     = 5,       # ← daily cap
    ):    
        super(FuturesTradingEnv, self).__init__()

        # ─── raw data & risk params ────────────────────────────────
        self.data            = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.stop_loss       = stop_loss
        self.take_profit     = take_profit

        # ─── daily‐trade throttling ────────────────────────────────
        self.max_daily_trades = max_daily_trades

        # ─── action & observation space ────────────────────────────
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.int32)

        self.feature_cols = [
            "price","balance",
            "ema_50","ema_200","vwap","rsi_1m",
            "ema_7_5m","ema_17_5m","ema_33_5m",
            "macd_5m","macd_signal_5m","macd_hist_5m","rsi_5m","rsi_ma_5m",
            "macd_15m","macd_signal_15m","macd_hist_15m",
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32
        )

        # ─── trade bookkeeping ──────────────────────────────────────
        if "symbol" not in data.columns:
            raise ValueError("DataFrame must have a 'symbol' column")
        self.asset       = data["symbol"].iloc[0]
        self.trades      = []    # completed trades
        self._open_trade = None  # currently open trade

        # ─── episode state (will also reset in `reset()`) ───────────
        self.current_step  = 0
        self.balance       = self.initial_balance
        self.position      = 0
        self.done          = False

        # ─── daily‐throttle state ───────────────────────────────────
        self.trade_count   = 0
        # assume your DataFrame has a `datetime` column you reset_index’d back in
        first_dt = self.data.loc[0, "datetime"]
        self.current_day   = pd.to_datetime(first_dt).date()
        

    def reset(self):
        self.current_step  = 0
        self.balance       = self.initial_balance
        self.position      = 0
        self.done          = False
        self.trades        = []
        self._open_trade   = None

        self.trade_count   = 0
        first_dt           = self.data.loc[0, "datetime"]
        self.current_day   = pd.to_datetime(first_dt).date()

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

    def _check_sl_tp(self, next_price):
        """
        Returns (obs, reward, done, info) if a stop-loss or take-profit
        fires on the *next_price*, else None.
        """
        if self._open_trade is None:
            return None

        contracts = self._open_trade["contracts"]
        entry     = self._open_trade["entry_price"]
        pnl       = contracts * (next_price - entry) * 100

        if pnl <= -self.stop_loss or pnl >= self.take_profit:
            # clamp P/L exactly at your thresholds
            if pnl <= -self.stop_loss:
                pnl = -self.stop_loss
                exit_price = entry + pnl / (contracts*100)
            else:
                pnl = self.take_profit
                exit_price = entry + pnl / (contracts*100)

            # settle P/L & log
            self.balance += contracts * exit_price * 100
            closed = {
                **self._open_trade,
                "exit_price":  exit_price,
                "exit_step":   self.current_step,
                "pl":          pnl,
                "timestamp":   self.data.iloc[self.current_step]["datetime"].isoformat(),
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

        return None
    
    def _throttle_daily(self, desired_contracts: int, prev_pos: int) -> int:
        # 1) Grab the bar’s datetime and calendar date
        bar_dt = pd.to_datetime(self.data.loc[self.current_step, "datetime"])
        today  = bar_dt.date()

        # 2) Show us everything we know at this moment
        print(f"[THROTTLE] step={self.current_step} now={bar_dt.isoformat()} "
            f"current_day={self.current_day} trade_count={self.trade_count} "
            f"prev_pos={prev_pos} desired={desired_contracts}")

        # 3) Reset at midnight (UTC) if needed
        if today != self.current_day:
            print("  → New UTC day! Resetting counter.")
            self.current_day = today
            self.trade_count = 0

        # 4) Only count real _entries_ (flat→nonzero)
        if prev_pos == 0 and desired_contracts != 0:
            print("  → Detected an entry attempt.")
            if self.trade_count >= self.max_daily_trades:
                print("  → DAILY CAP REACHED: blocking this trade!")
                return 0
            self.trade_count += 1
            print(f"  → Allowing entry #{self.trade_count} today.")

        # 5) Everything else passes through
        return desired_contracts
    
    def step(self, action):
        # 1) done‐check
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # 2) snapshot
        prev_pos = self.position
        bar      = self.data.iloc[self.current_step]
        price    = bar["price"]

        # 3) throttle daily entries
        desired_ct = int(action[0])
        trade_ct   = self._throttle_daily(desired_ct, prev_pos)

        # 4) apply that (possibly zeroed) trade
        cost           = trade_ct * price * 100
        self.position += trade_ct
        self.balance  -= cost

        # 5) record new open
        if prev_pos == 0 and self.position != 0:
            self._open_trade = {
                "asset":       self.asset,
                "contracts":   self.position,
                "entry_price": price,
                "entry_step":  self.current_step,
            }

        # 6) advance time
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_bar   = self.data.iloc[self.current_step]
        next_price = next_bar["price"]

        # 7) stop‐loss / take‐profit check (reuse your existing helper)
        sltp = self._check_sl_tp(next_price)
        if sltp is not None:
            return sltp

        # 8) normal reward & info
        reward = self.position * (next_price - price) * 100
        obs    = self._get_obs()
        info   = {
            "balance":       self.balance,
            "position":      self.position,
            "current_price": next_price,
        }

        # 9) natural agent‐initiated close logging
        if prev_pos != 0 and self.position == 0 and self._open_trade is not None:
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

        return obs, reward, self.done, info




    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}"
        )
