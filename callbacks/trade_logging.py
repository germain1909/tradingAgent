import json
from stable_baselines3.common.callbacks import BaseCallback

class TradeLoggingJSONCallback(BaseCallback):
    def __init__(self, log_path="trades.json", verbose=0):
        super().__init__(verbose)
        self.log_path  = log_path
        self.trade_log = []
        self.aggregate_pl = 0.0   # ← new running total


    def _on_step(self) -> bool:
        if self.verbose and "rewards" in self.locals:
            print("sample rewards:", self.locals["rewards"][:5])
            
        for info in self.locals.get("infos", []):
            tr = info.get("last_trade")
            if tr is not None:
                # 1) store per‐trade record
                self.trade_log.append(tr)
                # 2) update aggregate
                self.aggregate_pl += tr["pl"]

                # 3) decode the macd cross flag
                mc = tr.get("macd_cross_5m", 0)
                if mc ==  1: cross_str = "bullish"
                elif mc == -1: cross_str = "bearish"
                else:         cross_str = "none"

                # 4) print everything
                if self.verbose:
                    print(
                        f"[Trade Closed] {tr['asset']} | "
                        f"{tr['contracts']} @ {tr['entry_price']} → {tr['exit_price']} | "
                        f"Trade P/L={tr['pl']:.2f} | "
                        f"MACD entry={tr.get('macd_cross')} exit={tr.get('macd_cross_exit')} | "
                        f"Aggregate P/L={self.aggregate_pl:.2f} | "
                        f"Time={tr['timestamp']}"
                    )
        return True

    def _on_training_end(self) -> None:
        # write per-trade log
        with open(self.log_path, "w") as f:
            json.dump(self.trade_log, f, default=str, indent=2)
        if self.verbose:
            print(f"\nFinal Aggregate P/L = {self.aggregate_pl:.2f}")
            print(f"Saved {len(self.trade_log)} trades to {self.log_path}")
