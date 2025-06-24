from datetime import datetime

class BarAggregator:
    MAX_BARS = 200
    def __init__(self):
        self.current_bar = None
        self.bars = []

    def get_bar_timestamp(self, timestamp_str):
        dt = datetime.fromisoformat(timestamp_str.rstrip('Z'))
        return dt.replace(second=0, microsecond=0)

    def on_trade(self, trade):
        price = trade['price']
        quantity = trade['quantity']
        timestamp = trade['timestamp']

        bar_time = self.get_bar_timestamp(timestamp)

        if not self.current_bar or self.current_bar['timestamp'] != bar_time:
            if self.current_bar:
                self.bars.append(self.current_bar)
                # Keep bars list within MAX_BARS length
                if len(self.bars) > self.MAX_BARS:
                    self.bars.pop(0)  # Remove oldest bar
            self.current_bar = {
                'timestamp': bar_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': quantity
            }
        else:
            self.current_bar['high'] = max(self.current_bar['high'], price)
            self.current_bar['low'] = min(self.current_bar['low'], price)
            self.current_bar['close'] = price
            self.current_bar['volume'] += quantity

    def get_completed_bars(self):
        return self.bars + ([self.current_bar] if self.current_bar else [])