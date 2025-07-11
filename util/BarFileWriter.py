import json
import datetime
import pandas as pd

class BarFileWriter:
    def __init__(self, filename, max_lines=1000):
        self.filename = filename
        self.max_lines = max_lines
        self.lines_written = 0

    def serialize_bar(self, bar):
        # Make a copy so you don't mutate the original dict
        bar = bar.copy()
        ts = bar.get("timestamp")
        # Convert Timestamp/datetime to string
        if isinstance(ts, (pd.Timestamp, datetime.datetime)):
            bar["timestamp"] = ts.isoformat()
        else:
            bar["timestamp"] = str(ts)
        return bar

    def append_bar(self, bar):
        # bar is a dict, convert to JSON string
        serialized_bar = self.serialize_bar(bar)
        bar_json = json.dumps(serialized_bar)
        
        mode = "a"
        if self.lines_written >= self.max_lines:
            mode = "w"
            self.lines_written = 0
        
        with open(self.filename, mode) as f:
            f.write(bar_json + "\n")
            self.lines_written += 1