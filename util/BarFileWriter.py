import json

class BarFileWriter:
    def __init__(self, filename, max_lines=1000):
        self.filename = filename
        self.max_lines = max_lines
        self.lines_written = 0

    def append_bar(self, bar):
        # bar is a dict, convert to JSON string
        bar_json = json.dumps(bar)
        
        mode = "a"
        if self.lines_written >= self.max_lines:
            mode = "w"
            self.lines_written = 0
        
        with open(self.filename, mode) as f:
            f.write(bar_json + "\n")
            self.lines_written += 1