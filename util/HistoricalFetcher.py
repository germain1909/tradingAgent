
import requests
import os
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import sys
import pytz


WARMUP_PERIOD_FETCHER = 10
class HistoricalFetcher:
    def __init__(self, asset_id, output_dir, headers):
        self.asset_id = asset_id
        self.output_dir = output_dir
        self.headers = headers
        self.warm_up_complete = False  # Flag for external checks

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_and_save_day(self, date_str):
        start = f"{date_str}T00:00:00Z"
        end   = f"{date_str}T23:59:00Z"

        payload = {
            "contractId": self.asset_id,
            "live": False,
            "startTime": start,
            "endTime": end,
            "unit": 2,  # 2 = 1-minute bars
            "unitNumber": 1,
            "limit": 960,
            "includePartialBar": False
        }

        response = requests.post(
            "https://api.topstepx.com/api/History/retrieveBars", 
            headers=self.headers, 
            json=payload
        )

        if response.status_code == 200:
            bars = response.json()
            filename = f"{self.asset_id.split('.')[-1]}_{date_str}.json"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(bars, f, indent=2)
            print(f"✅ Saved: {filepath}")
            return True
        else:
            print(f"❌ Failed for {date_str}: {response.status_code} {response.text}")
            return False
    
    def get_past_hours_session_times(self, hours=10):
        eastern = pytz.timezone("US/Eastern")
        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(eastern)

        # Compute the earliest time you need
        earliest_needed_et = now_et - timedelta(hours=hours)

        # Find session start for current and previous sessions
        session_starts = []
        cursor = now_et
        while True:
            if cursor.hour < 18:
                session_start = cursor.replace(hour=18, minute=0, second=0, microsecond=0) - timedelta(days=1)
            else:
                session_start = cursor.replace(hour=18, minute=0, second=0, microsecond=0)
            session_starts.append(session_start)
            if session_start <= earliest_needed_et:
                break
            cursor = session_start - timedelta(seconds=1)  # Go back to just before previous session

        # The *earliest* session start that covers the needed window
        start_et = session_starts[-1]
        start_utc = start_et.astimezone(timezone.utc)
        now_utc_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_utc_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        return start_utc_str, now_utc_str
    

    def run_batch_fetch(self, start_date, end_date):
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end     = datetime.strptime(end_date, "%Y-%m-%d")

        all_success = True
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            success = self.fetch_and_save_day(date_str)
            if not success:
                all_success = False
            current += timedelta(days=1)

        self.warm_up_complete = all_success
        if all_success:
            print("✅ Warm-up data fetch complete.")
        else:
            print("⚠️ Warm-up data fetch completed with errors.")

    def fetch_past_hours(self, hours= WARMUP_PERIOD_FETCHER) -> pd.DataFrame:
        """
        Fetch historical bars for the past `hours` hours ending 1 minute before now.
        Returns the data as a pandas DataFrame.
        """

        date_str = "2025-07-09"
        now = datetime.now(timezone.utc)
        end_time = now
        start_time = f"{date_str}T23:00:00Z"
        print(end_time)

        payload = {
            "contractId": self.asset_id,
            "live": False,
            "startTime": start_time,
            "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": 2,  # 1-minute bars
            "unitNumber": 1,
            "limit": hours * 60,
            "includePartialBar": False
        }
        print(payload)
        response = requests.post(
            "https://api.topstepx.com/api/History/retrieveBars",
            headers=self.headers,
            json=payload
        )
        print(response)
        if response.status_code == 200:
            raw = response.json()
            print(raw)
            bars = raw["bars"][::-1]
            
            # Save to file if you want for backup
            filename = f"{self.asset_id.split('.')[-1]}_{start_time}_to_{end_time}.json"
            filepath = os.path.join(self.output_dir, filename)

            # Create directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            with open(filepath, "w") as f:
                json.dump({"bars": bars}, f, indent=2)
            print(f"✅ Saved: {filepath}")

            # Convert to DataFrame

            df = pd.DataFrame(bars).rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "price",
                "v": "volume",
            })
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            self.warm_up_complete = True
            return df
        else:
            print(f"❌ Failed to fetch past {hours} hours: {response.status_code} {response.text}")
            self.warm_up_complete = False
            sys.exit(1)  # Quit immediately with exit code 1


    def fetch_bars_by_time(self, start_time, end_time, limit=600) -> pd.DataFrame:
        """
        Fetch historical bars between explicit start_time and end_time (inclusive).
        `start_time` and `end_time` can be either ISO format strings or datetime objects.
        """
        # Handle datetime or string input
        if isinstance(start_time, datetime):
            start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start_time_str = str(start_time)
        if isinstance(end_time, datetime):
            end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end_time_str = str(end_time)

        payload = {
            "contractId": self.asset_id,
            "live": False,
            "startTime": start_time_str,
            "endTime": end_time_str,
            "unit": 2,  # 1-minute bars
            "unitNumber": 1,
            "limit": limit,
            "includePartialBar": False
        }
        print(payload)
        response = requests.post(
            "https://api.topstepx.com/api/History/retrieveBars",
            headers=self.headers,
            json=payload
        )
        print(response)
        if response.status_code == 200:
            raw = response.json()
            print(raw)
            bars = raw["bars"][::-1]
            
            # Save to file if you want for backup
            filename = f"{self.asset_id.split('.')[-1]}_{start_time_str}_to_{end_time_str}.json"
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(self.output_dir, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump({"bars": bars}, f, indent=2)
            print(f"✅ Saved: {filepath}")

            # Convert to DataFrame
            df = pd.DataFrame(bars).rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "price",
                "v": "volume",
            })
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            self.warm_up_complete = True
            return df
        else:
            print(f"❌ Failed to fetch bars: {response.status_code} {response.text}")
            self.warm_up_complete = False
            sys.exit(1)  # Quit immediately with exit code 1