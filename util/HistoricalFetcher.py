
import requests
import os
import json
from datetime import datetime, timedelta, timezone


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

    def fetch_past_hours(self, hours=10):
        """
        Fetch historical bars for the past `hours` hours ending 1 minute before now.
        """
        now = datetime.now(timezone.utc)  # Always in UTC for API compatibility
        end_time = now - timedelta(minutes=1)
        start_time = end_time - timedelta(hours=hours)

        payload = {
            "contractId": self.asset_id,
            "live": False,
            "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": 2,  # 1-minute bars
            "unitNumber": 1,
            "limit": hours * 60,
            "includePartialBar": False
        }

        response = requests.post(
            "https://api.topstepx.com/api/History/retrieveBars",
            headers=self.headers,
            json=payload
        )

        if response.status_code == 200:
            bars = response.json()
            filename = f"{self.asset_id.split('.')[-1]}_{start_time.strftime('%Y%m%d_%H%M')}_to_{end_time.strftime('%Y%m%d_%H%M')}.json"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(bars, f, indent=2)
            print(f"✅ Saved: {filepath}")
            self.warm_up_complete = True
        else:
            print(f"❌ Failed to fetch past {hours} hours: {response.status_code} {response.text}")
            self.warm_up_complete = False