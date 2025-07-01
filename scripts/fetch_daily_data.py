import os
import json
import requests
from datetime import datetime, timedelta

# === CONFIG ===
ASSET_ID = "CON.F.US.GCE.Q25"
OUTPUT_DIR = "data/month_json"
TOKEN = "ADD_TOKEN_HERE"  # ← replace with your real token
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

def fetch_and_save_day(date_str):
    start = f"{date_str}T00:00:00Z"
    end   = f"{date_str}T23:59:00Z"

    payload = {
        "contractId": ASSET_ID,
        "live": False,
        "startTime": start,
        "endTime": end,
        "unit": 2,  # 2 = 1-minute bars
        "unitNumber": 1,
        "limit": 960,  # adjust if needed
        "includePartialBar": False
    }

    response = requests.post("https://api.topstepx.com/api/History/retrieveBars", headers=HEADERS, json=payload)

    if response.status_code == 200:
        bars = response.json()
        filename = f"{ASSET_ID.split('.')[-1]}_{date_str}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(bars, f, indent=2)
        print(f"✅ Saved: {filepath}")
    else:
        print(f"❌ Failed for {date_str}: {response.status_code} {response.text}")

def run_batch_fetch(start_date, end_date):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end     = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        fetch_and_save_day(date_str)
        current += timedelta(days=1)

# === RUN THIS ===
if __name__ == "__main__":
    run_batch_fetch("2025-06-22", "2025-06-27")
