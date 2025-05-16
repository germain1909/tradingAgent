from flask import Flask, request, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)
os.makedirs("logs", exist_ok=True)

@app.route('/webhook', methods=['POST'])
def webhook():
    # Print raw headers
    print("Headers:", dict(request.headers))

    # Print raw data (bytes)
    raw_data = request.get_data()
    print("Raw data (bytes):", raw_data)

    # Print raw data as string (for readability)
    print("Raw data (str):", raw_data.decode('utf-8', errors='replace'))

    # Now try to parse JSON safely
    try:
        data = request.get_json(force=True)
    except Exception as e:
        print(f"❌ JSON parse error: {e}")
        return jsonify({"error": "Invalid JSON"}), 400

    if not data:
        print("❌ No JSON data found")
        return jsonify({"error": "No JSON data"}), 400

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    with open(f"logs/alert_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Received alert: {data}")

    return jsonify({"status": "received"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
