import requests

class TopstepExecutor:
    BASE_URL = "https://api.topstepx.com/api"

    def __init__(self, account_id,contract_id, headers):
        # Set Gold contract (full size)
        self.account_id = account_id
        self.contract_id = contract_id
        self.headers = headers

    def place_order(self, side, size=1, order_type=2, limit_price=None, stop_price=None, trail_price=None):
        # side: 0 = BUY, 1 = SELL
        data = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "type": order_type,
            "side": side,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price,
            "customTag": None,
            "linkedOrderId": None
        }
        resp = requests.post(f"{self.BASE_URL}/Order/place", json=data, headers=self.headers)
        try:
            return resp.json()
        except Exception:
            return {"error": "Failed to parse response", "raw": resp.text}

    def modify_order(self, order_id, size=1, limit_price=None, stop_price=None, trail_price=None):
        data = {
            "accountId": self.account_id,
            "orderId": order_id,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price
        }
        resp = requests.post(f"{self.BASE_URL}/Order/modify", json=data, headers=self.headers)
        try:
            return resp.json()
        except Exception:
            return {"error": "Failed to parse response", "raw": resp.text}

    def close_position(self):
        data = {
            "accountId": self.account_id,
            "contractId": self.contract_id
        }
        resp = requests.post(f"{self.BASE_URL}/Position/closeContract", json=data, headers=self.headers)
        try:
            return resp.json()
        except Exception:
            return {"error": "Failed to parse response", "raw": resp.text}
