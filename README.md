# Trading Agent Setup

## NGROK

```bash
ngrok config add-authtoken ************
ngrok http 5000
```

## Application 
```bash 
python3 app.py
```

## TopStepX Authentication 1 Every 24 Hours 
```bash
curl -X 'POST' \
  'https://api.topstepx.com/api/Auth/loginKey' \
  -H 'accept: text/plain' \
  -H 'Content-Type: application/json' \
  -d '{
  "userName": "string",
  "apiKey": "string"
}'
```
#Above will return the token then you have to validate the token

## Validate Token
```bash
curl -X 'POST' \
  'https://api.topstepx.com/api/Auth/validate' \
  -H 'accept: text/plain' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE' \
  -d ''
```
  #Above will return another token that we use below

## TopStepx Curl to get current contract
```bash
$ curl -X 'POST' \
  'https://api.topstepx.com/api/Contract/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE' \
  -d '{
    "live": false,
    "searchText": "GC"
}'
```
## TopStepx Retrieve Past Bars 
``` bash
curl -X POST 'https://api.topstepx.com/api/History/retrieveBars' \
-H 'accept: text/plain' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer YOUR_TOKEN_HERE' \
-d '{
    "contractId": "CON.F.US.GC.Z24",
    "live": false,
    "startTime": "2024-12-01T00:00:00Z",
    "endTime": "2024-12-31T21:00:00Z",
    "unit": 3,
    "unitNumber": 1,
    "limit": 7,
    "includePartialBar": false
  }'
```


## Need to install FinRL
```bash
git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL
pip3 install -r requirements.txt
pip3 install -e .
```



docker build -t my-trading-app .
docker run -p 5000:5000 my-trading-app



## Start the training
python3 -m training.train_macd_agent

## Start the fetching of daily data 
python3 -m scripts.fetch_daily_data

Need to add in your topstep token specify dates you want


## Notes regarding environment creation and using data frame in environment
1. You define your environment like this:
python
Copy
Edit
class FuturesTradingEnv(gym.Env):
    def __init__(self, data, normalized_data, initial_balance=100_000, asset_name="GC4"):
        self.data = data
        self.normalized_data = normalized_data
        ...
2. When you call:
python
Copy
Edit
env = self.env_class(**self.env_kwargs)
Python does this under the hood:

python
Copy
Edit
env = FuturesTradingEnv(data=df, normalized_data=df_norm)
So inside your environment, you can directly use:

python
Copy
Edit
self.data  # gives you the full unnormalized DataFrame
self.normalized_data  # gives you the normalized version
✅ How You Can Access df in the Env
Anywhere inside your environment class (like __init__, _get_obs, step, etc.), you can access:

python
Copy
Edit
self.data.iloc[...]  # for raw prices, timestamps, etc.
self.normalized_data.iloc[...]  # for features the model should learn from


## Germain