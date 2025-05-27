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
python -m training.train_macd_agent.py

## Start the fetching of daily data 
python3 scripts/fetch_daily_data.py

Need to add in your topstep token specify dates you want

## Germain