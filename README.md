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


```bash
curl -X 'POST' \
  'https://gateway-api-demo.s2f.projectx.com/api/Auth/loginKey' \
  -H 'accept: text/plain' \
  -H 'Content-Type: application/json' \
  -d '{
  "userName": "string",
  "apiKey": "string"
}'
```