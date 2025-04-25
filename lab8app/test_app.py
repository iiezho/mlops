import requests

# URL where FastAPI app is running
url = "http://127.0.0.1:8000/predict"

# Example payload based on your model features
payload = {
    "price_change": 3.5,
    "diff": 5.3,
    "adj_close": 148.5,
    "close_14_avg": 147.2
}

# Send POST request to FastAPI endpoint
response = requests.post(url, json=payload)

# Print the returned prediction
print("Prediction:", response.json())

