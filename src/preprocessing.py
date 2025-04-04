import pandas as pd

# Load the data
data = pd.read_csv("data/Amazon_stock_data_2000_2025.csv")

# Convert 'date' to datetime and set as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Create binary target: 1 if next day's close price is higher than today
data['price_movement'] = (data['close'].shift(-1) > data['close']).astype(int)

# Feature Engineering
data['price_change'] =data['close'] - data['open']
data['diff'] = data['high'] - data['low']
data['close_7_avg'] = data['close'].rolling(window=7).mean()
data['close_14_avg'] = data['close'].rolling(window=14).mean()

# Drop rows with missing values due to shift and rolling windows
data.dropna(inplace=True)

# Save processed data
data.to_csv("data/processed_amazon_stock.csv")
