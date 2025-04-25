import pandas as pd
from fastapi import FastAPI
import mlflow
from pydantic import BaseModel

app = FastAPI(
    title="Stock Predictor",
    description="Predict stock price in specific dates",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for predicting stock price'}

class StockInput(BaseModel):
    price_change: float
    diff: float
    adj_close: float
    close_14_avg: float

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    mlflow.set_tracking_uri("sqlite:////Users/zoe/Desktop/SpringII/mlops_lab/notebooks/mlflow.db")
    logged_model_uri = 'runs:/ba858a86dd5440b28e80eaacdb886296/logistic_regression_model'
    model_pipeline = mlflow.sklearn.load_model(logged_model_uri)

# Defining path operation for /predict endpoint
@app.post('/predict')
async def predict(data : StockInput):
    columns = ["price_change", "diff", "adj_close", "close_14_avg"]
    X = pd.DataFrame([data.dict()])[columns]
    predictions = model_pipeline.predict(X)
    return {'Predictions': float(predictions)}