from metaflow import FlowSpec, step
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        print("Loading test data...")
        df = pd.read_csv("./data/TSLA.csv")
        df = df.sort_values("date")
        self.X_test = df[["open", "high", "low", "volume"]].tail(30)
        self.y_true = df["adj_close"].tail(30)
        self.next(self.load_model)

    @step
    def load_model(self):
        print("Loading model from MLflow Registry...")
        model_uri = "models:/stock_model/Production"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        print("Running predictions...")
        self.y_pred = self.model.predict(self.X_test)
        self.r2 = r2_score(self.y_true, self.y_pred)
        self.rmse = mean_squared_error(self.y_true, self.y_pred, squared=False)
        print(f"R² score: {self.r2:.4f}")
        print(f"RMSE: {self.rmse:.4f}")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete.")

if __name__ == "__main__":
    ScoringFlow()
