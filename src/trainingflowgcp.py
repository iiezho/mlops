from metaflow import FlowSpec, step, Parameter, resources, retry, catch, timeout, pypi
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class TrainingFlow(FlowSpec):

    test_size = Parameter("test_size", default=0.2, type=float, help="Fraction of data used for testing")
    seed = Parameter("seed", default=42, type=int, help="Random seed for reproducibility")
    
    @retry(times=2)
    @pypi(packages={'pandas': '1.5.3', 'numpy': '1.23.5', 'scikit-learn': '1.2.2', 'mlflow': '2.11.1'}, python='3.9')
    @step
    def start(self):
        print("Loading data...")
        
        # Load and prepare the dataset
        #df = pd.read_csv("/Users/zoe/Desktop/SpringII/mlops_lab/data/Amazon_stock_data_2000_2025.csv")
        df = pd.read_csv("https://storage.googleapis.com/mlflow-artifacts-zoe-lab7/data/Amazon_stock_data_2000_2025.csv")
        df = df.sort_values("date")
        
        # Select features and target
        self.X = df[["open", "high", "low", "volume"]]
        self.y = df["adj_close"]
        
        self.next(self.split)
        
    @resources(cpu=1, memory=4096)
    
    @timeout(seconds=3600)
    @catch
    @step
    def split(self):
        print("Splitting data...")
        
        # Split data for training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed
        )
        self.next(self.train)
       
    @step
    def train(self):
        print("Preparing for model training...")

        print("Training set size:", self.X_train.shape)
        print("Testing set size:", self.X_test.shape)
        
        self.next(self.register)
        
    @step
    def register(self):
        uri = "https://mlflow-server-597557234431.us-west2.run.app"
        print(f"Mlflow address: {uri}")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("stock-analysis")

        print("Training Ridge models and logging to MLflow...")
        alphas = [0.1, 1.0, 10.0]
        intercepts = [True, False]
        self.results = []

        for alpha in alphas:
            for fit_intercept in intercepts:
                model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
                model.fit(self.X_train, self.y_train)

                r2 = model.score(self.X_test, self.y_test)
                rmse = mean_squared_error(self.y_test, model.predict(self.X_test), squared=False)

                run_name = f"Ridge_alpha={alpha}_intercept={fit_intercept}"
                print(f"Logging: {run_name}")

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("fit_intercept", fit_intercept)
                    mlflow.log_param("test_size", self.test_size)
                    mlflow.log_param("seed", self.seed)
                    mlflow.log_metric("r2_score", r2)
                    mlflow.log_metric("rmse", rmse)

                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model"
                    )

                self.results.append((r2, rmse, alpha, fit_intercept, model))

        # Select best model
        self.results.sort(key=lambda x: (-x[0], x[1]))
        best_model = self.results[0][4]

        print("Registering best model separately...")
        with mlflow.start_run(run_name="Best_Model_Registration"):
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name="stock_model"
            )

        self.next(self.end)

    @step
    def end(self):
        print("Training & registration complete!")

if __name__ == "__main__":
    TrainingFlow()
