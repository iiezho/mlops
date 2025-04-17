from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

class TrainingFlow(FlowSpec):

    test_size = Parameter("test_size", default=0.2, type=float, help="Fraction of data used for testing")
    seed = Parameter("seed", default=42, type=int, help="Random seed for reproducibility")

    @step
    def start(self):
        print("Loading data...")
        
        # Load and prepare the dataset
        df = pd.read_csv("./data/Amazon_stock_data_2000_2025.csv")
        df = df.sort_values("date")
        
        # Select features and target
        self.X = df[["open", "high", "low", "volume"]]
        self.y = df["adj_close"]
        
        self.next(self.split)

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
        print("Training model with hyperparameter tuning...")
        
        # Train the model using GridSearchCV for Ridge regression
        model = Ridge()
        param_grid = {
            "alpha": [0.1, 1.0, 10.0],
            "fit_intercept": [True, False]
        }

        # Grid search with 5-fold cross-validation
        grid = GridSearchCV(model, param_grid, scoring="r2", cv=5)
        grid.fit(self.X_train, self.y_train)

        # Save the best model and evaluation results
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.r2_score = grid.score(self.X_test, self.y_test)
        self.rmse = mean_squared_error(self.y_test, self.best_model.predict(self.X_test), squared=False)

        # Print results
        print("Best Params:", self.best_params)
        print("Test R²:", self.r2_score)
        print("Test RMSE:", self.rmse)

        self.next(self.register)

    @step
    def register(self):
        print("Logging model to MLflow...")
        # Log model to MLflow
        with mlflow.start_run():
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("seed", self.seed)
            mlflow.log_params(self.best_params)
            mlflow.log_metric("r2_score", self.r2_score)
            mlflow.log_metric("rmse", self.rmse)

            # Register the best model
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model",
                registered_model_name="stock_model"
            )
        self.next(self.end)

    @step
    def end(self):
        print("Training & registration complete!")

if __name__ == "__main__":
    TrainingFlow()
