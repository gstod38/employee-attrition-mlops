import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import sys

from src.preprocessing import preprocess_data

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    
    # Start MLflow Experiment
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        # DATA FALLBACK LOGIC
        path = config['data']['processed_path']
        
        is_ci_mode = False
        if not os.path.exists(path):
            path = "data/ci_sample.csv"
            is_ci_mode = True
            print(f"Running in CI mode with sample data: {path}")

        # 1. Load Data
        df = pd.read_csv(path)
        # 2. Preprocess Data
        df = preprocess_data(df)
        
        X = df.drop(columns=[config['train']['target_column']])
        y = df[config['train']['target_column']]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state']
        )
        
        # 2. Train Model
        params = {
            "n_estimators": config['train']['n_estimators'],
            "max_depth": config['train']['max_depth'],
            "random_state": config['data']['random_state']
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 3. Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics = {
            "accuracy": acc,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0)
        }
        
        # 4. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        # Exit with error if performance is too low
        threshold = 0.4 if is_ci_mode else 0.7
        
        if acc < threshold:
            print(f"Performance too low: {acc}. Requirement not met.")
            sys.exit(1)

        print(f"Run complete. Metrics: {metrics}")

if __name__ == "__main__":
    train_model()