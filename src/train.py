import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    
    # Start MLflow Experiment
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        # 1. Load Processed Data
        df = pd.read_csv(config['data']['processed_path'])
        X = df.drop(columns=[config['train']['target_column']])
        y = df[config['train']['target_column']]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state']
        )
        
        # 2. Get Hyperparameters from Config
        params = {
            "n_estimators": config['train']['n_estimators'],
            "max_depth": config['train']['max_depth'],
            "random_state": config['data']['random_state']
        }
        
        # 3. Train Model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 4. Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
        
        # 5. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run complete. Metrics: {metrics}")

if __name__ == "__main__":
    train_model()