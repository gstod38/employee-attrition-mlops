import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import sys

# Import the preprocessing function
from src.preprocessing import preprocess_data

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        path = config['data']['processed_path']
        
        if not os.path.exists(path):
            path = "data/ci_sample.csv"
            print(f"CI Mode: Using {path}")

        df = pd.read_csv(path)
        
        
        df = preprocess_data(df, config)
        
        X = df.drop(columns=[config['train']['target_column']])
        y = df[config['train']['target_column']]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state']
        )
        
        params = {
            "n_estimators": config['train']['n_estimators'],
            "max_depth": config['train']['max_depth'],
            "random_state": config['data']['random_state']
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc})
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run complete. Accuracy: {acc}")

if __name__ == "__main__":
    train_model()