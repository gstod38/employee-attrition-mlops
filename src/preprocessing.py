import pandas as pd
import yaml
import os

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_data(df, config):
    """
    Cleans the IBM HR dataset based on project requirements.
    """
    # 1. Target Encoding: 'Yes' -> 1, 'No' -> 0
    target = config['train']['target_column']
    if target in df.columns:
        df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 2. Feature Selection: Drop columns with zero variance (useless for ML)
    # These columns are the same for every single row in this dataset
    useless_cols = ['EmployeeCount', 'Over18', 'StandardHours']
    df = df.drop(columns=[col for col in useless_cols if col in df.columns])
    
    # 3. Handle Categorical Variables (One-Hot Encoding)
    # We'll use pandas get_dummies for simplicity
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Load raw data tracked by DVC
    raw_data_path = config['data']['raw_path']
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data not found at {raw_data_path}. Did you run 'dvc pull'?")
    else:
        df = pd.read_csv(raw_data_path)
        
        # Preprocess
        processed_df = preprocess_data(df, config)
        
        # Save output for the training script to use later
        output_path = config['data']['processed_path']
        processed_df.to_csv(output_path, index=False)
        print(f"Success! Processed data saved to {output_path}")