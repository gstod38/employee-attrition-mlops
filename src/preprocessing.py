import pandas as pd
import yaml
import os

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_data(df, config):
    # 1: Create a deep copy so we don't mutate the original input
    data = df.copy()
    
    target = config['train']['target_column']
    
    # Encode target
    if target in data.columns:
        data[target] = data[target].map({'Yes': 1, 'No': 0})
    
    # 2: Missing value handling for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
        
    # Drop constant columns (requirement for your tests)
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    
    # One-hot encoding for categorical variables
    data = pd.get_dummies(data)
    
    return data

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