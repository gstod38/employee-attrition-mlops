import sys
import pandas as pd
import numpy as np
import yaml
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.preprocessing import preprocess_data

def monitor_drift():
    # 1. Load Config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load Reference Data 
    path = config['data']['processed_path']
    if not os.path.exists(path):
        path = "data/ci_sample.csv"
    
    reference_df = pd.read_csv(path)
    
    # 3. Create Synthetic Production Data
    production_df = reference_df.copy()
    
    # SIMULATE DRIFT: Increase Age by 10 years and change DailyRate
    production_df['Age'] = production_df['Age'] + 10
    production_df['DailyRate'] = production_df['DailyRate'] * 1.5
    
    # 4. Run Evidently Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=production_df)
    
    # 5. Save Report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    report.save_html(report_path)
    
    # 6. Check Drift Status 
    result = report.as_dict()
    drift_share = result['metrics']['result']['drift_share']
    
    print(f"Drift Report Generated: {report_path}")
    print(f"Dataset Drift Share: {drift_share:.2f}")

    # Exit with code 1 if drift is too high (e.g., > 50%)
    if drift_share > 0.5:
        print("⚠️ High Data Drift detected!")
        sys.exit(1) 

if __name__ == "__main__":
    monitor_drift()