import mlflow
import pandas as pd

def compare_runs():
    experiment_name = "Employee-Attrition-Analysis"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # 1. Get the DataFrame
    df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if df.empty:
        print("No runs found.")
        return

    # 2. Sort by accuracy
    df = df.sort_values("metrics.accuracy", ascending=False)

    # 3. Iterate through the top row to get the best run
    best_row = None
    for _, row in df.head(1).iterrows():
        best_row = row
        break

    if best_row is not None:
        print("=" * 40)
        print("🏆 BEST EXPERIMENT RUN")
        print("=" * 40)
        
        # Accessing by column name directly from the row object
        print(f"Run ID:   {best_row['run_id']}")
        print(f"Accuracy: {float(best_row['metrics.accuracy']):.4f}")
        print(f"Recall:   {float(best_row['metrics.recall']):.4f}")
        
        print("-" * 40)
        print("WINNING PARAMETERS:")
        # Safe check for params
        cols = df.columns
        if 'params.max_depth' in cols:
            print(f"Max Depth:    {best_row['params.max_depth']}")
        if 'params.n_estimators' in cols:
            print(f"N Estimators: {best_row['params.n_estimators']}")
        print("=" * 40)

if __name__ == "__main__":
    compare_runs()