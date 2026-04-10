# Employee Attrition MLOps Pipeline

## Project Overview
This project implements a complete, end-to-end MLOps pipeline for predicting employee attrition using the IBM HR Analytics dataset. It covers the full lifecycle from data versioning to automated drift monitoring.

## 1. Repository Structure
- `data/`: Dataset tracked by DVC (not stored in Git).
- `configs/`: YAML files for hyperparameters and paths.
- `src/`: 
  - `preprocessing.py`: Data cleaning and feature engineering.
  - `train.py`: Model training with MLflow integration.
  - `monitor_drift.py`: Drift detection using Evidently.
- `tests/`: Pytest suite (Unit, Data, and Model tests).
- `reports/`: Generated drift monitoring reports.

## 2. Setup & Installation
1. Clone the repository: `git clone <your-repo-url>`
2. Create environment: `python -m venv .venv`
3. Activate: `source .venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Pull Data: `dvc pull` (Requires DVC configured)

## 3. How to Run
- **Train Model:** `PYTHONPATH=. python src/train.py`
- **Run Tests:** `pytest tests/ -v`
- **Compare Experiments:** `python compare_experiments.py`
- **Monitor Drift:** `PYTHONPATH=. python src/monitor_drift.py`

## 4. Experiment Tracking (MLflow)
We use MLflow to track hyperparameters (like `n_estimators` and `max_depth`) and metrics (Accuracy, F1-Score, Precision). 
* At least 5 experiments were run with varying configurations.
* Use `compare_experiments.py` to programmatically identify the best-performing model run.

## 5. CI/CD Pipeline
This project uses **GitHub Actions** to automate the workflow:
1. **Test Job:** Runs the full pytest suite.
2. **Train Job:** Re-trains the model and verifies performance thresholds (only runs if tests pass).

## 6. Drift Monitoring Analysis
For a detailed analysis of data drift detection and its impact on model performance, please see the [MONITORING.md]file.