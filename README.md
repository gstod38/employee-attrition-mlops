# Employee Attrition MLOps Pipeline

## Project Overview
This project implements a complete MLOps pipeline for predicting employee attrition using the IBM HR Analytics dataset. It demonstrates version control with Git and DVC, experiment tracking with MLflow, and automated testing with pytest.

## Setup
1. Clone this repository.
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Pull data: `dvc pull`

## Repository Structure
- `data/`: Dataset tracked by DVC.
- `src/`: Source code for preprocessing and training.
- `models/`: Trained model artifacts (ignored by Git).
- `tests/`: Automated test suite.