import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "random_forest"))
sys.path.append(os.path.join(os.path.dirname(__file__), "xgboost"))

import mlflow
import mlflow.sklearn
import optuna

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pipelinetraining import RandomForestIncomeModelTraining
from pipelinetraining import XGBoost

def main():
    load_dotenv(override=True)

    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())

    root_dir = script_dir.parent.parent.parent

    dataPath = os.getenv("DATA_PATH")
    mlflowUri = os.getenv("MLFLOW_TRACKING_URI")

    dataPath = Path(dataPath)
    mlflowUri = Path(mlflowUri)

    if not dataPath.is_absolute():
        dataPath = root_dir / dataPath

    if not mlflowUri.is_absolute():
        mlflowUri = root_dir / mlflowUri

    print(dataPath)
    print(mlflowUri)

    RandomForestIncomeModelTraining(mlflowUri = mlflowUri, dataPath = dataPath)

if __name__ == "__main__":
    main()