import os

import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pipeline import RandomForestPipeline

class XGBoostIncomeModelTraining:
    def __init__(self, mlflowUri, dataPath, optunaRunNumber = 5):
        mlflow.set_tracking_uri(mlflowUri)

        mlflow.set_experiment("Adult Income")

        load_dotenv(override=True)

        script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())

        root_dir = script_dir.parent.parent.parent.parent

        import sys

        sys.path.append(os.path.abspath(root_dir / Path('backend/models/income')))
        
        from data_preparation import AdultIncomeDataPreparation

        data = AdultIncomeDataPreparation(dataPath)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': 42
            }

            pipeline = RandomForestPipeline(data = data,
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            class_weight='balanced')
            
            with mlflow.start_run(nested=True):
                pipeline.fit(data.X_train, data.y_train)
                y_pred = pipeline.pipeline.predict(data.X_test)
                f1 = pipeline.f1score(data.y_test, y_pred)

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)
                mlflow.log_param("min_samples_leaf", min_samples_leaf)
                mlflow.log_metric("f1_score", f1)

                return f1
        
        with mlflow.start_run(run_name='Random Forest', nested=True):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=optunaRunNumber)
        
        best_params = study.best_params

        print(best_params)


def main():
    load_dotenv(override=True)

    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())

    root_dir = script_dir.parent.parent.parent.parent

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
        
        

