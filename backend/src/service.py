"""Module that orchestrates inference calls."""

import mlflow
from dynaconf import settings


class Service:
    def __init__(self) -> None:
        self.mlflow_uri = settings.MLFLOW_URI
        self.data_path = settings.DATA_PATH
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self) -> dict:
        """Orchestrates inference calls."""
        return {"message": "Hello World"}
