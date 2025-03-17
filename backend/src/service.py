import logging
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Service:
    _instance = None

    def __new__(cls):
        """Singleton instance"""
        if cls._instance is None:
            cls._instance = super(Service, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not self.initialized:
            # Set MLflow tracking URI
            self.mlflow_dir = settings["mlflow"]["tracking_uri"]
            mlflow.set_tracking_uri(self.mlflow_dir)
            logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

            self.client = MlflowClient()
            self.experiment_name = settings["mlflow"]["experiment_name"]
            self.model = None
            self.model_version = None
            self.model_metadata = None

            # Try to load model on initialization
            try:
                self.load_model_by_alias("best")
                logger.info(f"Loaded model version {self.model_version} successfully")
            except Exception as e:
                logger.warning(f"No model loaded at startup: {str(e)}")

            self.initialized = True

    def load_model_by_alias(self, alias: str = "newest") -> dict:
        """Load a model by alias from the model registry."""
        try:
            # First, find models that have our alias
            for model_name in settings["models"]:
                try:
                    # Try to get model version by alias
                    model_version = self.client.get_model_version_by_alias(
                        model_name, alias
                    )

                    # If we found it, load this model
                    run_id = model_version.run_id
                    self.model_version = run_id
                    model_uri = f"models:/{model_name}@{alias}"
                    self.model = mlflow.pyfunc.load_model(model_uri)

                    # Store model metadata for later use
                    run = self.client.get_run(run_id)
                    self.model_metadata = {
                        "model_version": self.model_version,
                        "alias": alias,
                        "run_id": run_id,
                        "model_name": model_name,
                        "model_type": run.data.tags.get("mlflow.runName", model_name),
                        "metrics": {k: v for k, v in run.data.metrics.items()},
                    }

                    return self.model_metadata

                except mlflow.exceptions.MlflowException:
                    continue

            raise ValueError(f"No model found with alias '{alias}'")

        except Exception as e:
            logger.error(f"Error loading model by alias '{alias}': {str(e)}")
            raise

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently loaded model"""
        if self.model is None or self.model_metadata is None:
            raise ValueError("No model loaded")

        return self.model_metadata

    def predict(self, features: pd.DataFrame) -> list[float]:
        """Make predictions using the currently loaded model"""
        if self.model is None:
            raise ValueError("No model loaded. Call set_model_by_tag first.")

        # Ensure features are in the expected format
        if hasattr(self.model, "feature_names_in_"):
            features = features[list(self.model.feature_names_in_)]

        # Get predictions (handle both binary classification and regression)
        try:
            # Try predict_proba first for classification
            predictions = self.model.predict_proba(features)[:, 1].tolist()
        except (AttributeError, IndexError):
            # Fall back to predict for regression
            predictions = self.model.predict(features).tolist()

        return predictions
