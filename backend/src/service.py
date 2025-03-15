import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Service:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Service, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not self.initialized:
            # Set MLflow tracking URI
            mlflow_dir = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
            mlflow.set_tracking_uri(f"file:{mlflow_dir}")
            logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

            self.client = MlflowClient()
            self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "AdultIncome")
            self.model = None
            self.model_version = None

            # Try to load model on initialization
            try:
                self._load_latest_model()
                logger.info(f"Loaded model version {self.model_version} successfully")
            except Exception as e:
                logger.warning(f"No model loaded at startup: {str(e)}")

            self.initialized = True

    def _load_latest_model(self) -> None:
        """Load the latest available model"""
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            logger.warning(f"Experiment '{self.experiment_name}' not found")
            return

        # Search for latest successful run
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )

        if not runs:
            logger.warning(
                f"No successful runs found for experiment {self.experiment_name}"
            )
            return

        run = runs[0]
        run_id = run.info.run_id

        # Load model from the run
        try:
            self.model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            self.model_version = run_id
            logger.info(f"Loaded model from run {run_id}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def set_model_by_tag(self, tag: str) -> Dict[str, Any]:
        """
        Load and set the current model by tag.

        Args:
            tag: Tag to identify the model ('production', 'staging', 'latest', or run_id)

        Returns:
            Dictionary with model metadata
        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")

        if tag == "latest":
            # Get the latest run
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=1,
            )
            if not runs:
                raise ValueError("No runs found in experiment")
            run_id = runs[0].info.run_id

        elif tag in ["production", "staging", "archived"]:
            # Find runs with the given tag
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.stage = '{tag}'",
                order_by=["attribute.start_time DESC"],
            )
            if not runs:
                raise ValueError(f"No runs found with stage '{tag}'")
            run_id = runs[0].info.run_id
        else:
            # Assume the tag is a run_id
            run_id = tag

        # Load the model
        self.model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        self.model_version = run_id

        # Get model metadata
        run = self.client.get_run(run_id)
        model_info = {
            "run_id": run_id,
            "model_name": run.data.tags.get("mlflow.runName", "unknown"),
            "stage": run.data.tags.get("stage", "none"),
            "metrics": {k: v for k, v in run.data.metrics.items()},
            "parameters": {k: v for k, v in run.data.params.items()},
            "timestamp": run.info.start_time,
        }

        return model_info

    def predict(self, features: pd.DataFrame) -> List[float]:
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if self.model is None or self.model_version is None:
            raise ValueError("No model loaded")

        run = self.client.get_run(self.model_version)

        return {
            "model_version": self.model_version,
            "last_updated": datetime.fromtimestamp(run.info.start_time / 1000),
            "features": (
                list(self.model.feature_names_in_)
                if hasattr(self.model, "feature_names_in_")
                else []
            ),
            "metrics": {k: v for k, v in run.data.metrics.items()},
        }
