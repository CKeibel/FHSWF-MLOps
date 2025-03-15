from models.base import BaseModel
from models.rf_classifier import RandomForestClassifier
from models.xgboost_classifier import XGBClassifier

models = {
    "XGBClassifier": XGBClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}


class ModelFactory:
    @staticmethod
    def create_model(
        model_name: str,
    ) -> BaseModel:
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
        return models[model_name]
