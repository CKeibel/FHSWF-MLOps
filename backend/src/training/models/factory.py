from training.models.base import BaseModel
from training.models.randomforest import RandomForestClassifier
from training.models.xgboost import XGBClassifier

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
