import mlflow
from config import settings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from training.dataset import AdultIncomeData
from training.models import BaseModel, ModelFactory


class Trainer:
    def __init__(self) -> None:
        self.config = settings
        self.data = AdultIncomeData(self.config["dataset"])
        self.preprocessor = self.__init_preprocessor()
        self.models: list[BaseModel] = self.__load_models_from_config(
            self.config["models"]
        )

    def __load_models_from_config(self, model_names: list[str]) -> list[BaseModel]:
        if self.preprocessor is None:
            raise ValueError("Preprocessor needs to be initialized first")
        models = [
            ModelFactory.create_model(model_name)(self.preprocessor)
            for model_name in model_names
        ]
        return models

    def __init_preprocessor(self) -> ColumnTransformer:
        if self.data is None:
            raise ValueError("Data needs to be loaded first")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(), self.data.get_nummerical_features()),
                (
                    "cat",
                    Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                    self.data.get_categorical_features(),
                ),
            ],
            remainder="drop",
        )
        return preprocessor

    def fit_and_log(self) -> None:
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        for model in self.models:
            with mlflow.start_run(run_name=model.name, nested=True):
                model.optimize(
                    self.data.X_train,
                    self.data.y_train,
                    self.data.X_test,
                    self.data.y_test,
                    self.config["optimization"]["n_trials"],
                )
                model.log(
                    self.data.X_test,
                    self.data.y_test,
                )
