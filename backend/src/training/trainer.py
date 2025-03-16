import mlflow
from config import settings
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
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
        """Load models from config"""
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

    def fit_and_log(self, n_trails: int = None) -> None:
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        for model in self.models:
            with mlflow.start_run(run_name=model.name, nested=True):
                mlflow.log_table(
                    data=self.data.origin_data, artifact_file="training_data.json"
                )
                model.optimize(
                    X_train=self.data.X_train,
                    y_train=self.data.y_train,
                    X_test=self.data.X_test,
                    y_test=self.data.y_test,
                    n_trials=(
                        self.config["optimization"]["n_trials"]
                        if n_trails is None
                        else n_trails
                    ),
                )
                y_pred = model.model.predict(self.data.X_test)

                # MLflow logging
                mlflow.log_params(model.study.best_params)
                mlflow.log_metrics(
                    {
                        "accuracy": accuracy_score(self.data.y_test, y_pred),
                        "f1_score": f1_score(self.data.y_test, y_pred),
                    }
                )

                # Model logging with registration
                signature = infer_signature(self.data.X_test, y_pred)
                mlflow.sklearn.log_model(
                    model.model,
                    "model",
                    signature=signature,
                    input_example=self.data.X_test[:5].to_dict(orient="records"),
                    registered_model_name=model.name,
                )

                # Register best version
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                model_version = mlflow.register_model(model_uri, model.name)
                client = MlflowClient()
                client.set_registered_model_alias(
                    model.name, "newest", model_version.version
                )
