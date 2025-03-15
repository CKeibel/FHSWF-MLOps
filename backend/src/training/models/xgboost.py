import mlflow
import optuna
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from training.models.base import BaseModel
from xgboost import XGBClassifier as XGB


class XGBClassifier(BaseModel):
    def __init__(self, preprocessor: ColumnTransformer) -> None:
        self.preprocessor = preprocessor
        self.model = None
        self.study = None
        self.name = "XGBClassifier"

    def optimize(
        self, X_train, y_train, X_test, y_test, n_trials=10
    ) -> "XGBClassifier":
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
                "eval_metric": "logloss",
            }
            model = Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    (
                        "classifier",
                        XGB(**params, random_state=42, use_label_encoder=False),
                    ),
                ]
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return f1_score(y_test, y_pred)

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=n_trials)

        # Bestes Modell erstellen und trainieren
        self.model = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                (
                    "classifier",
                    XGB(
                        **self.study.best_params,
                        random_state=42,
                        use_label_encoder=False,
                    ),
                ),
            ]
        )
        self.model.fit(X_train, y_train)
        return self

    def log(self, X_test, y_test) -> "XGBClassifier":
        if self.model is None or self.study is None:
            raise ValueError("Model needs to be optimized first using optimize()")

        y_pred = self.model.predict(X_test)

        # MLflow logging
        mlflow.log_params(self.study.best_params)
        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
            }
        )

        # Model logging with registration
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            self.model,
            "model",
            signature=signature,
            input_example=X_test[:5].to_dict(orient="records"),
            registered_model_name=self.name,
        )

        # Register best version
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_version = mlflow.register_model(model_uri, self.name)
        client = MlflowClient()
        client.set_registered_model_alias(self.name, "newest", model_version.version)

        return self
