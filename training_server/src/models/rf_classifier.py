import mlflow
import optuna
from mlflow.models import infer_signature
from models.base import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


class RandomForestClassifier(BaseModel):
    def __init__(self, preprocessor: ColumnTransformer) -> None:
        self.preprocessor = preprocessor
        self.model = None
        self.study = None

    def optimize(
        self, X_train, y_train, X_test, y_test, n_trials=10
    ) -> "RandomForestClassifier":
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "class_weight": "balanced",
            }

            model = Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("classifier", RandomForestClassifier(**params, random_state=42)),
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
                    RFC(
                        **self.study.best_params,
                        random_state=42,
                        use_label_encoder=False
                    ),
                ),
            ]
        )
        self.model.fit(X_train, y_train)
        return self

    def log(self, X_test, y_test) -> "RandomForestClassifier":
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

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            self.model,
            "RandomForestClassifier",
            signature=signature,
            input_example=X_test[:5].to_dict(orient="records"),
        )
        return self
