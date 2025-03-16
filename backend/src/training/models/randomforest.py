import optuna
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from training.models.base import BaseModel


class RandomForestClassifier(BaseModel):
    def __init__(self, preprocessor: ColumnTransformer) -> None:
        self.preprocessor = preprocessor
        self.model = None
        self.study = None
        self.name = "RandomForestClassifier"

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
                    ("classifier", RFC(**params, random_state=42)),
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
                    ),
                ),
            ]
        )
        self.model.fit(X_train, y_train)
        return self
