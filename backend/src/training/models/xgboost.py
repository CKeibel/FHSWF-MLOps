import optuna
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from src.training.models.base import BaseModel
from xgboost import XGBClassifier as XGB
from sklearn.preprocessing import FunctionTransformer


class XGBClassifier(BaseModel):
    def __init__(self, preprocessor: ColumnTransformer, customprocessor: FunctionTransformer) -> None:
        self.preprocessor = preprocessor
        self.customprocessor = customprocessor
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
                    ("custom_preprocessor", self.customprocessor),
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
