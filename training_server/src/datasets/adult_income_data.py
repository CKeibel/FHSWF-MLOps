import pandas as pd
from datasets.base import BaseDataset
from sklearn.model_selection import train_test_split


class AdultIncomeData(BaseDataset):
    def __init__(self, config: dict):
        self.target: str = config["columns"]["target"]
        self.features: list[str] = config["columns"]["features"]
        df = pd.read_csv(
            config["path"],
            names=self.features + [self.target],
            na_values=" ?",
            header=0,
        )
        df = df.dropna()
        df[self.target] = df[self.target].apply(lambda x: 1 if x == ">50K" else 0)
        X = df.drop(self.target, axis=1)
        self.detect_featrue_type(X)
        y = df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

    def detect_featrue_type(self, df: pd.DataFrame) -> dict:
        self.numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
        self.categorical_features = df.select_dtypes(include=["object"]).columns

    def get_categorical_features(self) -> list[str]:
        return self.categorical_features

    def get_nummerical_features(self) -> list[str]:
        return self.numerical_features
