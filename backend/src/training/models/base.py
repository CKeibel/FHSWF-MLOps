from abc import ABC, abstractmethod

from sklearn.compose import ColumnTransformer


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, preprocessor: ColumnTransformer) -> None:
        self.preprocessor = preprocessor
        self.model = None
        self.study = None

    @abstractmethod
    def optimize(self, X_train, y_train, X_test, y_test, n_trials=10) -> "BaseModel":
        pass
