from abc import ABC, abstractmethod

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, preprocessor: ColumnTransformer, customprocessor: FunctionTransformer) -> None:
        self.preprocessor = preprocessor
        self.model = None
        self.study = None

    @abstractmethod
    def optimize(self, X_train, y_train, X_test, y_test, n_trials=10) -> "BaseModel":
        pass
