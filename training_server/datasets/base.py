from abc import ABC, abstractmethod


class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def get_categorical_features(self) -> list[str]:
        pass

    @abstractmethod
    def get_nummerical_features(self) -> list[str]:
        pass
