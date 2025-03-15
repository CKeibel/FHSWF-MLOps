from adult_income_data import AdultIncomeData
from base import BaseDataset

datasets = {"AdultIncomeData": AdultIncomeData}


class DatasetFactory:
    @staticmethod
    def get_dataset(name: str, config: dict) -> BaseDataset:
        return datasets[name](config)
