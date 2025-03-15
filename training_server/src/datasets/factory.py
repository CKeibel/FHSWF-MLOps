from datasets.adult_income_data import AdultIncomeData
from datasets.base import BaseDataset

datasets = {"AdultIncomeData": AdultIncomeData}


class DatasetFactory:
    @staticmethod
    def get_dataset(config: dict) -> BaseDataset:
        return datasets[config["name"]](config)
