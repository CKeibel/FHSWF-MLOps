import pandas as pd
from sklearn.model_selection import train_test_split

class AdultIncomeDataPreparation:
    def __init__(self, datapath):
        columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

        self.originData = pd.read_csv(datapath / "adult.csv")

        df = pd.read_csv(datapath / "adult.csv", names=columns, na_values=" ?", header=0)
        
        df.dropna(inplace=True)
        df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

        self.X = df.drop("income", axis=1)
        self.y = df["income"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15, random_state=42, stratify=self.y)


    def getCategoricalFeatures(self):
        return ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    def getNummericalFeatures(self):
        return self.X.select_dtypes(include=['int64', 'float64']).columns

      