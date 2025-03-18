import os


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

class RandomForestPipeline:
    def __init__(self, data, **kwargs):
        
        self.f1score = 0

        catFeatures = data.getCategoricalFeatures()
        numFeatures = data.getNummericalFeatures()

        self.pipeline = Pipeline([
            ('custom_preprocessing', FunctionTransformer(self.custom_preprocessing)),
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(), numFeatures),
                    ('cat', Pipeline([
                        ('encoder', OneHotEncoder(handle_unknown='ignore'))
                    ]), catFeatures)
                ]
            )),
            ('classifier', RandomForestClassifier(
                **kwargs,
                random_state=42
            ))
        ])

    def custom_preprocessing(self, dfX):
        #dfX = dfX.copy()  # Um Änderungen am Original-Dataset zu vermeiden
        dfX["native-country"] = dfX["native-country"].apply(lambda x: x if x == "United-States" else "Other Countries")
        dfX["race"] = dfX["race"].apply(lambda x: x if x == "White" else "Other")
        #dfX.drop(columns=["education"], inplace=True)
        
        return dfX 
        
    def fit(self, X, y):
        """Trainiert die Pipeline."""
        self.pipeline.fit(X, y)
    
    def setf1score(self, y_test, y_pred):
        """Bewertet das Modell auf Testdaten."""
        self.f1score = f1_score(y_test, y_pred)
        return self.f1score
    
    def getDescTag(self):
        return {"items" : "RandomForstAdultIncome"}
    
    def getDescription(self):
        return "A Random forest Model for the prediction of the Adult Income Dataset"
    
    def getExperimentName(self):
        return 'RandomForestAdultIncome'

def main():
    load_dotenv(override=True)

    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())

    root_dir = script_dir.parent.parent.parent.parent

    dataPath = os.getenv("DATA_PATH")

    dataPath = Path(dataPath)

    if not dataPath.is_absolute():
        dataPath = root_dir / dataPath

    import sys
    temppath = os.path.abspath(root_dir / Path('backend/models/income'))
    sys.path.append(os.path.abspath(root_dir / Path('backend/models/income')))

    from data_preparation import AdultIncomeDataPreparation

    data = AdultIncomeDataPreparation(dataPath)
    
    model = RandomForestPipeline(data = data)
    
    # Pipeline trainieren
    model.fit(data.X_train, data.y_train)
    
    y_pred = model.pipeline.predict(data.X_test)

    accuracy = model.setf1score(data.y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()