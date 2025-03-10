import os


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

class RandomForestPipeline:
    def __init__(self, data, **kwargs):
        
        catFeatures = data.getCategoricalFeatures()
        numFeatures = data.getNummericalFeatures()

        self.pipeline = Pipeline([
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
        
    def fit(self, X, y):
        """Trainiert die Pipeline."""
        self.pipeline.fit(X, y)
    
    def f1score(self, y_test, y_pred):
        """Bewertet das Modell auf Testdaten."""
        return f1_score(y_test, y_pred)

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

    accuracy = model.f1score(data.y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()