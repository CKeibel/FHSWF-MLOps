
import os
from fastapi import FastAPI, Query, HTTPException
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

app = FastAPI()

class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model_by_alias = True)
def predict(
    model: str = Query(..., title="Model Name", description="Name of the MLflow model to load"),
    alias: str = Query(..., title="Model Alias", description="Alias for the model version"),
    data: PredictionInput = None
):
    try:
        load_dotenv(override=True)

        script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())

        print(f"Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

        root_dir = script_dir.parent.parent

        print(root_dir)

        data_path = os.getenv("DATA_PATH")
        mlflowUri = os.getenv("MLFLOW_TRACKING_URI")
        print('Origin from env: ' + data_path)
        print('Origin from env: ' + mlflowUri)

        data_path = Path(data_path)
        mlflowUri = Path(mlflowUri)

        print(data_path)
        print(mlflowUri)

        if not data_path.is_absolute():
            data_path = root_dir / data_path

        if not mlflowUri.is_absolute():
            mlflowUri = root_dir / mlflowUri

        print(data_path)
        print(mlflowUri)    

        mlflow.set_tracking_uri(mlflowUri)
        client = mlflow.tracking.MlflowClient()
        model = client.get_model_version_by_alias(model,alias)
        #model_uri = "runs:/{}/{}/model.pkl".format(run_id, 'random_forest_model')

        #pymodel = mlflow.sklearn.load_model(model_uri)

        #model = client.get_model_version_by_alias('random_forest_pipeline', 'best')
        loaded_model = mlflow.sklearn.load_model(f"runs:/{model.run_id}/random_forest_model")
        
        # Eingabe als DataFrame formatieren
        df = pd.DataFrame([data.dict()])
        
        df = df.rename(columns={'education_num': 'education-num', 'capital_loss': 'capital-loss', 'hours_per_week': 'hours-per-week', 'marital_status': 'marital-status', 'native_country': 'native-country', 'capital_gain': 'capital-gain'})
                

        # Vorhersage durchführen
        prediction = loaded_model.predict(df)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))