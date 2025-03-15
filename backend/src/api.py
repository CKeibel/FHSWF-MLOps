# -*- coding: utf-8 -*-
"""API."""
import io

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from training import Trainer

router = APIRouter()


@router.get("/health", response_model=dict)
def read_root():
    return {"status": "ok"}


@router.post("/predict", response_model=dict)
def predict():
    return {"status": "ok"}


@router.post("/set_model")
def set_model():
    return {"status": "ok"}


@router.post("/retrain")
async def trigger_retraining(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    trainer: Trainer = Depends(lambda: Trainer()),
):
    # Process the uploaded file
    try:
        # Read the uploaded CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), na_values=" ?")

        # Add the new data to the trainer's dataset
        trainer.data.append_data(df)

        # Schedule the training task in the background
        background_tasks.add_task(trainer.fit_and_log)

        return {"status": "Training scheduled with new data", "rows_added": len(df)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
