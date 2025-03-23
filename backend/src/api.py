# -*- coding: utf-8 -*-
"""API."""
import io
import logging

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from src.schemas import (
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    SetModelRequest,
)
from src.service import Service
from src.training import Trainer

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
service = Service()  # Singleton instance


@router.get("/", response_model=dict)
def read_root():
    model_loaded = service.model is not None
    return {
        "status": "ok" if model_loaded else "no_model",
        "model_version": service.model_version if model_loaded else None,
    }


@router.get("/health", response_model=dict)
def health():
    model_loaded = service.model is not None
    return {
        "status": "ok" if model_loaded else "no_model",
        "model_version": service.model_version if model_loaded else None,
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if not service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame with all required features
        input_data = pd.DataFrame([request.data.dict(by_alias=True)])

        # Make predictions
        prediction = service.predict(input_data)

        return PredictionResponse(
            prediction=prediction, model_version=service.model_version
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/set_model", response_model=ModelInfo)
async def set_model(request: SetModelRequest) -> ModelInfo:
    try:
        model_info = service.load_model_by_alias(request.alias)
        return ModelInfo(**model_info)

    except Exception as e:
        logger.error(f"Error setting model: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Error setting model: {str(e)}")


@router.get("/model-info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    if not service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        model_info = service.get_model_info()
        return ModelInfo(**model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/upload_data", response_model=dict)
async def upload_data(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    trainer: Trainer = Depends(lambda: Trainer()),
):
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
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
