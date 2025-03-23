# -*- coding: utf-8 -*-
"""Entrypoint module for the API."""
import sys
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from src.api import router
from src.config import settings
from src.service import Service
from src.training import Trainer

warnings.filterwarnings("ignore")


# Set up logging
logger.add(
    "logs/{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {name} - {message}",
    rotation="00:00",
    level=settings.logging.level,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the service (will try to load latest model)
    logger.info("Starting application...")
    service = Service()

    if service.model is None:
        logger.info("No model found. Starting initial training...")
        trainer = Trainer()
        trainer.fit_and_log(n_trails=2)
        try:
            service.load_model_by_alias(alias="newest")
            logger.info(f"Loaded model version {service.model_version} successfully")
        except Exception as e:
            logger.error(f"Failed to load model after training: {str(e)}")
            logger.error("Shutting down application...")
            sys.exit(1)
    yield

    # Cleanup
    logger.info("Shutting down application...")


app = FastAPI(
    title="Adult Income Prediction API",
    description="API for predicting income levels",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])

app.include_router(router)
