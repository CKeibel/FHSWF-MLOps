# -*- coding: utf-8 -*-
"""Entrypoint module for the API."""
import logging
import sys
from contextlib import asynccontextmanager

from api import router
from fastapi import FastAPI
from service import Service
from training import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

app.include_router(router)
