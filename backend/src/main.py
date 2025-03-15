# -*- coding: utf-8 -*-
"""Entrypoint module for the API."""
import logging
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

    # Run training on startup if no model is loaded
    if service.model is None:
        logger.info("No model found. Starting initial training...")
        trainer = Trainer()
        trainer.fit_and_log()

        # Try to load the newly trained model
        try:
            service.set_model_by_tag("latest")
        except Exception as e:
            logger.error(f"Failed to load model after training: {str(e)}")

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
