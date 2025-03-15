# -*- coding: utf-8 -*-
"""Entrypoint module for the API."""
from contextlib import asynccontextmanager

from api import router
from fastapi import FastAPI
from training import Trainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run training on startup
    trainer = Trainer()
    trainer.fit_and_log()
    del trainer
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
