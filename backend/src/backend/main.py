# -*- coding: utf-8 -*-
"""Entrypoint module for the API."""
from fastapi import FastAPI

from .api import router

app = FastAPI()
app.include_router(router)
