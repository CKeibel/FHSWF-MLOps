# -*- coding: utf-8 -*-
"""Entrypoint module for the API."""
from api import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
