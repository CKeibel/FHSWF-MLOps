# -*- coding: utf-8 -*-
"""API."""
from fastapi import APIRouter
from service import Service

router = APIRouter()
service = Service()


@router.get("/health", response_model=dict)
def read_root():
    return {"status": "ok"}


@router.get("/predict", response_model=dict)
def predict():
    return service.predict()
