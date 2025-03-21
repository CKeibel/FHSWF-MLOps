import json

import requests
import streamlit as st
from form import get_education_work_fields, get_financial_fields, get_personal_fields
from loguru import logger
from src.config import settings

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    h1 {
        color: #F63366;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("💰 Adult Income Prediction")
st.markdown(
    """
    Predict whether an adult earns more than $50,000 per year.
    Data from [Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset/data).
"""
)

# Fronend form
with st.form(key="MLOps Project"):
    personal, education_work = st.columns(2)

    with personal:
        age, gender, marital_status, relationship, race, native_country = (
            get_personal_fields()
        )

    with education_work:
        workclass, education, occupation, hours_per_week = get_education_work_fields()

    captial_gain, capital_loss, fnlwgt = get_financial_fields()

    submit_button = st.form_submit_button(label="Predict Income")


def send_request(data: dict) -> int:
    payload = {"data": data}
    logger.info(f"Sending request to backend: {payload} to {settings.backend_url}")
    res = requests.post(settings.backend_url, data=json.dumps(payload))
    if res.status_code == 200:
        return res.json()
    return -1


# Submit
if submit_button:
    education_mapping = {
        "Preschool": 1,
        "1st-4th": 2,
        "5th-6th": 3,
        "7th-8th": 4,
        "9th": 5,
        "10th": 6,
        "11th": 7,
        "12th": 8,
        "HS-grad": 9,
        "Some-college": 10,
        "Assoc-voc": 11,
        "Assoc-acdm": 12,
        "Bachelors": 13,
        "Masters": 14,
        "Prof-school": 15,
        "Doctorate": 16,
    }

    data = {
        "age": age,
        "gender": gender,
        "marital-status": marital_status,
        "relationship": relationship,
        "race": race,
        "native-country": native_country,
        "workclass": workclass,
        "education": education,
        "educational-num": education_mapping.get(education, 0),
        "occupation": occupation,
        "hours-per-week": hours_per_week,
        "capital-gain": captial_gain,
        "capital-loss": capital_loss,
        "fnlwgt": fnlwgt,
    }

    response = send_request(data)

    if response != -1:
        mapping = {0: "Less than $50,000", 1: "More than $50,000"}
        pred = int(response["prediction"])

        st.markdown(f"Prediction: {mapping[pred]}")
    else:
        st.error("Error: Could not get prediction.")
