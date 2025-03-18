import streamlit as st
from form import get_education_work_fields, get_financial_fields, get_personal_fields

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


# Main form
with st.form(key="MLOps Project"):
    personal, education_work = st.columns(2)

    with personal:
        age, gender, marital_status, relationship, race, native_country = (
            get_personal_fields()
        )

    with education_work:
        workclass, education, education_num, occupation, hours_per_week = (
            get_education_work_fields()
        )

    captial_gain, capital_loss, fnlwgt = get_financial_fields()

    submit_button = st.form_submit_button(label="Predict Income")
