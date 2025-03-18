import streamlit as st


def get_education_work_fields():
    st.markdown("### Education & Work")
    workclass = st.selectbox(
        label="Workclass",
        options=(
            "Private",
            "Local-gov",
            "Self-emp-not-inc",
            "Federal-gov",
            "State-gov",
            "Self-emp-inc",
            "Without-pay",
            "Never-worked",
        ),
    )
    education = st.selectbox(
        label="Education",
        options=(
            "Preschool",
            "1st-4th",
            "5th-6th",
            "7th-8th",
            "9th",
            "10th",
            "11th",
            "12th",
            "HS-grad",
            "Assoc-acdm",
            "Some-college",
            "Prof-school",
            "Bachelors",
            "Masters",
            "Doctorate",
            "Assoc-voc",
        ),
    )

    education_num = st.number_input(
        label="Education Number",
        min_value=0,
        value=0,
        help="Number of years of education completed",
    )

    occupation = st.selectbox(
        label="Occupation",
        options=(
            "Machine-op-inspct",
            "Farming-fishing",
            "Protective-serv",
            "Other-service",
            "Prof-specialty",
            "Craft-repair",
            "Adm-clerical",
            "Exec-managerial",
            "Tech-support",
            "Sales",
            "Priv-house-serv",
            "Transport-moving",
            "Handlers-cleaners",
            "Armed-Forces",
        ),
    )
    hours_per_week = st.slider(
        label="Hours per Week",
        min_value=0,
        max_value=24 * 7,
        value=40,
        help="Average hours worked per week",
    )

    return workclass, education, education_num, occupation, hours_per_week
