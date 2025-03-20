import streamlit as st


def get_personal_fields():
    st.markdown("### Personal Information")
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox(label="Gender", options=("Male", "Female"))
    marital_status = st.selectbox(
        label="Marital Status",
        options=(
            "Never-married",
            "Married-civ-spouse",
            "Widowed",
            "Divorced",
            "Separated",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ),
    )
    relationship = st.selectbox(
        label="Relationship",
        options=(
            "Own-child",
            "Husband",
            "Not-in-family",
            "Unmarried",
            "Wife",
            "Other-relative",
        ),
        help="Family role classification",
    )
    race = st.selectbox(
        label="Race",
        options=("Black", "White", "Asian-Pac-Islander", "Other", "Amer-Indian-Eskimo"),
    )

    native_country = st.selectbox(
        label="Native Country",
        options=(
            "United-States",
            "Peru",
            "Guatemala",
            "Mexico",
            "Dominican-Republic",
            "Ireland",
            "Germany",
            "Philippines",
            "Thailand",
            "Haiti",
            "El-Salvador",
            "Puerto-Rico",
            "Vietnam",
            "South",
            "Columbia",
            "Japan",
            "India",
            "Cambodia",
            "Poland",
            "Laos",
            "England",
            "Cuba",
            "Taiwan",
            "Italy",
            "Canada",
            "Portugal",
            "China",
            "Nicaragua",
            "Honduras",
            "Iran",
            "Scotland",
            "Jamaica",
            "Ecuador",
            "Yugoslavia",
            "Hungary",
            "Hong",
            "Greece",
            "Trinadad&Tobago",
            "Outlying-US(Guam-USVI-etc)",
            "France",
            "Holand-Netherlands",
        ),
    )

    return age, gender, marital_status, relationship, race, native_country
