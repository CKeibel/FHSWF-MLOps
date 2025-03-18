import streamlit as st


def get_financial_fields():
    st.markdown("### Financial Metrics")
    captial_gain = st.number_input(
        label="Capital Gain", min_value=0, value=0, help="Annual capital profits"
    )
    capital_loss = st.number_input(
        label="Capital Loss", min_value=0, value=0, help="Annual capital losses"
    )
    fnlwgt = st.number_input(
        label="Final Weight",
        min_value=0,
        value=0,
        help="Census weighting factor representing demographic characteristics",
    )
    return captial_gain, capital_loss, fnlwgt
