import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# This allows importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import preprocess_data, split_and_scale, train_logistic_regression


st.title("Vehicle Insurance Fraud Checker")

# Simulate input form
st.subheader("Enter Claim Information")

sex = st.selectbox("Sex", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
age = st.slider("Age", 18, 100)
vehicle_price = st.selectbox("Vehicle Price", ["less than 20000", "20000 to 29000", "more than 69000"])
deductible = st.number_input("Deductible", min_value=0, step=100)
past_claims = st.slider("Past Number of Claims", 0, 5)
fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])

# Button to predict
if st.button("Check for Fraud"):
    # Build a single-row dataframe to simulate new input
    input_df = pd.DataFrame({
        "Sex": [sex],
        "MaritalStatus": [marital_status],
        "Age": [age],
        "VehiclePrice": [vehicle_price],
        "Deductible": [deductible],
        "PastNumberOfClaims": [past_claims],
        "Fault": [fault]
    })

    # Load your pre-trained model here
    # Preprocess `input_df` same as your training data (use get_dummies, etc.)
    # model = load_model('your_saved_model.pkl')  ← optional step if serialized

    # Fake prediction for now
    prediction = np.random.choice([0, 1], p=[0.7, 0.3])  # simulate fraud result

    if prediction == 1:
        st.error("Fraudulent Claim Detected!")
    else:
        st.success("Claim appears genuine.")
