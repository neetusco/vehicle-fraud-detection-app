import streamlit as st 
import pandas as pd
import numpy as np
import os, sys
from importlib import import_module

# Allow import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Insurance Fraud Checker", layout="centered")
st.title(" Vehicle Insurance Fraud Checker")

# --- MODEL OPTIONS ---
model_options = {
    "Logistic Regression": ("Modules.logistic_regression_model", "predict_logistic"),
    "Random Forest": ("Modules.random_forest_model", "predict_random_forest"),
    "Neural Network": ("Modules.neural_network_model", "predict_neural_network"),
    "XGBoost": ("Modules.XGBoost_classifier", "predict_xgboost")
}

# --- STEP 1: MODEL SELECTION ---
st.subheader("Step 1: Select Model")
selected_model = st.selectbox("Choose a model", list(model_options.keys()))

# --- STEP 2: FEATURE INPUTS ---
if selected_model:
    st.subheader("Step 2: Enter Claim Information")

    sex = st.selectbox("Sex", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    age = st.slider("Age", 18, 100)
    vehicle_price = st.selectbox("Vehicle Price", ["less than 20000", "20000 to 29000", "more than 69000"])
    deductible = st.number_input("Deductible", min_value=0, step=100)
    past_claims = st.slider("Past Number of Claims", 0, 10)
    fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Sedan - Liability", "Sedan - All Perils", "Sedan - Collision",
                                               "Sport - All Perils", "Utility - Collision", "Utility - All Perils"])
    vehicle_category = st.selectbox("Vehicle Category", ["Sedan", "Sports Car", "Utility"])
    police_report = st.selectbox("Police Report Filed", ["Yes", "No"])
    witness_present = st.selectbox("Witness Present", ["Yes", "No"])
    age_of_vehicle = st.slider("Age of Vehicle (years)", 0, 20)
    days_policy_accident = st.slider("Days Between Policy Start and Accident", 0, 1000)

    if st.button("Check for Fraud"):
        input_df = pd.DataFrame({
            "Sex": [sex],
            "MaritalStatus": [marital_status],
            "Age": [age],
            "VehiclePrice": [vehicle_price],
            "Deductible": [deductible],
            "PastNumberOfClaims": [past_claims],
            "Fault": [fault],
            "AccidentArea": [accident_area],
            "PolicyType": [policy_type],
            "VehicleCategory": [vehicle_category],
            "PoliceReportFiled": [police_report],
            "WitnessPresent": [witness_present],
            "AgeOfVehicle": [age_of_vehicle],
            "Days_Policy_Accident": [days_policy_accident]
        })

        try:
            # Dynamically import and call model
            module_name, func_name = model_options[selected_model]
            model_module = import_module(module_name)
            predict_method = getattr(model_module, func_name)

            prediction, model, processed_df, accuracy = predict_method(input_df)

            if prediction == 1:
                st.error("Fraudulent Claim Detected!")
            else:
                st.success("Claim appears genuine.")
                
            # Show model accuracy   
            if accuracy is not None:
                st.markdown(f"🧪 **Model Accuracy (on test data):** {accuracy:.2%}")

            # if model and hasattr(model, "predict_proba"):
            #      prob = model.predict_proba(processed_df)[0][1]
            #      st.info(f"Fraud Probability: **{prob:.2%}**")


        except Exception as e:
            st.error(f"Error running model: {e}")