import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path so we can import main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import preprocess_data, split_and_scale, train_logistic_regression, train_random_forest, train_balanced_rf

st.title("Vehicle Insurance Fraud Detection")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview", df.head())

    df = preprocess_data(df)
    xtrain, xtest, ytrain, ytest = split_and_scale(df)

    st.subheader("Model Performance")

    for name, trainer in [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),
        ("Balanced Random Forest", train_balanced_rf)
    ]:
        model, ypred = trainer(xtrain, xtest, ytrain, ytest)
        st.markdown(f"**{name} Results:**")
        st.text("Accuracy: {:.2f}".format((ypred == ytest).mean()))
