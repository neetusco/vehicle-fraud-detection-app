"""
Main script for Vehicle Insurance Claim Fraud Detection.
Includes EDA, preprocessing, training, and evaluation using Logistic Regression,
Random Forest, and Balanced Random Forest.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def load_and_explore_data(csv_file):
    df = pd.read_csv(csv_file)
    logging.info(f"Data loaded. Shape: {df.shape}")
    return df

def preprocess_data(df):
    id_col = 'PolicyNumber'
    target_col = 'FraudFound_P'
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [col for col in cat_cols if col not in [target_col, id_col]]
    num_impute_cols = ['Age']

    # Fill missing values
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in num_impute_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Drop ID
    if id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)

    # Encode categorical features
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Drop rows with any remaining missing values
    df.dropna(inplace=True)

    return df

def split_and_scale(df):
    target_col = 'FraudFound_P'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    return xtrain_scaled, xtest_scaled, ytrain, ytest

def train_logistic_regression(xtrain, xtest, ytrain, ytest):
    model = LogisticRegression(max_iter=1000)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    return model, ypred

def train_random_forest(xtrain, xtest, ytrain, ytest):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    return model, ypred

def train_balanced_rf(xtrain, xtest, ytrain, ytest):
    model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    return model, ypred

def evaluate_model(ytest, ypred, model_name):
    print(f"\n--- {model_name} ---")
    print("Accuracy:", accuracy_score(ytest, ypred))
    print("Classification Report:\n", classification_report(ytest, ypred))
    print("Confusion Matrix:\n", confusion_matrix(ytest, ypred))

def main():
    df = load_and_explore_data('data/fraud_oracle.csv')
    df = preprocess_data(df)
    xtrain, xtest, ytrain, ytest = split_and_scale(df)

    for name, trainer in [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),
        ("Balanced Random Forest", train_balanced_rf)
    ]:
        model, ypred = trainer(xtrain, xtest, ytrain, ytest)
        evaluate_model(ytest, ypred, name)

if __name__ == "__main__":
    main()
