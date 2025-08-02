import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger("data_processing")

def preprocess_data(df, training=True):
    # Fix Days_Policy_Accident if string-based (e.g., 'more than 30' -> 31)
    if 'Days_Policy_Accident' in df.columns:
        df['Days_Policy_Accident'] = df['Days_Policy_Accident'].replace({'more than 30': 31})
        df['Days_Policy_Accident'] = pd.to_numeric(df['Days_Policy_Accident'], errors='coerce')

    # Fix PastNumberOfClaims if string-based
    if 'PastNumberOfClaims' in df.columns:
        df['PastNumberOfClaims'] = df['PastNumberOfClaims'].replace({
            'none': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            'more than 4': 5
        })
        df['PastNumberOfClaims'] = pd.to_numeric(df['PastNumberOfClaims'], errors='coerce')

    # Fix AgeOfVehicle if string-based (e.g., '3 years' -> 3)
    if 'AgeOfVehicle' in df.columns:
        df['AgeOfVehicle'] = df['AgeOfVehicle'].astype(str).str.extract(r'(\d+)').astype(float)

    selected_cols = [
        'Sex', 'MaritalStatus', 'Age', 'Fault', 'VehiclePrice', 'Deductible', 'PastNumberOfClaims',
        'AccidentArea', 'PolicyType', 'VehicleCategory', 'PoliceReportFiled', 'WitnessPresent',
        'AgeOfVehicle', 'Days_Policy_Accident'
    ]
    if training:
        selected_cols.append('FraudFound_P')

    df = df[selected_cols].copy()

    # Impute numeric columns
    numeric_cols = ['Age', 'Deductible', 'PastNumberOfClaims', 'AgeOfVehicle', 'Days_Policy_Accident']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Impute categorical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'FraudFound_P']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Target separation
    if training:
        y = df['FraudFound_P']
        df.drop(columns=['FraudFound_P'], inplace=True)
    else:
        y = None

    df = pd.get_dummies(df, drop_first=True)
    df.dropna(inplace=True)

    # Align columns with expected order during prediction
    if not training:
        feature_path = os.path.join("outputs", "feature_columns.pkl")
        if os.path.exists(feature_path):
            expected_cols = joblib.load(open(feature_path, 'rb'))
            df = df.reindex(columns=expected_cols, fill_value=0)

    logger.info(f"Preprocessed data shape: {df.shape}")
    return df, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "data", "vehicle_insurance_fraud_dataset.csv")

    df = pd.read_csv(dataset_path)
    X, y = preprocess_data(df)

    # Save expected columns for testing
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(X.columns.tolist(), os.path.join("outputs", "feature_columns.pkl"))

    X_train, X_test, y_train, y_test = split_and_scale(X, y)