import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger("data_processing")

def preprocess_data(df):
    """
    Encode categorical features and scale numerical features.
    """
    logger.info("Starting preprocessing...")

    # Drop identifier columns (if any)
    if 'PolicyNumber' in df.columns:
        df = df.drop(columns=['PolicyNumber'])

    # Separate features and target
    X = df.drop(columns=['FraudFound_P'])
    y = df['FraudFound_P']

    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    logger.info(f"Categorical columns: {list(cat_cols)}")
    logger.info(f"Numerical columns: {list(num_cols)}")

    # Encode categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col])
        logger.debug(f"Encoded {col}")

    # Scale numeric columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    logger.info("Preprocessing completed.")
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "data", "vehicle_insurance_fraud_dataset.csv")

    df = pd.read_csv(dataset_path)

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)