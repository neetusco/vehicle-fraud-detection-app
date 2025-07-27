import os
import sys
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
from Modules.data_loader import load_dataset
from Modules.data_processing import preprocess_data, split_and_scale

logger = setup_logger("logistic_model")

MODEL_PATH = os.path.join("outputs", "models", "logistic_model.pkl")
FEATURES_PATH = os.path.join("outputs", "models", "logistic_features.pkl")

# ----------------- TRAINING -----------------

def train_logistic_regression(X_train, y_train):
    logger.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_resampled, y_resampled)
    logger.info("Training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating Logistic Regression model...")
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    print("\nLogistic Regression Evaluation Results")
    print("----------------------------------------")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

# ----------------- PREDICTION FUNCTION -----------------

def predict_logistic(input_df):
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError("Model or feature column file not found. Please train the model first.")

        model = joblib.load(MODEL_PATH)
        expected_cols = joblib.load(FEATURES_PATH)

        processed_df, _ = preprocess_data(input_df, training=False)
        processed_df = processed_df.reindex(columns=expected_cols, fill_value=0)

        prediction = model.predict(processed_df)[0]
        return prediction, model, processed_df

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e

# ----------------- MAIN EXECUTION -----------------

if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        X, y = preprocess_data(df, training=True)
        X_train, X_test, y_train, y_test = split_and_scale(X, y)

        model = train_logistic_regression(X_train, y_train)

        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        # Save model and feature columns
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X_train.columns.tolist(), FEATURES_PATH)
        logger.info(f"Model saved to: {MODEL_PATH}")
        logger.info(f"Feature columns saved to: {FEATURES_PATH}")

        evaluate_model(model, X_test, y_test)