import os
import sys
import json
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
from Modules.data_loader import load_dataset
from Modules.data_processing import preprocess_data, split_and_scale

logger = setup_logger("xgboost_model")

# ----------------- File Paths -----------------
MODEL_PATH = os.path.join("outputs", "models", "xgboost_model.pkl")
FEATURES_PATH = os.path.join("outputs", "models", "xgboost_features.pkl")
ACCURACY_PATH = os.path.join("outputs", "models", "xgboost_accuracy.json")

# ----------------- TRAINING -----------------
def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=10,
        random_state=42
    )
    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model

# ----------------- EVALUATION -----------------
def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating XGBoost model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + report)

    # Save accuracy to JSON
    with open(ACCURACY_PATH, "w") as f:
        json.dump({"accuracy": acc}, f)

    print("XGBoost Accuracy:", acc)
    print("Classification Report:\n", report)

# ----------------- PREDICTION FUNCTION -----------------
def predict_xgboost(input_df):
    try:
        model = joblib.load(MODEL_PATH)
        expected_cols = joblib.load(FEATURES_PATH)

        processed_df, _ = preprocess_data(input_df, training=False)
        processed_df = processed_df.reindex(columns=expected_cols, fill_value=0)

        prediction = model.predict(processed_df)[0]

        # Load accuracy
        accuracy = None
        if os.path.exists(ACCURACY_PATH):
            with open(ACCURACY_PATH) as f:
                accuracy = json.load(f).get("accuracy")

        return prediction, model, processed_df, accuracy

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        X, y = preprocess_data(df, training=True)
        X_train, X_test, y_train, y_test = split_and_scale(X, y)

        model = train_xgboost(X_train, y_train)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X_train.columns.tolist(), FEATURES_PATH)

        evaluate_model(model, X_test, y_test)