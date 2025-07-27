import os
import sys
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
from Modules.data_loader import load_dataset
from Modules.data_processing import preprocess_data, split_and_scale

logger = setup_logger("ensemble_model")

MODEL_PATH = os.path.join("outputs", "models", "random_forest_ensemble.pkl")
FEATURES_PATH = os.path.join("outputs", "models", "random_forest_features.pkl")

# ------------------ TRAINING ------------------

def train_ensemble_model(X_train, y_train):
    logger.info("Training ensemble model with Logistic Regression and BalancedRandomForest...")

    lr = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('brf', brf)],
        voting='soft'
    )

    ensemble.fit(X_train, y_train)
    logger.info("Ensemble model training complete.")
    return ensemble

def evaluate_ensemble(model, X_test, y_test):
    logger.info("Evaluating ensemble model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info("Classification Report:\n" + report)

    print("Ensemble Model Accuracy:", acc)
    print("ROC AUC:", roc_auc)
    print("Classification Report:\n", report)

# ------------------ PREDICTION ------------------

def predict_random_forest(input_df):
    try:
        model = joblib.load(MODEL_PATH)
        expected_cols = joblib.load(FEATURES_PATH)

        processed_df, _ = preprocess_data(input_df, training=False)
        processed_df = processed_df.reindex(columns=expected_cols, fill_value=0)

        prediction = model.predict(processed_df)[0]
        return prediction, model, processed_df

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e

# ------------------ MAIN BLOCK ------------------

if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        X, y = preprocess_data(df, training=True)
        X_train, X_test, y_train, y_test = split_and_scale(X, y)

        model = train_ensemble_model(X_train, y_train)

        # Save model and features
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X_train.columns.tolist(), FEATURES_PATH)
        logger.info(f"Model and features saved to {MODEL_PATH}")

        evaluate_ensemble(model, X_test, y_test)