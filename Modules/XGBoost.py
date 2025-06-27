import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger
from Modules.data_loader import load_dataset
from Modules.data_processing import preprocess_data, split_and_scale

# Setup logger
logger = setup_logger("xgboost_model")

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=10,  # try tuning this
        random_state=42
    )
    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating XGBoost model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + report)
    print("XGBoost Accuracy:", acc)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_scale(X, y)
        model = train_xgboost(X_train, y_train)
        evaluate_model(model, X_test, y_test)