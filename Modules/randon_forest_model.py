import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger
from data_loader import load_dataset
from data_processing import preprocess_data, split_and_scale

# Initialize logger
logger = setup_logger("ensemble_model")

def train_ensemble_model(X_train, y_train):
    """
    Train an ensemble of Logistic Regression and Balanced Random Forest.
    """
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
    """
    Evaluate ensemble model and log results.
    """
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

if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_scale(X, y)

        model = train_ensemble_model(X_train, y_train)
        evaluate_ensemble(model, X_test, y_test)
