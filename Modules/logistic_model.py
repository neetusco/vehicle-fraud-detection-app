import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add project root to path to access logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger("logistic_model")

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a logistic regression model.
    """
    logger.info("Training Logistic Regression model...")

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    # Logging results
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    # Console output
    print("\nLogistic Regression Evaluation Results")
    print("----------------------------------------")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return model, acc, cm, report
# Example usage (optional testing)
if __name__ == "__main__":
    from data_loader import load_dataset
    from data_processing import preprocess_data, split_and_scale

    df = load_dataset()
    if df is not None:
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_scale(X, y)
        model, acc, cm, report = train_logistic_regression(X_train, y_train, X_test, y_test)