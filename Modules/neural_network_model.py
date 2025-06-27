import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

# Add project root to sys.path to access utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

# Setup logger
logger = setup_logger("neural_network")

def train_neural_network_with_smote(X_train, y_train):
    logger.info("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info("Training neural network model with early stopping...")
    model = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        learning_rate_init=0.001,
        alpha=0.0001,
        verbose=False
    )
    model.fit(X_resampled, y_resampled)
    logger.info("Model training complete.")
    return model

def evaluate_with_thresholds(model, X_test, y_test):
    logger.info("Evaluating thresholds for better recall on fraud class...")
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.1)
    results = []

    for t in thresholds:
        y_pred_thresh = (y_proba >= t).astype(int)
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        acc = accuracy_score(y_test, y_pred_thresh)
        results.append((t, precision, recall, f1, acc))

    logger.info("Threshold tuning results (Fraud class):")
    print("\nThreshold | Precision | Recall | F1-score | Accuracy")
    for r in results:
        print(f"{r[0]:.2f}       | {r[1]:.2f}      | {r[2]:.2f}   | {r[3]:.2f}    | {r[4]:.2f}")

    best_threshold = max(results, key=lambda x: x[2])[0]  # Max recall
    logger.info(f"Recommended threshold (max recall): {best_threshold:.2f}")
    return y_proba, best_threshold

def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from Modules.data_loader import load_dataset
    from Modules.data_processing import preprocess_data, split_and_scale

    df = load_dataset()
    if df is not None:
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_scale(X, y)

        model = train_neural_network_with_smote(X_train, y_train)
        y_proba, best_thresh = evaluate_with_thresholds(model, X_test, y_test)
        y_pred_best = (y_proba >= best_thresh).astype(int)

        logger.info("Final Evaluation using selected threshold...")
        acc = accuracy_score(y_test, y_pred_best)
        report = classification_report(y_test, y_pred_best)
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info("Classification Report:\n" + report)
        print("\nNeural Network Accuracy:", acc)
        print("Classification Report:\n", report)

        plot_roc_curve(y_test, y_proba)