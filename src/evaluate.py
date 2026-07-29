import os

from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):

    y_preds=model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    metrics={}

    metrics["accuracy"] = accuracy_score(y_test, y_preds)
    metrics["precision"] = precision_score(y_test, y_preds)
    metrics["recall"] = recall_score(y_test, y_preds)
    metrics["f1"] = f1_score(y_test, y_preds)
    metrics["roc_auc"] = roc_auc_score(y_test, y_probs)

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_preds)
    metrics["classification_report"] = classification_report(y_test, y_preds)

    # -------------------------------
    # Print Metrics
    # -------------------------------
    
    print("=" * 40)
    print("\nModel Performance")
    print("=" * 40)

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key:<12}: {metrics[key]:.4f}")

    print("\nConfusion Matrix")
    print(metrics["confusion_matrix"])

    print("\nClassification Report")
    print(metrics["classification_report"])
  
    return metrics

