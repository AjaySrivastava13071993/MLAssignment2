"""
Evaluation Module for Multiclass Classification Metrics.
Computes Accuracy, AUC, Precision, Recall, F1-Score, and MCC.
Author: Ajay Srivastava
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

def evaluate_classification_performance(model_name: str, y_true, y_pred, y_prob=None):
    """
    Computes all 6 compulsory evaluation metrics for a given classification model.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate One-vs-Rest Macro AUC score if probabilities are provided
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
    else:
        auc = 0.0
        
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        'ML Model Name': model_name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    }
    return metrics

def get_confusion_matrix_and_report(y_true, y_pred):
    """Returns confusion matrix array and detailed classification report dictionary."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return cm, report
