"""Metrics computation for IDS evaluation."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple


def compute_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute False Alarm Rate (FAR).
    
    FAR = FP / (FP + TN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        False Alarm Rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return far


def compute_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Detection Rate (DR) = Recall for attack class.
    
    DR = TP / (TP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Detection rate
    """
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    task_type: str = "binary"
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: "binary" or "multiclass"
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    if task_type == "binary":
        # Binary classification metrics
        metrics["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics["detection_rate"] = metrics["recall"]
        metrics["far"] = compute_far(y_true, y_pred)
        
    else:  # multiclass
        # Macro-averaged metrics
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Weighted-averaged metrics
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for printing (e.g., "Train", "Val", "Test")
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("-" * 50)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names
        
    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ["Benign", "Attack"]
    
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Labels tensor
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    # Count samples per class
    counts = torch.bincount(y, minlength=num_classes).float()
    
    # Compute weights: inverse frequency
    weights = len(y) / (num_classes * counts)
    
    # Handle zero counts
    weights[counts == 0] = 0.0
    
    return weights
