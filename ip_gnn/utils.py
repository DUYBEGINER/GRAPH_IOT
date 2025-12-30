"""Utilities for endpoint_gnn package."""

import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report
)
from typing import Dict, Optional
import json

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """Get PyTorch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    
    return device


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional, for AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    }
    
    # FAR (False Alarm Rate) and Detection Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["far"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["detection_rate"] = metrics["recall"]
    
    # AUC if probabilities provided
    if y_probs is not None:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        metrics["auc"] = auc(fpr, tpr)
    
    return metrics


def save_metrics_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_probs: np.ndarray, metrics: Dict[str, float],
                       output_dir: str, history: Optional[Dict] = None):
    """Save comprehensive performance visualization plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        metrics: Computed metrics dictionary
        output_dir: Directory to save plots
        history: Training history (optional)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'], ax=ax)
    ax.set_title('Confusion Matrix - IP GNN', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - IP GNN', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1', 0),
        metrics.get('auc', 0)
    ]
    
    bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics - IP GNN', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Plots saved to {output_path}/")


def save_metrics_report(metrics: Dict[str, float], output_dir: str, 
                        y_true: np.ndarray = None, y_pred: np.ndarray = None,
                        latency: Optional[float] = None):
    """Save metrics to JSON and CSV files.
    
    Args:
        metrics: Metrics dictionary
        output_dir: Directory to save reports
        y_true: True labels (for classification report)
        y_pred: Predicted labels (for classification report)
        latency: Inference latency in seconds (optional)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add latency if provided
    if latency is not None:
        metrics['latency_seconds'] = latency
        metrics['latency_ms'] = latency * 1000
    
    # Save as JSON
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame([metrics])
    df.to_csv(output_path / 'metrics.csv', index=False)
    
    # Save classification report if labels provided
    if y_true is not None and y_pred is not None:
        report = classification_report(y_true, y_pred, 
                                       target_names=['Benign', 'Attack'],
                                       digits=4)
        with open(output_path / 'classification_report.txt', 'w') as f:
            f.write("Classification Report - IP GNN\n")
            f.write("=" * 60 + "\n")
            f.write(report)
            f.write("\n\nDetailed Metrics\n")
            f.write("=" * 60 + "\n")
            for key, value in metrics.items():
                f.write(f"{key:20s}: {value:.6f}\n")
    
    logger.info(f"📄 Metrics saved to {output_path}/")


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        """Check if should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False
