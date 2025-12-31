"""Utilities for ip_gnn package."""

import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Optional
import json


def get_device(device_str: str = "auto") -> torch.device:
    """Get PyTorch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"   Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("   Using Apple Metal (MPS)")
        else:
            device = torch.device("cpu")
            print("   Using CPU")
    else:
        device = torch.device(device_str)
        print(f"   Using device: {device}")
    
    return device


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    }
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positive"] = int(tp)
    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    
    # FAR (False Alarm Rate) and Detection Rate
    metrics["far"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["detection_rate"] = metrics["recall"]
    
    # AUC and AP if probabilities provided
    if y_probs is not None:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        metrics["auc"] = auc(fpr, tpr)
        metrics["average_precision"] = average_precision_score(y_true, y_probs)
    
    return metrics


def save_metrics_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_probs: np.ndarray, metrics: Dict[str, float],
                       output_dir: str, history: Optional[Dict] = None,
                       latency_ms: Optional[float] = None):
    """Save comprehensive performance visualization plots.
    
    Saves:
    - Confusion Matrix (raw counts)
    - Confusion Matrix (normalized)
    - ROC Curve with AUC
    - Precision-Recall Curve
    - Metrics Bar Chart
    - Training History (loss, F1, accuracy)
    - All Metrics Summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # =========================================================================
    # 1. CONFUSION MATRIX (Raw counts)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'], ax=ax,
                annot_kws={'size': 14})
    ax.set_title('Confusion Matrix - IP GNN', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 2. CONFUSION MATRIX (Normalized)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', 
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'], ax=ax,
                annot_kws={'size': 14}, vmin=0, vmax=1)
    ax.set_title('Normalized Confusion Matrix - IP GNN', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 3. ROC CURVE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#4CAF50', lw=2.5, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#4CAF50')
    ax.plot([0, 1], [0, 1], color='#9E9E9E', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - IP GNN', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 4. PRECISION-RECALL CURVE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    
    ax.plot(recall_curve, precision_curve, color='#FF9800', lw=2.5,
            label=f'PR Curve (AP = {ap:.4f})')
    ax.fill_between(recall_curve, precision_curve, alpha=0.2, color='#FF9800')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - IP GNN', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 5. METRICS BAR CHART
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Avg Precision']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1', 0),
        metrics.get('auc', 0),
        metrics.get('average_precision', 0)
    ]
    colors = ['#1976D2', '#388E3C', '#FBC02D', '#D32F2F', '#7B1FA2', '#00796B']
    
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics - IP GNN', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 6. TRAINING HISTORY
    # =========================================================================
    if history is not None and len(history.get('train_loss', [])) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # F1 plot
        axes[1].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'val_accuracy' in history:
            axes[2].plot(epochs, history['val_accuracy'], 'm-', label='Val Accuracy', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Accuracy', fontsize=12)
            axes[2].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # 7. ALL METRICS SUMMARY
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    summary_lines = [
        "═" * 50,
        "       IP-GNN EVALUATION SUMMARY",
        "═" * 50,
        "",
        f"  Accuracy:          {metrics.get('accuracy', 0):.4f}",
        f"  Precision:         {metrics.get('precision', 0):.4f}",
        f"  Recall:            {metrics.get('recall', 0):.4f}",
        f"  F1 Score:          {metrics.get('f1', 0):.4f}",
        f"  AUC:               {metrics.get('auc', 0):.4f}",
        f"  Average Precision: {metrics.get('average_precision', 0):.4f}",
        "",
        "─" * 50,
        "",
        f"  False Alarm Rate:  {metrics.get('far', 0):.4f}",
        f"  Detection Rate:    {metrics.get('detection_rate', 0):.4f}",
        "",
        f"  True Positives:    {metrics.get('true_positive', 0):,}",
        f"  True Negatives:    {metrics.get('true_negative', 0):,}",
        f"  False Positives:   {metrics.get('false_positive', 0):,}",
        f"  False Negatives:   {metrics.get('false_negative', 0):,}",
    ]
    
    if latency_ms is not None:
        summary_lines.extend([
            "",
            "─" * 50,
            "",
            f"  Latency:           {latency_ms:.4f} ms/sample",
        ])
    
    summary_lines.append("")
    summary_lines.append("═" * 50)
    
    summary_text = "\n".join(summary_lines)
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {output_path}/")


def save_metrics_report(metrics: Dict[str, float], output_dir: str, 
                        y_true: np.ndarray = None, y_pred: np.ndarray = None,
                        latency: Optional[float] = None):
    """Save metrics to JSON, CSV and text files."""
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
    
    # Save classification report
    if y_true is not None and y_pred is not None:
        report = classification_report(y_true, y_pred, 
                                       target_names=['Benign', 'Attack'],
                                       digits=4)
        with open(output_path / 'classification_report.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("   IP-GNN CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("   DETAILED METRICS\n")
            f.write("=" * 60 + "\n\n")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    f.write(f"  {key:25s}: {value:.6f}\n")
                else:
                    f.write(f"  {key:25s}: {value}\n")
    
    print(f"📄 Metrics saved to {output_path}/")


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
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


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
