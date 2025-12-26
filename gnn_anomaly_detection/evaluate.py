"""
Evaluation and Visualization Module
====================================
Module này chứa các functions để đánh giá và visualize kết quả.

Bao gồm:
- Visualization của training curves
- Confusion matrix plot
- ROC curve và Precision-Recall curve
- Feature importance (nếu có)
- Classification report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                            roc_curve, precision_recall_curve, auc)
import torch
import os

import config


def plot_training_history(history: dict, save_path: str = None):
    """
    Vẽ biểu đồ training history.

    Args:
        history: Dictionary chứa training history
        save_path: Đường dẫn để lưu figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # F1 Score
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_f1'], 'g-', label='Val F1')
    ax3.axvline(x=history['best_epoch'], color='r', linestyle='--',
                label=f'Best Epoch ({history["best_epoch"]})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Validation F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # AUC-ROC
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_auc'], 'purple', label='Val AUC-ROC')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC-ROC')
    ax4.set_title('Validation AUC-ROC')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved training history plot to: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None,
                         labels: list = None):
    """
    Vẽ confusion matrix.

    Args:
        cm: Confusion matrix array
        save_path: Đường dẫn để lưu
        labels: Tên của các classes
    """
    if labels is None:
        labels = ['Normal', 'Anomaly']

    plt.figure(figsize=(8, 6))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    # Add percentage annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]*100:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)

    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved confusion matrix to: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray,
                   save_path: str = None):
    """
    Vẽ ROC curve.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        save_path: Đường dẫn để lưu
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random classifier')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'roc_curve.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved ROC curve to: {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_probs: np.ndarray,
                                save_path: str = None):
    """
    Vẽ Precision-Recall curve.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        save_path: Đường dẫn để lưu
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.axhline(y=baseline, color='r', linestyle='--', linewidth=1,
                label=f'Baseline ({baseline:.4f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'pr_curve.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved Precision-Recall curve to: {save_path}")


def plot_all_curves(model, data, device: str = None, save_dir: str = None):
    """
    Vẽ tất cả các curves (ROC, PR) cho test set.

    Args:
        model: Trained model
        data: Graph data
        device: Device
        save_dir: Directory để lưu các plots
    """
    if device is None:
        device = config.DEVICE
    if save_dir is None:
        save_dir = config.OUTPUT_DIR

    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        out = model(data)
        probs = torch.exp(out[data.test_mask])[:, 1].cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()

    # Plot curves
    plot_roc_curve(y_true, probs, os.path.join(save_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_true, probs, os.path.join(save_dir, 'pr_curve.png'))


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  save_path: str = None) -> str:
    """
    Tạo classification report chi tiết.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Đường dẫn để lưu report

    Returns:
        Classification report string
    """
    report = classification_report(y_true, y_pred,
                                   target_names=['Normal', 'Anomaly'],
                                   digits=4)

    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'classification_report.txt')

    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)

    print(f"[INFO] Saved classification report to: {save_path}")

    return report


def print_detailed_metrics(test_metrics: dict):
    """
    In các metrics chi tiết.

    Args:
        test_metrics: Dictionary chứa test metrics
    """
    print("\n" + "="*60)
    print("DETAILED METRICS")
    print("="*60)

    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-"*32)
    print(f"{'Accuracy':<20} {test_metrics['accuracy']:>10.4f}")
    print(f"{'Precision':<20} {test_metrics['precision']:>10.4f}")
    print(f"{'Recall':<20} {test_metrics['recall']:>10.4f}")
    print(f"{'F1 Score':<20} {test_metrics['f1']:>10.4f}")
    print(f"{'AUC-ROC':<20} {test_metrics['auc_roc']:>10.4f}")
    print(f"{'AUC-PR':<20} {test_metrics['auc_pr']:>10.4f}")

    cm = test_metrics['confusion_matrix']
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print(f"\n{'Additional Metrics':<20}")
    print("-"*32)
    print(f"{'True Negatives':<20} {tn:>10,}")
    print(f"{'False Positives':<20} {fp:>10,}")
    print(f"{'False Negatives':<20} {fn:>10,}")
    print(f"{'True Positives':<20} {tp:>10,}")

    # Specificity và các metrics khác
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n{'Specificity':<20} {specificity:>10.4f}")
    print(f"{'False Positive Rate':<20} {fpr:>10.4f}")
    print(f"{'False Negative Rate':<20} {fnr:>10.4f}")


def create_summary_report(history: dict, test_metrics: dict,
                         model_info: dict, save_path: str = None):
    """
    Tạo báo cáo tổng hợp.

    Args:
        history: Training history
        test_metrics: Test metrics
        model_info: Thông tin về model
        save_path: Đường dẫn để lưu report
    """
    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'summary_report.txt')

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("GNN ANOMALY DETECTION - SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")

        f.write("MODEL INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Model Type: {model_info.get('type', 'N/A')}\n")
        f.write(f"Input Dimension: {model_info.get('input_dim', 'N/A')}\n")
        f.write(f"Hidden Dimension: {model_info.get('hidden_dim', 'N/A')}\n")
        f.write(f"Number of Layers: {model_info.get('num_layers', 'N/A')}\n")
        f.write(f"Total Parameters: {model_info.get('num_params', 'N/A'):,}\n")

        f.write("\n\nTRAINING INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Training Time: {history.get('training_time', 0):.2f} seconds\n")
        f.write(f"Best Epoch: {history.get('best_epoch', 'N/A')}\n")
        f.write(f"Best Validation F1: {history.get('best_val_f1', 0):.4f}\n")

        f.write("\n\nTEST RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {test_metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC:   {test_metrics['auc_roc']:.4f}\n")
        f.write(f"AUC-PR:    {test_metrics['auc_pr']:.4f}\n")

        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-"*40 + "\n")
        cm = test_metrics['confusion_matrix']
        f.write(f"                 Predicted\n")
        f.write(f"                 Normal  Anomaly\n")
        f.write(f"Actual Normal    {cm[0,0]:6d}  {cm[0,1]:6d}\n")
        f.write(f"       Anomaly   {cm[1,0]:6d}  {cm[1,1]:6d}\n")

    print(f"[INFO] Saved summary report to: {save_path}")

