"""
Visualization and Analysis Script
Script để visualize và phân tích kết quả GNN model
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = r"D:\PROJECT\Machine Learning\IOT\results"
GRAPH_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\graph_data"
MODEL_DIR = r"D:\PROJECT\Machine Learning\IOT\models"
PROCESSED_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\processed_data"

TASK = 'binary'  # 'binary' hoặc 'multi'

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_comprehensive_results():
    """Tạo comprehensive visualization của kết quả"""

    print("=" * 80)
    print("GNN MODEL ANALYSIS AND VISUALIZATION")
    print("=" * 80)

    # Load results
    print("\nLoading results...")
    results_file = os.path.join(RESULTS_DIR, f'results_{TASK}.pkl')
    config_file = os.path.join(RESULTS_DIR, f'config_{TASK}.pkl')

    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Please run training first: python train_gnn.py")
        return

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    with open(config_file, 'rb') as f:
        config = pickle.load(f)

    print(f"✓ Loaded results for {config['model_name']} model")

    # Load metadata
    with open(os.path.join(PROCESSED_DATA_DIR, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Training History
    ax1 = plt.subplot(2, 3, 1)
    plot_training_curves(config, ax1)

    # 2. Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    plot_confusion_matrix_detailed(results, metadata, TASK, ax2)

    # 3. Performance Metrics
    ax3 = plt.subplot(2, 3, 3)
    plot_performance_metrics(results, config, ax3)

    # 4. ROC Curve (if binary)
    if TASK == 'binary' and results.get('roc_auc') is not None:
        ax4 = plt.subplot(2, 3, 4)
        plot_roc_curve_detailed(results, ax4)

    # 5. Prediction Distribution
    ax5 = plt.subplot(2, 3, 5)
    plot_prediction_distribution(results, metadata, TASK, ax5)

    # 6. Model Info
    ax6 = plt.subplot(2, 3, 6)
    plot_model_info(config, metadata, ax6)

    plt.tight_layout()

    # Save
    output_path = os.path.join(RESULTS_DIR, f'comprehensive_analysis_{TASK}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive analysis saved to: {output_path}")

    plt.show()


def plot_training_curves(config, ax):
    """Plot training curves"""

    history = config.get('history', {})

    if 'train_loss' not in history:
        # Load from checkpoint
        checkpoint_path = os.path.join(MODEL_DIR, f'best_model_{TASK}.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('history', {})

    if history:
        epochs = range(1, len(history['train_loss']) + 1)

        ax2 = ax.twinx()

        # Loss
        line1 = ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.7)
        line2 = ax.plot(epochs, history['val_loss'], 'b--', label='Val Loss', linewidth=2, alpha=0.7)

        # Accuracy
        line3 = ax2.plot(epochs, history['train_acc'], 'r-', label='Train Acc', linewidth=2, alpha=0.7)
        line4 = ax2.plot(epochs, history['val_acc'], 'r--', label='Val Acc', linewidth=2, alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', color='b', fontsize=11)
        ax2.set_ylabel('Accuracy', color='r', fontsize=11)
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right', fontsize=9)

        ax.set_title('Training History', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training history available',
                ha='center', va='center', transform=ax.transAxes)


def plot_confusion_matrix_detailed(results, metadata, task, ax):
    """Plot detailed confusion matrix"""

    cm = results['confusion_matrix']

    if task == 'binary':
        labels = ['Benign', 'Attack']
    else:
        labels = metadata['class_names']

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Normalized'}, ax=ax)

    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)


def plot_performance_metrics(results, config, ax):
    """Plot performance metrics as bar chart"""

    metrics = {
        'Accuracy': results['test_acc'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1']
    }

    if results.get('roc_auc') is not None:
        metrics['ROC-AUC'] = results['roc_auc']

    names = list(metrics.keys())
    values = list(metrics.values())

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Rotate x labels if too many
    if len(names) > 4:
        ax.tick_params(axis='x', rotation=45)


def plot_roc_curve_detailed(results, ax):
    """Plot ROC curve"""

    # Need to recalculate from predictions
    # This is a placeholder - in real scenario, save TPR/FPR during training

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2, alpha=0.5)

    if results.get('roc_auc'):
        # Placeholder diagonal
        ax.text(0.5, 0.5, f'ROC-AUC = {results["roc_auc"]:.4f}',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_prediction_distribution(results, metadata, task, ax):
    """Plot prediction distribution"""

    predictions = results['predictions']
    true_labels = results['true_labels']

    if task == 'binary':
        labels = ['Benign', 'Attack']
    else:
        labels = [metadata['class_names'][i] for i in range(len(metadata['class_names']))]

    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)

    colors = ['#2ecc71' if task == 'binary' and i == 0 else '#e74c3c'
              for i in range(len(unique))]

    bars = ax.bar([labels[i] for i in unique], counts, color=colors, alpha=0.7, edgecolor='black')

    # Add percentages
    total = len(predictions)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    if len(labels) > 5:
        ax.tick_params(axis='x', rotation=45)


def plot_model_info(config, metadata, ax):
    """Display model information"""

    ax.axis('off')

    info_text = f"""
    MODEL CONFIGURATION
    {'=' * 40}
    
    Model Type: {config['model_name']}
    Hidden Channels: {config['hidden_channels']}
    Number of Layers: {config['num_layers']}
    Dropout Rate: {config['dropout']}
    
    TRAINING SETUP
    {'=' * 40}
    
    Learning Rate: {config['learning_rate']}
    Weight Decay: {config['weight_decay']}
    Epochs Trained: {config['best_epoch']}
    Task: {config['task'].upper()}
    
    RESULTS
    {'=' * 40}
    
    Best Val Accuracy: {config['best_val_acc']:.4f}
    Test Accuracy: {config['test_results']['accuracy']:.4f}
    Test F1-Score: {config['test_results']['f1']:.4f}
    Test Precision: {config['test_results']['precision']:.4f}
    Test Recall: {config['test_results']['recall']:.4f}
    """

    if config['test_results'].get('roc_auc'):
        info_text += f"\n    Test ROC-AUC: {config['test_results']['roc_auc']:.4f}"

    ax.text(0.1, 0.95, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def compare_models():
    """So sánh nhiều models nếu có"""

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Check for multiple model results
    model_results = {}

    for task in ['binary', 'multi']:
        config_file = os.path.join(RESULTS_DIR, f'config_{task}.pkl')
        if os.path.exists(config_file):
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
            model_results[f"{config['model_name']}_{task}"] = config['test_results']

    if len(model_results) == 0:
        print("No model results found for comparison.")
        return

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Accuracy comparison
    models = list(model_results.keys())
    accuracies = [model_results[m]['accuracy'] for m in models]

    axes[0].barh(models, accuracies, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(models))))
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlim([0, 1])

    for i, (model, acc) in enumerate(zip(models, accuracies)):
        axes[0].text(acc, i, f' {acc:.4f}', va='center', fontsize=10, fontweight='bold')

    # Plot 2: All metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [model_results[m].get(metric, 0) for m in models]
        axes[1].bar(x + i * width, values, width, label=metric.capitalize())

    axes[1].set_xlabel('Models', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend(fontsize=10)
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Model comparison saved to: {output_path}")

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main visualization function"""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "GNN MODEL VISUALIZATION" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Plot comprehensive results
    plot_comprehensive_results()

    # Compare models if available
    compare_models()

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETED!")
    print("=" * 80)
    print(f"Results saved in: {RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

