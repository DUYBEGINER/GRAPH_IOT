"""
GNN Training Script for IoT Network Anomaly Detection
Script để train GNN model cho phát hiện anomaly
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from gnn_models import create_model, count_parameters

# ============================================================================
# CONFIGURATION
# ============================================================================
GRAPH_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\graph_data"
MODEL_DIR = r"D:\PROJECT\Machine Learning\IOT\models"
RESULTS_DIR = r"D:\PROJECT\Machine Learning\IOT\results"

# Model configuration
MODEL_NAME = 'GAT'  # 'GCN', 'GAT', 'GraphSAGE', 'Hybrid'
HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
HEADS = 4  # Chỉ dùng cho GAT
DROPOUT = 0.3

# Training configuration
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 5
PATIENCE = 15  # Early stopping patience
TASK = 'binary'  # 'binary' hoặc 'multi'

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda' if torch.cuda.is_available() else 'cpu')
# ============================================================================
# TRAINING CLASS
# ============================================================================

class GNNTrainer:
    """Class để train GNN model"""

    def __init__(self, model, device, task='binary'):
        self.model = model.to(device)
        self.device = device
        self.task = task
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.best_val_acc = 0
        self.best_epoch = 0

    def train_epoch(self, data, train_mask, optimizer):
        """Train một epoch"""
        self.model.train()
        optimizer.zero_grad()

        # Forward pass
        out = self.model(data.x, data.edge_index)

        # Calculate loss chỉ trên training nodes
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = out[train_mask].argmax(dim=1)
        acc = (pred == data.y[train_mask]).float().mean()

        return loss.item(), acc.item()

    @torch.no_grad()
    def evaluate(self, data, mask):
        """Evaluate model"""
        self.model.eval()

        out = self.model(data.x, data.edge_index)

        # Calculate loss
        loss = F.cross_entropy(out[mask], data.y[mask])

        # Calculate accuracy
        pred = out[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).float().mean()

        return loss.item(), acc.item(), pred.cpu().numpy(), data.y[mask].cpu().numpy()

    def train(self, data, train_mask, val_mask, optimizer, scheduler, num_epochs, patience):
        """Full training loop"""

        print("\n" + "=" * 80)
        print("TRAINING GNN MODEL")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print("=" * 80 + "\n")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(data, train_mask, optimizer)

            # Validate
            val_loss, val_acc, _, _ = self.evaluate(data, val_mask)

            # Update scheduler
            if scheduler is not None:
                scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Check if best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                # Save best model
                self.save_model(os.path.join(MODEL_DIR, f'best_model_{self.task}.pt'))
            else:
                patience_counter += 1

            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
                break

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print("=" * 80 + "\n")

        return self.history

    def test(self, data, test_mask):
        """Test model và tính toán metrics chi tiết"""

        print("\n" + "=" * 80)
        print("TESTING MODEL")
        print("=" * 80)

        test_loss, test_acc, pred, true = self.evaluate(data, test_mask)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Calculate detailed metrics
        precision = precision_score(true, pred, average='weighted', zero_division=0)
        recall = recall_score(true, pred, average='weighted', zero_division=0)
        f1 = f1_score(true, pred, average='weighted', zero_division=0)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # ROC-AUC for binary classification
        if self.task == 'binary':
            try:
                # Get probabilities
                self.model.eval()
                with torch.no_grad():
                    out = self.model(data.x, data.edge_index)
                    probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()

                roc_auc = roc_auc_score(true, probs)
                print(f"ROC-AUC: {roc_auc:.4f}")
            except:
                roc_auc = None
        else:
            roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(true, pred)

        # Classification report
        print("\n" + "-" * 80)
        print("Classification Report:")
        print("-" * 80)
        print(classification_report(true, pred, zero_division=0))

        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': pred,
            'true_labels': true
        }

        print("=" * 80 + "\n")

        return results

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, path)

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path=None):
    """Plot training history"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")

    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """Plot confusion matrix"""

    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")

    plt.close()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function"""

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "GNN TRAINING FOR IoT ANOMALY DETECTION" + " " * 19 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Load graph data
    print("Loading graph data...")
    graph_file = f"graph_{TASK}.pt"
    data = torch.load(os.path.join(GRAPH_DATA_DIR, graph_file), weights_only=False)
    data = data.to(DEVICE)

    print(f"✓ Graph loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"✓ Features: {data.num_features}")
    print(f"✓ Classes: {len(torch.unique(data.y))}")

    # Create train/val/test masks
    print("\nCreating train/val/test splits...")
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(num_nodes * TRAIN_RATIO)
    val_size = int(num_nodes * VAL_RATIO)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    print(f"✓ Train: {train_mask.sum():,} nodes ({TRAIN_RATIO * 100:.0f}%)")
    print(f"✓ Val: {val_mask.sum():,} nodes ({VAL_RATIO * 100:.0f}%)")
    print(f"✓ Test: {test_mask.sum():,} nodes ({TEST_RATIO * 100:.0f}%)")

    # Create model
    print(f"\nCreating {MODEL_NAME} model...")
    num_classes = len(torch.unique(data.y))

    model_kwargs = {
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }

    if MODEL_NAME in ['GAT', 'Hybrid']:
        model_kwargs['heads'] = HEADS

    model = create_model(
        MODEL_NAME,
        in_channels=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=num_classes,
        **model_kwargs
    )

    print(f"✓ Model created: {count_parameters(model):,} parameters")

    # Create trainer
    trainer = GNNTrainer(model, DEVICE, task=TASK)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Train
    history = trainer.train(
        data, train_mask, val_mask,
        optimizer, scheduler,
        NUM_EPOCHS, PATIENCE
    )

    # Load best model
    print("Loading best model...")
    trainer.load_model(os.path.join(MODEL_DIR, f'best_model_{TASK}.pt'))

    # Test
    results = trainer.test(data, test_mask)

    # Save results
    print("Saving results...")

    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(RESULTS_DIR, f'training_history_{TASK}.png')
    )

    # Plot confusion matrix
    if TASK == 'binary':
        class_names = ['Benign', 'Attack']
    else:
        # Load class names from metadata
        with open(os.path.join(r"D:\PROJECT\Machine Learning\IOT\processed_data", "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        class_names = metadata['class_names']

    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=class_names,
        save_path=os.path.join(RESULTS_DIR, f'confusion_matrix_{TASK}.png')
    )

    # Save results to file
    results_file = os.path.join(RESULTS_DIR, f'results_{TASK}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    # Save configuration
    config = {
        'model_name': MODEL_NAME,
        'hidden_channels': HIDDEN_CHANNELS,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'num_epochs': NUM_EPOCHS,
        'task': TASK,
        'best_epoch': trainer.best_epoch,
        'best_val_acc': trainer.best_val_acc,
        'test_results': {
            'accuracy': results['test_acc'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'roc_auc': results['roc_auc']
        }
    }

    with open(os.path.join(RESULTS_DIR, f'config_{TASK}.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

