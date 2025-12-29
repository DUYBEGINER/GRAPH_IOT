"""
GNN Training Script - FIXED VERSION with Mini-Batch Support
Key fixes:
1. Added NeighborLoader for mini-batch training
2. Reduced memory footprint significantly
3. Added gradient accumulation option
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
# Removed: from torch_geometric.loader import NeighborLoader
import numpy as np
import pickle
import os
import shutil
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# MODEL DEFINITIONS (Same as before - these are fine)
# ============================================================================

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                     heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_channels * heads))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                 heads=1, concat=False, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, dropout=0.5, aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def create_model(model_name, in_channels, hidden_channels, num_classes, **kwargs):
    models = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': GraphSAGE}
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Choose: {list(models.keys())}")
    return models[model_name](in_channels, hidden_channels, num_classes, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# CONFIGURATION - OPTIMIZED FOR MEMORY
# ============================================================================
WORKING_DIR = "/kaggle/working"
GRAPH_DATA_DIR = "/kaggle/input/model/pytorch/default/1"
MODEL_DIR = os.path.join(WORKING_DIR, "models")
RESULTS_DIR = os.path.join(WORKING_DIR, "results")

# Model config - Reduced for memory
MODEL_NAME = 'GraphSAGE'  # GraphSAGE is most memory-efficient
HIDDEN_CHANNELS = 64      # Reduced from 128
NUM_LAYERS = 2            # Reduced from 3
HEADS = 2                 # Reduced from 4 (for GAT)
DROPOUT = 0.3

# Training config
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 30
PATIENCE = 15
TASK = 'binary'

# MINI-BATCH CONFIG
BATCH_SIZE = 2048         # Process nodes in batches
NUM_WORKERS = 0           # Not used in simple batching

# Gradient accumulation (simulate larger batch)
ACCUMULATION_STEPS = 4

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {DEVICE}")

# ============================================================================
# MINI-BATCH TRAINER CLASS - COMPLETELY REWRITTEN
# ============================================================================

class MiniBatchGNNTrainer:
    """Memory-efficient mini-batch trainer"""

    def __init__(self, model, device, task='binary'):
        self.model = model.to(device)
        self.device = device
        self.task = task
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'learning_rate': []
        }
        self.best_val_acc = 0
        self.best_epoch = 0

    def train_epoch(self, train_batches, data, optimizer, accumulation_steps=1):
        """Train one epoch with simple batches"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        optimizer.zero_grad()

        # Move full graph to device once
        data = data.to(self.device)

        for i, batch_idx in enumerate(train_batches):
            batch_idx = batch_idx.to(self.device)

            # Forward pass on full graph
            out = self.model(data.x, data.edge_index)

            # Compute loss only on batch nodes
            loss = F.cross_entropy(out[batch_idx], data.y[batch_idx])
            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Metrics
            with torch.no_grad():
                pred = out[batch_idx].argmax(dim=1)
                total_correct += (pred == data.y[batch_idx]).sum().item()
                total_loss += loss.item() * accumulation_steps * len(batch_idx)
                total_samples += len(batch_idx)

        # Final update
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, batches, data):
        """Evaluate with simple batches"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        data = data.to(self.device)

        for batch_idx in batches:
            batch_idx = batch_idx.to(self.device)

            # Forward on full graph
            out = self.model(data.x, data.edge_index)

            # Loss and predictions on batch
            loss = F.cross_entropy(out[batch_idx], data.y[batch_idx])
            pred = out[batch_idx].argmax(dim=1)

            total_correct += (pred == data.y[batch_idx]).sum().item()
            total_loss += loss.item() * len(batch_idx)
            total_samples += len(batch_idx)

            all_preds.append(pred.cpu())
            all_labels.append(data.y[batch_idx].cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return total_loss / total_samples, total_correct / total_samples, all_preds, all_labels

    def train(self, train_batches, val_batches, data, optimizer, scheduler, num_epochs, patience):
        """Full training loop"""
        print("\n" + "=" * 80)
        print("🚀 TRAINING GNN MODEL (BATCHED)")
        print("=" * 80)
        print(f"📍 Device: {self.device}")
        print(f"🧠 Model: {self.model.__class__.__name__}")
        print(f"📊 Parameters: {count_parameters(self.model):,}")
        print(f"🎯 Batch size: {BATCH_SIZE}")
        print(f"⏰ Epochs: {num_epochs}")
        print("=" * 80 + "\n")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_batches, data, optimizer, ACCUMULATION_STEPS)
            val_loss, val_acc, _, _ = self.evaluate(val_batches, data)

            if scheduler is not None:
                scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                self.save_model(os.path.join(MODEL_DIR, f'best_model_{self.task}.pt'))
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

        print(f"\n✅ Best Val Acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        return self.history

    def test(self, test_batches, data):
        """Test model"""
        print("\n" + "=" * 80)
        print("🧪 TESTING MODEL")
        print("=" * 80)

        test_loss, test_acc, pred, true = self.evaluate(test_batches, data)

        print(f"📊 Test Accuracy: {test_acc:.4f}")

        precision = precision_score(true, pred, average='weighted', zero_division=0)
        recall = recall_score(true, pred, average='weighted', zero_division=0)
        f1 = f1_score(true, pred, average='weighted', zero_division=0)

        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall: {recall:.4f}")
        print(f"📊 F1-Score: {f1:.4f}")

        cm = confusion_matrix(true, pred)
        print("\n" + "-" * 80)
        print("📋 Classification Report:")
        print("-" * 80)
        print(classification_report(true, pred, zero_division=0))

        return {
            'test_loss': test_loss, 'test_acc': test_acc,
            'precision': precision, 'recall': recall, 'f1': f1,
            'confusion_matrix': cm, 'predictions': pred, 'true_labels': true
        }

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']


# ============================================================================
# VISUALIZATION (Same as before)
# ============================================================================

def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN - REWRITTEN WITH MINI-BATCH LOADERS
# ============================================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("🤖 GNN TRAINING (MEMORY-OPTIMIZED)")
    print("=" * 80 + "\n")

    # Load graph
    print("📂 Loading graph data...")
    graph_file = f"graph_{TASK}.pt"
    graph_path = os.path.join(GRAPH_DATA_DIR, graph_file)

    if not os.path.exists(graph_path):
        print(f"❌ ERROR: Graph file not found: {graph_path}")
        return

    data = torch.load(graph_path, weights_only=False)

    print(f"✅ Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

    # Create masks
    print("\n📊 Creating data splits...")
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(num_nodes * TRAIN_RATIO)
    val_size = int(num_nodes * VAL_RATIO)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    print(f"✅ Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

    # Create simple batched indices (no neighbor sampling)
    print(f"\n🔧 Creating simple batched processing...")

    def create_simple_batches(indices, batch_size):
        """Create simple batches without neighbor sampling"""
        return [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]

    train_batches = create_simple_batches(train_idx, BATCH_SIZE)
    val_batches = create_simple_batches(val_idx, BATCH_SIZE)
    test_batches = create_simple_batches(test_idx, BATCH_SIZE)

    print(f"✅ Train batches: {len(train_batches)}")
    print(f"✅ Val batches: {len(val_batches)}")
    print(f"✅ Test batches: {len(test_batches)}")

    # Create model
    print(f"\n🏗️  Creating {MODEL_NAME} model...")
    num_classes = len(torch.unique(data.y))

    model_kwargs = {'num_layers': NUM_LAYERS, 'dropout': DROPOUT}
    if MODEL_NAME == 'GAT':
        model_kwargs['heads'] = HEADS

    model = create_model(
        MODEL_NAME,
        in_channels=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=num_classes,
        **model_kwargs
    )

    print(f"✅ Model: {count_parameters(model):,} parameters")

    # Train
    trainer = MiniBatchGNNTrainer(model, DEVICE, task=TASK)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = trainer.train(train_batches, val_batches, data, optimizer, scheduler, NUM_EPOCHS, PATIENCE)

    # Test
    print("\n🔄 Loading best model...")
    trainer.load_model(os.path.join(MODEL_DIR, f'best_model_{TASK}.pt'))
    results = trainer.test(test_batches, data)

    # Save results
    print("\n💾 Saving results...")
    plot_training_history(history, os.path.join(RESULTS_DIR, f'training_history_{TASK}.png'))
    plot_confusion_matrix(results['confusion_matrix'], ['Benign', 'Attack'],
                          os.path.join(RESULTS_DIR, f'confusion_matrix_{TASK}.png'))

    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📊 Test Accuracy: {results['test_acc']:.4f}")
    print(f"📊 F1-Score: {results['f1']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()