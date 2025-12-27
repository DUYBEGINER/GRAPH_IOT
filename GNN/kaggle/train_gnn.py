"""
GNN Training Script for Kaggle - IoT Network Anomaly Detection
Script hoàn chỉnh bao gồm cả model definitions và training
Chỉ cần chạy file này trên Kaggle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
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
# GNN MODEL DEFINITIONS
# ============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Classifier
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
    """Graph Attention Network"""

    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_channels * heads))

        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                  heads=1, concat=False, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Classifier
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
    """GraphSAGE"""

    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, dropout=0.5, aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class HybridGNN(nn.Module):
    """Hybrid GNN combining GCN and GAT"""

    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, heads=4, dropout=0.5):
        super(HybridGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GCN branch
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()

        # GAT branch
        self.gat_convs = nn.ModuleList()
        self.gat_bns = nn.ModuleList()

        # First layer
        self.gcn_convs.append(GCNConv(in_channels, hidden_channels))
        self.gcn_bns.append(BatchNorm(hidden_channels))
        self.gat_convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.gat_bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gcn_bns.append(BatchNorm(hidden_channels))
            self.gat_convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
            self.gat_bns.append(BatchNorm(hidden_channels))

        # Fusion
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_bn = BatchNorm(hidden_channels)

        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # GCN branch
        x_gcn = x
        for i in range(self.num_layers - 1):
            x_gcn = self.gcn_convs[i](x_gcn, edge_index)
            x_gcn = self.gcn_bns[i](x_gcn)
            x_gcn = F.relu(x_gcn)
            x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)

        # GAT branch
        x_gat = x
        for i in range(self.num_layers - 1):
            x_gat = self.gat_convs[i](x_gat, edge_index)
            x_gat = self.gat_bns[i](x_gat)
            x_gat = F.elu(x_gat)
            x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)

        # Fusion
        x = torch.cat([x_gcn, x_gat], dim=1)
        x = self.fusion(x)
        x = self.fusion_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def create_model(model_name, in_channels, hidden_channels, num_classes, **kwargs):
    """Factory function để tạo model"""
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'Hybrid': HybridGNN
    }
    if model_name not in models:
        raise ValueError(f"Model {model_name} không hỗ trợ. Chọn: {list(models.keys())}")
    return models[model_name](in_channels, hidden_channels, num_classes, **kwargs)


def count_parameters(model):
    """Đếm số parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# KAGGLE CONFIGURATION
# ============================================================================
WORKING_DIR = "/kaggle/working"
GRAPH_DATA_DIR = "/kaggle/working"
MODEL_DIR = os.path.join(WORKING_DIR, "models")
RESULTS_DIR = os.path.join(WORKING_DIR, "results")

# Model config
MODEL_NAME = 'GAT'  # 'GCN', 'GAT', 'GraphSAGE', 'Hybrid'
HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
HEADS = 4
DROPOUT = 0.3

# Training config
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 30
PATIENCE = 15
TASK = 'binary'  # 'binary' hoặc 'multi'

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {DEVICE}")


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

        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        pred = out[train_mask].argmax(dim=1)
        acc = (pred == data.y[train_mask]).float().mean()
        return loss.item(), acc.item()

    @torch.no_grad()
    def evaluate(self, data, mask):
        """Evaluate model"""
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
        pred = out[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).float().mean()
        return loss.item(), acc.item(), pred.cpu().numpy(), data.y[mask].cpu().numpy()

    def train(self, data, train_mask, val_mask, optimizer, scheduler, num_epochs, patience):
        """Full training loop"""
        print("\n" + "=" * 80)
        print("🚀 TRAINING GNN MODEL")
        print("=" * 80)
        print(f"📍 Device: {self.device}")
        print(f"🧠 Model: {self.model.__class__.__name__}")
        print(f"📊 Parameters: {count_parameters(self.model):,}")
        print(f"⏰ Epochs: {num_epochs}")
        print("=" * 80 + "\n")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(data, train_mask, optimizer)
            val_loss, val_acc, _, _ = self.evaluate(data, val_mask)

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

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

        print(f"\n✅ Best Val Acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        return self.history

    def test(self, data, test_mask):
        """Test model"""
        print("\n" + "=" * 80)
        print("🧪 TESTING MODEL")
        print("=" * 80)

        test_loss, test_acc, pred, true = self.evaluate(data, test_mask)

        print(f"📊 Test Accuracy: {test_acc:.4f}")

        precision = precision_score(true, pred, average='weighted', zero_division=0)
        recall = recall_score(true, pred, average='weighted', zero_division=0)
        f1 = f1_score(true, pred, average='weighted', zero_division=0)

        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall: {recall:.4f}")
        print(f"📊 F1-Score: {f1:.4f}")

        # ROC-AUC for binary
        roc_auc = None
        if self.task == 'binary':
            try:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(data.x, data.edge_index)
                    probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
                roc_auc = roc_auc_score(true, probs)
                print(f"📊 ROC-AUC: {roc_auc:.4f}")
            except:
                pass

        cm = confusion_matrix(true, pred)
        print("\n" + "-" * 80)
        print("📋 Classification Report:")
        print("-" * 80)
        print(classification_report(true, pred, zero_division=0))

        return {
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

    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, path)

    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
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
    """Plot confusion matrix"""
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
# MAIN
# ============================================================================

def main():
    """Main training function"""

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("🤖 GNN TRAINING FOR IoT ANOMALY DETECTION - KAGGLE")
    print("=" * 80 + "\n")

    # Load graph
    print("📂 Loading graph data...")
    graph_file = f"graph_{TASK}.pt"
    graph_path = os.path.join(GRAPH_DATA_DIR, graph_file)

    if not os.path.exists(graph_path):
        print(f"❌ ERROR: Graph file not found: {graph_path}")
        print("📁 Available files:")
        for f in os.listdir(WORKING_DIR):
            print(f"  - {f}")
        return

    data = torch.load(graph_path, weights_only=False)
    data = data.to(DEVICE)

    print(f"✅ Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"✅ Features: {data.num_features}")
    print(f"✅ Classes: {len(torch.unique(data.y))}")

    # Load metadata
    metadata_path = os.path.join(GRAPH_DATA_DIR, "graph_metadata.pkl")
    class_names = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            if TASK in metadata and 'class_names' in metadata[TASK]:
                class_names = metadata[TASK]['class_names']

    if class_names is None:
        class_names = ['Benign', 'Attack'] if TASK == 'binary' else [f'Class_{i}' for i in
                                                                     range(len(torch.unique(data.y)))]

    # Create masks
    print("\n📊 Creating data splits...")
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

    print(f"✅ Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

    # Create model
    print(f"\n🏗️  Creating {MODEL_NAME} model...")
    num_classes = len(torch.unique(data.y))

    model_kwargs = {'num_layers': NUM_LAYERS, 'dropout': DROPOUT}
    if MODEL_NAME in ['GAT', 'Hybrid']:
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
    trainer = GNNTrainer(model, DEVICE, task=TASK)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = trainer.train(data, train_mask, val_mask, optimizer, scheduler, NUM_EPOCHS, PATIENCE)

    # Test
    print("\n🔄 Loading best model...")
    trainer.load_model(os.path.join(MODEL_DIR, f'best_model_{TASK}.pt'))
    results = trainer.test(data, test_mask)

    # Save results
    print("\n💾 Saving results...")
    plot_training_history(history, os.path.join(RESULTS_DIR, f'training_history_{TASK}.png'))
    plot_confusion_matrix(results['confusion_matrix'], class_names,
                          os.path.join(RESULTS_DIR, f'confusion_matrix_{TASK}.png'))

    with open(os.path.join(RESULTS_DIR, f'results_{TASK}.pkl'), 'wb') as f:
        pickle.dump(results, f)

    config = {
        'model_name': MODEL_NAME,
        'hidden_channels': HIDDEN_CHANNELS,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
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

    # Summary
    with open(os.path.join(RESULTS_DIR, 'summary.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Task: {TASK}\n")
        f.write(f"Best Epoch: {trainer.best_epoch}\n")
        f.write(f"Best Val Acc: {trainer.best_val_acc:.4f}\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  Accuracy: {results['test_acc']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall: {results['recall']:.4f}\n")
        f.write(f"  F1-Score: {results['f1']:.4f}\n")
        if results['roc_auc']:
            f.write(f"  ROC-AUC: {results['roc_auc']:.4f}\n")

    # ZIP for download
    print("\n" + "=" * 80)
    print("📦 CREATING ZIP FILES")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_results = f'gnn_results_{TASK}_{timestamp}'
    zip_models = f'gnn_models_{TASK}_{timestamp}'

    shutil.make_archive(os.path.join(WORKING_DIR, zip_results), 'zip', RESULTS_DIR)
    shutil.make_archive(os.path.join(WORKING_DIR, zip_models), 'zip', MODEL_DIR)

    print(f"✅ Results: {zip_results}.zip")
    print(f"✅ Models: {zip_models}.zip")

    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📊 Test Accuracy: {results['test_acc']:.4f}")
    print(f"📊 F1-Score: {results['f1']:.4f}")
    print(f"\n📥 Download: {zip_results}.zip & {zip_models}.zip")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()