"""
Training GNN on Host-Connection Graph for IDS
Train Graph Neural Network trên Host-Connection Graph để detect malicious hosts
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
GRAPH_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\graph_data"
MODEL_DIR = r"D:\PROJECT\Machine Learning\IOT\models"
RESULTS_DIR = r"D:\PROJECT\Machine Learning\IOT\results"

# Training parameters
EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.5

# Model type: 'GCN', 'GAT', 'GraphSAGE'
MODEL_TYPE = 'GAT'

# Task: 'binary' hoặc 'multi'
TASK = 'binary'

# Train/Val/Test split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# GNN MODELS
# ============================================================================

class HostGCN(torch.nn.Module):
    """GCN model cho Host-Connection Graph"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class HostGAT(torch.nn.Module):
    """GAT model cho Host-Connection Graph"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, heads=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class HostGraphSAGE(torch.nn.Module):
    """GraphSAGE model cho Host-Connection Graph"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Create train/val/test masks"""
    indices = torch.randperm(num_nodes)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


def train_epoch(model, data, optimizer, mask):
    """Train one epoch"""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[mask], data.y[mask])

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, data, mask):
    """Evaluate model"""
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # Metrics
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted', zero_division=0)

        # Loss
        loss = F.nll_loss(out[mask], data.y[mask]).item()

    return loss, accuracy, precision, recall, f1, y_true, y_pred


def plot_training_history(history, output_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision
    axes[1, 0].plot(history['train_precision'], label='Train')
    axes[1, 0].plot(history['val_precision'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # F1 Score
    axes[1, 1].plot(history['train_f1'], label='Train')
    axes[1, 1].plot(history['val_f1'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score over Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to: {output_path}")


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {output_path}")


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING GNN ON HOST-CONNECTION GRAPH")
    print("=" * 80)

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load graph data
    print("\nLoading graph data...")
    graph_path = os.path.join(GRAPH_DATA_DIR, "host_graph.pt")
    metadata_path = os.path.join(GRAPH_DATA_DIR, "host_graph_metadata.pkl")

    data = torch.load(graph_path, weights_only=False)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"✓ Loaded graph:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Node features: {data.num_node_features}")
    print(f"  Classes: {metadata['n_classes']}")

    # Select target labels
    if TASK == 'binary':
        data.y = data.y_binary
        num_classes = 2
        class_labels = ['Benign', 'Attack']
        print(f"✓ Task: Binary classification")
    else:
        num_classes = metadata['n_classes']
        class_labels = metadata['labels']
        print(f"✓ Task: Multi-class classification ({num_classes} classes)")

    # Make graph undirected (optional - many GNNs work better with undirected graphs)
    print("\nMaking graph undirected...")
    data.edge_index = to_undirected(data.edge_index)
    print(f"✓ Edges after making undirected: {data.edge_index.shape[1]}")

    # Add self-loops (optional - helps with node's own features)
    print("Adding self-loops...")
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    print(f"✓ Edges after adding self-loops: {data.edge_index.shape[1]}")

    # Normalize node features
    print("Normalizing node features...")
    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True) + 1e-8
    data.x = (data.x - mean) / std
    print(f"✓ Features normalized (mean≈0, std≈1)")

    # Create train/val/test masks
    print(f"\nCreating masks (train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO})...")
    train_mask, val_mask, test_mask = create_masks(
        data.num_nodes, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    print(f"✓ Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # Move to device
    data = data.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)

    print(f"\n✓ Using device: {DEVICE}")

    # Create model
    print(f"\nCreating {MODEL_TYPE} model...")
    if MODEL_TYPE == 'GCN':
        model = HostGCN(
            in_channels=data.num_node_features,
            hidden_channels=HIDDEN_DIM,
            out_channels=num_classes,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
    elif MODEL_TYPE == 'GAT':
        model = HostGAT(
            in_channels=data.num_node_features,
            hidden_channels=HIDDEN_DIM,
            out_channels=num_classes,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            heads=8
        )
    elif MODEL_TYPE == 'GraphSAGE':
        model = HostGraphSAGE(
            in_channels=data.num_node_features,
            hidden_channels=HIDDEN_DIM,
            out_channels=num_classes,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    model = model.to(DEVICE)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    print("\n" + "=" * 80)
    print(f"TRAINING FOR {EPOCHS} EPOCHS")
    print("=" * 80)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': []
    }

    best_val_f1 = 0
    best_epoch = 0

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):
        # Train
        train_loss = train_epoch(model, data, optimizer, train_mask)

        # Evaluate
        train_loss, train_acc, train_prec, train_rec, train_f1, _, _ = evaluate(model, data, train_mask)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, data, val_mask)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_precision'].append(train_prec)
        history['val_precision'].append(val_prec)
        history['train_recall'].append(train_rec)
        history['val_recall'].append(val_rec)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            model_path = os.path.join(MODEL_DIR, f"host_gnn_{MODEL_TYPE.lower()}_{TASK}_best.pt")
            torch.save(model.state_dict(), model_path)

        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}/{EPOCHS}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    print(f"\n✓ Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    print("\n" + "=" * 80)
    print("TESTING BEST MODEL")
    print("=" * 80)

    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, data, test_mask)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall: {test_rec:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

    # Save results
    print("\nSaving results...")

    # Plot training history
    history_plot_path = os.path.join(RESULTS_DIR, f"host_gnn_{MODEL_TYPE.lower()}_{TASK}_history.png")
    plot_training_history(history, history_plot_path)

    # Plot confusion matrix
    cm_plot_path = os.path.join(RESULTS_DIR, f"host_gnn_{MODEL_TYPE.lower()}_{TASK}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_labels, cm_plot_path)

    # Save metrics
    results = {
        'model_type': MODEL_TYPE,
        'task': TASK,
        'epochs': EPOCHS,
        'best_epoch': best_epoch,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'history': history,
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }

    results_path = os.path.join(RESULTS_DIR, f"host_gnn_{MODEL_TYPE.lower()}_{TASK}_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to: {results_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

