"""
GraphSAGE Training for Flow-based GNN
Binary Classification: Benign vs Attack
Optimized for Kaggle Notebook with GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv, BatchNorm
import numpy as np
import os
import json
import time
import shutil
import pickle
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
GRAPH_DIR = "/kaggle/working/graph_flow"
OUTPUT_DIR = "/kaggle/working/output_flow"
MODEL_DIR = "/kaggle/working/output_flow/models"
RESULTS_DIR = "/kaggle/working/output_flow/results"

HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 4096

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# MODEL
# ============================================================================
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_embeddings(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return x


# ============================================================================
# TRAINER
# ============================================================================
class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0
        self.best_epoch = 0

    def train_epoch(self, data, optimizer):
        self.model.train()
        data = data.to(self.device)

        optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        pred = out[data.train_mask].argmax(dim=1)
        acc = (pred == data.y[data.train_mask]).float().mean().item()

        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, data, mask):
        self.model.eval()
        data = data.to(self.device)

        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask]).item()

        pred = out[mask].argmax(dim=1)
        probs = F.softmax(out[mask], dim=1)

        acc = (pred == data.y[mask]).float().mean().item()

        return loss, acc, pred.cpu().numpy(), data.y[mask].cpu().numpy(), probs.cpu().numpy()

    def train(self, data, optimizer, scheduler, num_epochs, patience):
        print(f"\nTraining GraphSAGE on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("-" * 50)

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(data, optimizer)
            val_loss, val_acc, _, _, _ = self.evaluate(data, data.val_mask)

            if scheduler:
                scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                self.save_model(os.path.join(MODEL_DIR, "best_model.pt"))
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\nBest Val Acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        return self.history

    def test(self, data):
        print("\nTesting...")

        start_time = time.time()
        test_loss, test_acc, pred, true, probs = self.evaluate(data, data.test_mask)
        latency = ((time.time() - start_time) * 1000) / len(true)

        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)

        try:
            auc = roc_auc_score(true, probs[:, 1])
        except:
            auc = 0.0

        cm = confusion_matrix(true, pred)

        results = {
            'accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'latency_ms': float(latency),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist(),
            'n_samples': len(true)
        }

        print(f"Accuracy:  {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print(f"Latency:   {latency:.4f} ms/sample")

        return results

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Attack'])
    ax.set_yticklabels(['Benign', 'Attack'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color=color, fontsize=12)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metrics(results, save_path):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1_score'], results['auc']]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results(results, history, output_dir):
    """Save all results."""
    results['timestamp'] = datetime.now().isoformat()
    results['history'] = history

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def create_zip():
    """Create zip file for download."""
    zip_path = "/kaggle/working/gnn_flow_output"
    shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)
    print(f"Created: {zip_path}.zip")
    return f"{zip_path}.zip"


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("GRAPHSAGE TRAINING (Flow-based)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load graph
    print("\nLoading graph...")
    graph_path = os.path.join(GRAPH_DIR, "graph.pt")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    data = torch.load(graph_path, weights_only=False)
    print(f"Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}, Features: {data.num_features}")

    # Create model
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=2,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # Train
    trainer = Trainer(model, DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = trainer.train(data, optimizer, scheduler, NUM_EPOCHS, PATIENCE)

    # Test
    trainer.load_model(os.path.join(MODEL_DIR, "best_model.pt"))
    results = trainer.test(data)

    # Save results
    plot_training_history(history, os.path.join(RESULTS_DIR, "training_history.png"))
    plot_confusion_matrix(np.array(results['confusion_matrix']), os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_metrics(results, os.path.join(RESULTS_DIR, "metrics.png"))
    save_results(results, history, RESULTS_DIR)

    # Create zip
    zip_path = create_zip()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()

