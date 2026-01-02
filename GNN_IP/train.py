"""
E-GraphSAGE Training for IP-based GNN
Nodes: IP addresses (endpoints), Edges: Flows
Edge Classification: Benign Flow vs Attack Flow
Optimized for Kaggle Notebook with GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import json
import time
import shutil
import pickle
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
GRAPH_DIR = "/kaggle/working/graph_ip"
OUTPUT_DIR = "/kaggle/working/output_ip"
MODEL_DIR = "/kaggle/working/output_ip/models"
RESULTS_DIR = "/kaggle/working/output_ip/results"

HIDDEN_CHANNELS = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
AGGR = "mean"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# MODEL: E-GraphSAGE for Edge Classification
# ============================================================================
class EdgeFeatureSAGEConv(nn.Module):
    """SAGEConv layer that incorporates edge features during aggregation."""
    
    def __init__(self, in_dim, out_dim, in_edge_dim, aggr="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_edge_dim = in_edge_dim
        self.aggr = aggr
        
        # Transform self node features
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        # Transform aggregated neighbor edge features
        self.lin_neigh = nn.Linear(out_dim, out_dim, bias=False)
        # Transform edge features
        self.lin_edge = nn.Linear(in_edge_dim, out_dim, bias=False)
        # Final linear layer after concatenation (W_k in paper)
        self.lin_final = nn.Linear(2 * out_dim, out_dim, bias=True)
        
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Transform self node features
        out_self = self.lin_self(x)
        
        # Transform and aggregate edge features from neighbors
        edge_projected = self.lin_edge(edge_attr)
        aggregated = torch.zeros(num_nodes, self.out_dim, device=x.device)
        
        if self.aggr == "mean":
            ones = torch.ones(edge_index.size(1), device=x.device)
            degree = torch.zeros(num_nodes, device=x.device)
            degree.scatter_add_(0, dst, ones)
            degree = degree.clamp(min=1)
            
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_projected), edge_projected)
            aggregated = aggregated / degree.unsqueeze(1)
        elif self.aggr == "sum":
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_projected), edge_projected)
        
        out_neigh = self.lin_neigh(aggregated)
        
        # Concatenate self and neighbor features (following paper's Eq. 2)
        h_combined = torch.cat([out_self, out_neigh], dim=1)
        
        # Apply final linear transformation (W_k in paper)
        out = self.lin_final(h_combined)
        
        return out


class EGraphSAGE(nn.Module):
    """E-GraphSAGE for edge classification.
    
    Architecture:
    - K layers of EdgeFeatureSAGEConv
    - Edge representation: concat(z_u, z_v)
    - Edge classifier: Linear(2*hidden_dim -> num_classes)
    """
    
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.2, aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.in_edge_dim = in_dim
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(EdgeFeatureSAGEConv(in_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(EdgeFeatureSAGEConv(hidden_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, edge_attr, edge_label_index=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        if edge_label_index is None:
            edge_label_index = edge_index
        
        src_emb = h[edge_label_index[0]]
        dst_emb = h[edge_label_index[1]]
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        
        return self.edge_classifier(edge_emb)


# ============================================================================
# TRAINER for Edge Classification
# ============================================================================
class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
        self.best_val_f1 = 0
        self.best_epoch = 0
        self.best_threshold = 0.5

    def train_epoch(self, data, train_idx, optimizer, criterion):
        self.model.train()
        data = data.to(self.device)
        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=self.device)

        optimizer.zero_grad()
        logits = self.model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[train_idx_t], data.edge_y[train_idx_t])
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, edge_indices, criterion, threshold=0.5):
        self.model.eval()
        data = data.to(self.device)
        edge_idx_t = torch.tensor(edge_indices, dtype=torch.long, device=self.device)

        logits = self.model(data.x, data.edge_index, data.edge_attr)
        logits_eval = logits[edge_idx_t]
        edge_labels = data.edge_y[edge_idx_t]
        loss = criterion(logits_eval, edge_labels.long()).item()

        probs = F.softmax(logits_eval, dim=1)[:, 1].cpu().numpy()
        pred = (probs >= threshold).astype(int)
        true = edge_labels.cpu().numpy()

        metrics = self._compute_metrics(true, pred, probs)
        return loss, metrics, pred, true, probs

    def _compute_metrics(self, y_true, y_pred, y_probs=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['far'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        if y_probs is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_probs)
                metrics['auc'] = auc(fpr, tpr)
                metrics['ap'] = average_precision_score(y_true, y_probs)
            except:
                metrics['auc'] = 0.0
                metrics['ap'] = 0.0
        
        return metrics

    def tune_threshold(self, data, val_idx):
        """Find optimal threshold to maximize F1 score."""
        self.model.eval()
        data = data.to(self.device)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, data.edge_attr)
            probs = F.softmax(logits[val_idx_t], dim=1)[:, 1].cpu().numpy()
            true = data.edge_y[val_idx_t].cpu().numpy()

        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.01, 0.99, 99):
            pred = (probs >= t).astype(int)
            f1 = f1_score(true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return best_t

    def train(self, data, train_idx, val_idx, optimizer, scheduler, criterion, num_epochs, patience):
        print(f"\nTraining E-GraphSAGE (Edge Classification) on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Training edges: {len(train_idx):,}, Validation edges: {len(val_idx):,}")
        print("-" * 60)

        patience_counter = 0
        epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch", ncols=100)

        for epoch in epoch_pbar:
            train_loss = self.train_epoch(data, train_idx, optimizer, criterion)
            val_loss, val_metrics, _, _, _ = self.evaluate(data, val_idx, criterion)

            if scheduler:
                scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            epoch_pbar.set_postfix({
                'loss': f"{train_loss:.4f}",
                'val_f1': f"{val_metrics['f1']:.4f}",
                'val_acc': f"{val_metrics['accuracy']:.4f}"
            })

            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                patience_counter = 0
                self.save_model(os.path.join(MODEL_DIR, "best_model.pt"))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nBest Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        
        # Tune threshold
        print("\nTuning decision threshold...")
        self.load_model(os.path.join(MODEL_DIR, "best_model.pt"))
        self.best_threshold = self.tune_threshold(data, val_idx)
        print(f"Optimal threshold: {self.best_threshold:.4f}")
        
        return self.history

    def test(self, data, test_idx, criterion):
        print("\nTesting on test set...")

        start_time = time.time()
        test_loss, metrics, pred, true, probs = self.evaluate(
            data, test_idx, criterion, threshold=self.best_threshold
        )
        inference_time = time.time() - start_time
        latency = (inference_time * 1000) / len(true)

        results = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'auc': float(metrics.get('auc', 0)),
            'ap': float(metrics.get('ap', 0)),
            'far': float(metrics['far']),
            'latency_ms': float(latency),
            'test_loss': float(test_loss),
            'threshold': float(self.best_threshold),
            'confusion_matrix': [[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
            'n_samples': len(true)
        }

        print(f"\nTest Results (threshold={self.best_threshold:.4f}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics.get('auc', 0):.4f}")
        print(f"  FAR:       {metrics['far']:.4f}")
        print(f"  Latency:   {latency:.4f} ms/sample")

        return results, true, pred, probs

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'best_threshold': self.best_threshold,
            'history': self.history
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_f1 = checkpoint.get('best_val_f1', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.best_threshold = checkpoint.get('best_threshold', 0.5)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 Score
    axes[1].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Accuracy
    axes[2].plot(epochs, history['val_acc'], 'purple', label='Val Acc', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Validation Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, save_path, normalized=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if normalized:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix - E-GraphSAGE'
    else:
        cm_plot = cm
        fmt = 'd'
        title = 'Confusion Matrix - E-GraphSAGE'

    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Greens',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'], ax=ax,
                annot_kws={'size': 14})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#4CAF50', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#4CAF50')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - E-GraphSAGE', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_probs, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    
    ax.plot(recall_vals, precision_vals, color='#FF9800', lw=2.5, label=f'PR (AP = {ap:.4f})')
    ax.fill_between(recall_vals, precision_vals, alpha=0.2, color='#FF9800')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - E-GraphSAGE', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metrics(results, save_path):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'AP']
    values = [results['accuracy'], results['precision'], results['recall'], 
              results['f1_score'], results['auc'], results.get('ap', 0)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1976D2', '#388E3C', '#FBC02D', '#D32F2F', '#7B1FA2', '#00796B']
    bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics - E-GraphSAGE (Edge Classification)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results(results, history, output_dir, y_true=None, y_pred=None):
    results['timestamp'] = datetime.now().isoformat()
    results['history'] = history

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save classification report
    if y_true is not None and y_pred is not None:
        report = classification_report(y_true, y_pred, target_names=['Benign', 'Attack'], digits=4)
        with open(os.path.join(output_dir, "classification_report.txt"), 'w') as f:
            f.write("E-GraphSAGE Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)

    print(f"\nResults saved to {output_dir}")


def create_zip():
    zip_path = "/kaggle/working/gnn_ip_output"
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
    print("E-GRAPHSAGE TRAINING (Edge Classification)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load graph
    print("\nLoading graph...")
    graph_path = os.path.join(GRAPH_DIR, "graph.pt")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    data = torch.load(graph_path, weights_only=False)
    
    # Load split indices
    split_path = os.path.join(GRAPH_DIR, "split_indices.pkl")
    with open(split_path, 'rb') as f:
        split_indices = pickle.load(f)
    
    train_idx = split_indices['train_idx']
    val_idx = split_indices['val_idx']
    test_idx = split_indices['test_idx']
    
    print(f"Nodes (IPs): {data.num_nodes:,}")
    print(f"Edges (Flows): {data.edge_index.shape[1]:,}")
    print(f"Edge features: {data.edge_attr.shape[1]}")
    print(f"Train/Val/Test edges: {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
    
    # Class distribution
    train_labels = data.edge_y[train_idx]
    pos = (train_labels == 1).sum().item()
    neg = (train_labels == 0).sum().item()
    print(f"Class distribution: Benign={neg:,} ({neg/(neg+pos)*100:.1f}%), Attack={pos:,} ({pos/(neg+pos)*100:.1f}%)")

    # Calculate class weights
    total = pos + neg
    class_weights = torch.tensor([total / (2 * neg), total / (2 * pos)], device=DEVICE)
    print(f"Class weights: [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")

    # Create model
    model = EGraphSAGE(
        in_dim=data.edge_attr.shape[1],
        hidden_dim=HIDDEN_CHANNELS,
        num_classes=2,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        aggr=AGGR
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Train
    trainer = Trainer(model, DEVICE)
    history = trainer.train(data, train_idx, val_idx, optimizer, scheduler, criterion, NUM_EPOCHS, PATIENCE)

    # Test
    trainer.load_model(os.path.join(MODEL_DIR, "best_model.pt"))
    results, y_true, y_pred, y_probs = trainer.test(data, test_idx, criterion)

    # Save plots
    plot_training_history(history, os.path.join(RESULTS_DIR, "training_history.png"))
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png"), normalized=True)
    plot_roc_curve(y_true, y_probs, os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_pr_curve(y_true, y_probs, os.path.join(RESULTS_DIR, "pr_curve.png"))
    plot_metrics(results, os.path.join(RESULTS_DIR, "metrics.png"))
    save_results(results, history, RESULTS_DIR, y_true, y_pred)

    # Create zip
    zip_path = create_zip()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"FAR:       {results['far']:.4f}")
    print(f"Threshold: {results['threshold']:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()

