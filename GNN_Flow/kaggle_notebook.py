"""
Complete Pipeline for Flow-based GraphSAGE Training on Kaggle
Dataset: CICIDS2018 (excluding Thuesday-20-02-2018)
Binary Classification: Benign vs Attack

Run this entire script in a Kaggle notebook with GPU enabled.
Dataset should be uploaded to Kaggle as "cicids2018-csv"
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import json
import os
import gc
import time
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = "/kaggle/input/cicids2018-csv"
OUTPUT_DIR = "/kaggle/working/output"
EXCLUDED_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

# Preprocessing
SAMPLE_SIZE = None
COLS_TO_DROP = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port',
                'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count']

# Graph
K_NEIGHBORS = 5
MAX_GRAPH_SAMPLES = 500000

# Model
HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
PATIENCE = 15

# Split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_STATE = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================
def preprocess_data():
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    csv_files = sorted(Path(DATA_DIR).glob("*_TrafficForML_CICFlowMeter.csv"))
    csv_files = [f for f in csv_files if f.name != EXCLUDED_FILE]
    print(f"Found {len(csv_files)} files")

    dfs = []
    for f in csv_files:
        print(f"  Loading {f.name}...", end=" ")
        try:
            df = pd.read_csv(f, low_memory=False, encoding='utf-8')
        except:
            df = pd.read_csv(f, low_memory=False, encoding='latin-1')
        if 'Label' in df.columns:
            df = df[df['Label'] != 'Label'].copy()
        print(f"{len(df):,} rows")
        dfs.append(df)
        gc.collect()

    data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"Total: {len(data):,} rows")

    # Drop columns
    drops = [c for c in COLS_TO_DROP if c in data.columns]
    data = data.drop(columns=drops)

    # Convert to numeric
    for col in data.columns:
        if col != 'Label':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle missing/inf
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Remove duplicates
    data = data.drop_duplicates()
    gc.collect()

    # Create labels
    data['binary_label'] = (data['Label'] != 'Benign').astype(int)
    print(f"Benign: {(data['binary_label']==0).sum():,}, Attack: {(data['binary_label']==1).sum():,}")

    # Extract features
    feature_cols = [c for c in data.columns if c not in ['Label', 'binary_label']]
    feature_cols = [c for c in feature_cols if data[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    variances = data[feature_cols].var()
    feature_cols = variances[variances > 0].index.tolist()
    print(f"Features: {len(feature_cols)}")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(data[feature_cols])
    y = data['binary_label'].values

    del data
    gc.collect()

    return X, y, feature_cols, scaler

# ============================================================================
# 2. GRAPH CONSTRUCTION
# ============================================================================
def build_graph(X, y, max_samples=MAX_GRAPH_SAMPLES):
    print("\n" + "=" * 60)
    print("STEP 2: GRAPH CONSTRUCTION")
    print("=" * 60)

    # Sample if needed
    if len(X) > max_samples:
        print(f"Sampling {max_samples:,} from {len(X):,}")
        indices = np.arange(len(X))
        _, sample_idx = train_test_split(indices, test_size=max_samples, stratify=y, random_state=RANDOM_STATE)
        X = X[sample_idx]
        y = y[sample_idx]
        gc.collect()

    print(f"Samples: {len(X):,}")

    # Build KNN with FAISS
    print(f"Building KNN graph (k={K_NEIGHBORS})...")
    try:
        import faiss
        X_f32 = X.astype(np.float32)
        faiss.normalize_L2(X_f32)

        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, X.shape[1])
            print("  Using FAISS GPU")
        except:
            index = faiss.IndexFlatIP(X.shape[1])
            print("  Using FAISS CPU")

        index.add(X_f32)
        _, indices = index.search(X_f32, K_NEIGHBORS + 1)

        edges_src, edges_dst = [], []
        for i in range(len(X)):
            for j in indices[i][1:]:
                if j >= 0:
                    edges_src.append(i)
                    edges_dst.append(j)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        del X_f32, indices
        gc.collect()

    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        print("  Using sklearn (FAISS not available)")
        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(X)
        _, indices = nbrs.kneighbors(X)

        edges_src, edges_dst = [], []
        for i in range(len(X)):
            for j in indices[i][1:]:
                edges_src.append(i)
                edges_dst.append(j)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        del indices
        gc.collect()

    print(f"Edges: {edge_index.shape[1]:,}")

    # Create masks
    n = len(X)
    perm = np.random.permutation(n)
    train_size = int(n * TRAIN_RATIO)
    val_size = int(n * VAL_RATIO)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True

    print(f"Train: {train_mask.sum().item():,}, Val: {val_mask.sum().item():,}, Test: {test_mask.sum().item():,}")

    # Create data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data

# ============================================================================
# 3. MODEL DEFINITION
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

# ============================================================================
# 4. TRAINING
# ============================================================================
def train_model(data):
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=2,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    data = data.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc = (out[data.train_mask].argmax(1) == data.y[data.train_mask]).float().mean().item()

        # Val
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
            val_acc = (out[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean().item()

        scheduler.step(val_loss)

        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_state)

    return model, history

# ============================================================================
# 5. EVALUATION
# ============================================================================
def evaluate_model(model, data):
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATION")
    print("=" * 60)

    model.eval()
    data = data.to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[data.test_mask], dim=1).cpu().numpy()
        preds = out[data.test_mask].argmax(1).cpu().numpy()

    latency = ((time.time() - start_time) * 1000) / data.test_mask.sum().item()
    true = data.y[data.test_mask].cpu().numpy()

    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, zero_division=0)
    rec = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)
    try:
        auc = roc_auc_score(true, probs[:, 1])
    except:
        auc = 0.0

    cm = confusion_matrix(true, preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Latency:   {latency:.4f} ms/sample")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'auc': auc, 'latency_ms': latency,
        'confusion_matrix': cm.tolist()
    }

# ============================================================================
# 6. SAVE & VISUALIZE
# ============================================================================
def save_results(model, history, results, scaler, feature_cols):
    print("\n" + "=" * 60)
    print("STEP 5: SAVING RESULTS")
    print("=" * 60)

    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_channels': len(feature_cols),
        'hidden_channels': HIDDEN_CHANNELS,
        'num_layers': NUM_LAYERS
    }, f"{OUTPUT_DIR}/models/best_model.pt")

    # Save preprocessors
    with open(f"{OUTPUT_DIR}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)

    # Save results
    results['timestamp'] = datetime.now().isoformat()
    results['history'] = history
    with open(f"{OUTPUT_DIR}/results/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/training_history.png", dpi=150)
    plt.close()

    # Plot confusion matrix
    cm = np.array(results['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Attack']); ax.set_yticklabels(['Benign', 'Attack'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                   color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/confusion_matrix.png", dpi=150)
    plt.close()

    # Plot metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1_score'], results['auc']]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f'])
    ax.set_ylim(0, 1); ax.set_ylabel('Score'); ax.set_title('Model Performance')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/metrics.png", dpi=150)
    plt.close()

    # Create zip
    shutil.make_archive("/kaggle/working/gnn_flow_output", 'zip', OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR}")
    print("Created: /kaggle/working/gnn_flow_output.zip")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 60)
    print("GRAPHSAGE TRAINING - FLOW-BASED")
    print("CICIDS2018 Binary Classification")
    print("=" * 60 + "\n")

    X, y, feature_cols, scaler = preprocess_data()
    data = build_graph(X, y)
    del X, y
    gc.collect()

    model, history = train_model(data)
    results = evaluate_model(model, data)
    save_results(model, history, results, scaler, feature_cols)

    print("\n" + "=" * 60)
    print("COMPLETED")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"AUC:      {results['auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

