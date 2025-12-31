"""
Complete Pipeline for IP-based GraphSAGE Training on Kaggle
Dataset: CICIDS2018 - Thuesday-20-02-2018 (contains IP info)
Binary Classification: Benign vs Attack

Run this entire script in a Kaggle notebook with GPU enabled.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import json
import os
import gc
import time
import shutil
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = "/kaggle/input/cicids2018-csv"
OUTPUT_DIR = "/kaggle/working/output"
TARGET_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

# Preprocessing
SAMPLE_SIZE = 1000000
COLS_TO_DROP = ['Timestamp', 'Flow ID', 'Src Port',
                'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count']

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

    file_path = os.path.join(DATA_DIR, TARGET_FILE)
    print(f"Loading: {TARGET_FILE}...")

    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')

    if 'Label' in df.columns:
        df = df[df['Label'] != 'Label'].copy()

    print(f"Loaded: {len(df):,} rows")

    # Sample if needed
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE:,}...")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        gc.collect()

    # Keep IP data
    ip_data = df[['Src IP', 'Dst IP']].copy()

    # Drop columns
    drops = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drops)

    # Convert to numeric
    non_numeric = ['Label', 'Src IP', 'Dst IP']
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing/inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Create labels
    df['binary_label'] = (df['Label'] != 'Benign').astype(int)
    print(f"Benign: {(df['binary_label']==0).sum():,}, Attack: {(df['binary_label']==1).sum():,}")

    # Extract features
    exclude = ['Label', 'binary_label', 'Src IP', 'Dst IP']
    feature_cols = [c for c in df.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    variances = df[feature_cols].var()
    feature_cols = variances[variances > 0].index.tolist()
    print(f"Features: {len(feature_cols)}")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df['binary_label'].values

    # Create IP indices
    all_ips = pd.concat([ip_data['Src IP'], ip_data['Dst IP']]).unique()
    ip_encoder = LabelEncoder()
    ip_encoder.fit(all_ips)

    src_idx = ip_encoder.transform(ip_data['Src IP'].values)
    dst_idx = ip_encoder.transform(ip_data['Dst IP'].values)

    print(f"Unique IPs: {len(all_ips):,}")

    del df, ip_data
    gc.collect()

    return X, y, src_idx, dst_idx, len(all_ips), feature_cols, scaler

# ============================================================================
# 2. GRAPH CONSTRUCTION
# ============================================================================
def build_ip_graph(X, y, src_idx, dst_idx, n_ips):
    print("\n" + "=" * 60)
    print("STEP 2: GRAPH CONSTRUCTION")
    print("=" * 60)

    n_features = X.shape[1]

    # Aggregate features per IP
    print("Aggregating features per IP...")
    ip_features = defaultdict(list)
    ip_labels = defaultdict(list)

    for i in range(len(X)):
        ip_features[src_idx[i]].append(X[i])
        ip_features[dst_idx[i]].append(X[i])
        ip_labels[src_idx[i]].append(y[i])
        ip_labels[dst_idx[i]].append(y[i])

    # Create node features (mean, std, max)
    node_features = np.zeros((n_ips, n_features * 3))
    node_labels = np.zeros(n_ips, dtype=np.int64)

    for ip in range(n_ips):
        if ip in ip_features:
            feats = np.array(ip_features[ip])
            node_features[ip, :n_features] = feats.mean(axis=0)
            node_features[ip, n_features:n_features*2] = feats.std(axis=0)
            node_features[ip, n_features*2:] = feats.max(axis=0)
            node_labels[ip] = 1 if sum(ip_labels[ip]) > 0 else 0

    node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    del ip_features, ip_labels
    gc.collect()

    # Create edges
    print("Creating edges...")
    edge_set = set()
    for i in range(len(src_idx)):
        if src_idx[i] != dst_idx[i]:
            edge_set.add((src_idx[i], dst_idx[i]))
            edge_set.add((dst_idx[i], src_idx[i]))

    edges = list(edge_set)
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)

    print(f"Nodes (IPs): {n_ips:,}")
    print(f"Edges: {edge_index.shape[1]:,}")
    print(f"Benign IPs: {(node_labels==0).sum():,}, Attack IPs: {(node_labels==1).sum():,}")

    # Create masks
    perm = np.random.permutation(n_ips)
    train_size = int(n_ips * TRAIN_RATIO)
    val_size = int(n_ips * VAL_RATIO)

    train_mask = torch.zeros(n_ips, dtype=torch.bool)
    val_mask = torch.zeros(n_ips, dtype=torch.bool)
    test_mask = torch.zeros(n_ips, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True

    print(f"Train: {train_mask.sum().item():,}, Val: {val_mask.sum().item():,}, Test: {test_mask.sum().item():,}")

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(node_labels, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data

# ============================================================================
# 3. MODEL
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
# 6. SAVE
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
        'in_channels': len(feature_cols) * 3,
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

    # Plot history
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
    ax.set_ylim(0, 1); ax.set_ylabel('Score'); ax.set_title('Model Performance (IP-based)')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/metrics.png", dpi=150)
    plt.close()

    # Create zip
    shutil.make_archive("/kaggle/working/gnn_ip_output", 'zip', OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR}")
    print("Created: /kaggle/working/gnn_ip_output.zip")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 60)
    print("GRAPHSAGE TRAINING - IP-BASED")
    print("CICIDS2018 Binary Classification")
    print("=" * 60 + "\n")

    X, y, src_idx, dst_idx, n_ips, feature_cols, scaler = preprocess_data()
    data = build_ip_graph(X, y, src_idx, dst_idx, n_ips)
    del X, y, src_idx, dst_idx
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

