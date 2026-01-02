"""
Complete Pipeline for E-GraphSAGE Training on Kaggle
Dataset: CICIDS2018 - Thuesday-20-02-2018 (contains IP info)
Edge Classification: Benign Flow vs Attack Flow

Run this entire script in a Kaggle notebook with GPU enabled.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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

# Column names
SRC_IP_COL = "Src IP"
DST_IP_COL = "Dst IP"
SRC_PORT_COL = "Src Port"
DST_PORT_COL = "Dst Port"
LABEL_COL = "Label"

# Preprocessing
SAMPLE_SIZE = None
# Note: Src Port and Dst Port are kept for node definition (IP:Port)
COLS_TO_DROP = ['Timestamp', 'Flow ID',
                'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count']

# Anti-leakage (applied to endpoints = IP:Port)
ANTI_LEAKAGE_ENABLED = True
ANTI_LEAKAGE_SCOPE = "src_endpoint_only"  # all_endpoints or src_endpoint_only

# Model
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
AGGR = "mean"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10

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

    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL] != 'Label'].copy()

    print(f"Loaded: {len(df):,} rows")

    # Sample if needed
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE:,}...")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        gc.collect()

    # Keep IP and Port data for endpoint (IP:Port) definition
    endpoint_data = df[[SRC_IP_COL, DST_IP_COL, SRC_PORT_COL, DST_PORT_COL]].copy()

    # Drop columns
    drops = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drops)

    # Convert to numeric (exclude label, IP, and Port columns)
    non_numeric = [LABEL_COL, SRC_IP_COL, DST_IP_COL, SRC_PORT_COL, DST_PORT_COL]
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing/inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Create labels
    df['binary_label'] = (df[LABEL_COL] != 'Benign').astype(int)
    benign = (df['binary_label'] == 0).sum()
    attack = (df['binary_label'] == 1).sum()
    print(f"Benign flows: {benign:,} ({benign/len(df)*100:.1f}%)")
    print(f"Attack flows: {attack:,} ({attack/len(df)*100:.1f}%)")

    # Extract features (exclude Port columns from features - they are used for node definition)
    exclude = [LABEL_COL, 'binary_label', SRC_IP_COL, DST_IP_COL, SRC_PORT_COL, DST_PORT_COL]
    feature_cols = [c for c in df.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    variances = df[feature_cols].var()
    feature_cols = variances[variances > 0].index.tolist()
    print(f"Features: {len(feature_cols)}")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df['binary_label'].values

    # Create endpoint (IP:Port) indices - nodes are IP:Port pairs
    src_endpoints = endpoint_data[SRC_IP_COL].astype(str) + ":" + endpoint_data[SRC_PORT_COL].astype(str)
    dst_endpoints = endpoint_data[DST_IP_COL].astype(str) + ":" + endpoint_data[DST_PORT_COL].astype(str)
    
    all_endpoints = pd.concat([src_endpoints, dst_endpoints]).unique()
    endpoint_encoder = LabelEncoder()
    endpoint_encoder.fit(all_endpoints)

    src_idx = endpoint_encoder.transform(src_endpoints.values)
    dst_idx = endpoint_encoder.transform(dst_endpoints.values)

    print(f"Unique endpoints (IP:Port): {len(all_endpoints):,}")

    del df, endpoint_data
    gc.collect()

    return X, y, src_idx, dst_idx, feature_cols, scaler

# ============================================================================
# 2. GRAPH CONSTRUCTION (Endpoint-based for E-GraphSAGE)
# Nodes are defined as IP:Port pairs for finer granularity
# ============================================================================
def build_endpoint_graph(X, y, src_idx, dst_idx):
    print("\n" + "=" * 60)
    print("STEP 2: GRAPH CONSTRUCTION (E-GraphSAGE)")
    print("Nodes: IP:Port endpoints")
    print("=" * 60)
    print(f"Anti-leakage: {ANTI_LEAKAGE_ENABLED} ({ANTI_LEAKAGE_SCOPE})")

    # Apply anti-leakage endpoint mapping if enabled
    if ANTI_LEAKAGE_ENABLED:
        if ANTI_LEAKAGE_SCOPE == "all_endpoints":
            unique_endpoints = np.unique(np.concatenate([src_idx, dst_idx]))
        else:  # src_endpoint_only
            unique_endpoints = np.unique(src_idx)
        
        endpoint_map = {old: new for new, old in enumerate(unique_endpoints)}
        
        new_src_idx = np.array([endpoint_map.get(i, i) for i in src_idx])
        if ANTI_LEAKAGE_SCOPE == "all_endpoints":
            new_dst_idx = np.array([endpoint_map.get(i, i) for i in dst_idx])
        else:
            max_idx = len(unique_endpoints)
            dst_unique = np.unique(dst_idx)
            for dst_ep in dst_unique:
                if dst_ep not in endpoint_map:
                    endpoint_map[dst_ep] = max_idx
                    max_idx += 1
            new_dst_idx = np.array([endpoint_map[i] for i in dst_idx])
        
        src_idx = new_src_idx
        dst_idx = new_dst_idx

    # Build endpoint mapping
    unique_endpoints = np.unique(np.concatenate([src_idx, dst_idx]))
    endpoint_to_idx = {ep: idx for idx, ep in enumerate(unique_endpoints)}
    
    src_indices = np.array([endpoint_to_idx[ep] for ep in src_idx])
    dst_indices = np.array([endpoint_to_idx[ep] for ep in dst_idx])
    
    num_nodes = len(endpoint_to_idx)
    num_edges = len(X)
    n_features = X.shape[1]

    # Build edge_index
    edge_index = torch.tensor(np.stack([src_indices, dst_indices], axis=0), dtype=torch.long)
    
    # Edge features
    edge_attr = torch.tensor(X, dtype=torch.float)
    
    # Edge labels
    edge_y = torch.tensor(y, dtype=torch.long)
    
    # Node features (ones vector as per E-GraphSAGE)
    node_x = torch.ones((num_nodes, n_features), dtype=torch.float)

    # Create edge splits
    print("Creating edge splits...")
    indices = np.arange(num_edges)
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=1-TRAIN_RATIO, stratify=y, random_state=RANDOM_STATE
    )
    val_ratio_adjusted = VAL_RATIO / (1 - TRAIN_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1-val_ratio_adjusted, stratify=y[temp_idx], random_state=RANDOM_STATE
    )

    # Statistics
    benign = (edge_y == 0).sum().item()
    attack = (edge_y == 1).sum().item()
    
    print(f"Nodes (IP:Port endpoints): {num_nodes:,}")
    print(f"Edges (flows):             {num_edges:,}")
    print(f"Edge features:             {n_features}")
    print(f"Benign edges:              {benign:,} ({benign/num_edges*100:.1f}%)")
    print(f"Attack edges:              {attack:,} ({attack/num_edges*100:.1f}%)")
    print(f"Train/Val/Test:            {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")

    data = Data(
        x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
        num_nodes=num_nodes
    )

    return data, train_idx, val_idx, test_idx


# ============================================================================
# 3. MODEL: E-GraphSAGE for Edge Classification
# ============================================================================
class EdgeFeatureSAGEConv(nn.Module):
    """SAGEConv layer that incorporates edge features during aggregation.
    
    Following Eq. 2 of E-GraphSAGE paper:
    h_v^k = σ(W_k · CONCAT(h_v^{k-1}, AGG({h_e : e ∈ N(v)})))
    Uses concatenation instead of addition for combining self and neighbor info.
    """
    
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
    """E-GraphSAGE for edge classification."""
    
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.2, aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(EdgeFeatureSAGEConv(in_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(EdgeFeatureSAGEConv(hidden_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
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
# 4. TRAINING (Edge Classification)
# ============================================================================
def train_model(data, train_idx, val_idx):
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING (Edge Classification)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Calculate class weights
    train_labels = data.edge_y[train_idx]
    pos = (train_labels == 1).sum().item()
    neg = (train_labels == 0).sum().item()
    total = pos + neg
    class_weights = torch.tensor([total / (2 * neg), total / (2 * pos)], device=DEVICE)
    print(f"Class weights: [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")

    model = EGraphSAGE(
        in_dim=data.edge_attr.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        aggr=AGGR
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    data = data.to(DEVICE)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
    best_val_f1 = 0
    best_state = None
    patience_counter = 0

    epoch_pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training", unit="epoch", ncols=100)

    for epoch in epoch_pbar:
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[train_idx_t], data.edge_y[train_idx_t])
        loss.backward()
        optimizer.step()

        # Val
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_attr)
            val_loss = criterion(logits[val_idx_t], data.edge_y[val_idx_t]).item()
            
            probs = F.softmax(logits[val_idx_t], dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            true = data.edge_y[val_idx_t].cpu().numpy()
            
            val_acc = accuracy_score(true, preds)
            val_f1 = f1_score(true, preds, zero_division=0)

        scheduler.step(val_loss)

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)

        epoch_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'val_f1': f"{val_f1:.4f}",
            'val_acc': f"{val_acc:.4f}"
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest Val F1: {best_val_f1:.4f}")
    model.load_state_dict(best_state)

    # Tune threshold
    print("\nTuning decision threshold...")
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits[val_idx_t], dim=1)[:, 1].cpu().numpy()
        true = data.edge_y[val_idx_t].cpu().numpy()

    best_threshold, best_t_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (probs >= t).astype(int)
        t_f1 = f1_score(true, pred, zero_division=0)
        if t_f1 > best_t_f1:
            best_t_f1 = t_f1
            best_threshold = t

    print(f"Optimal threshold: {best_threshold:.4f}")

    return model, history, best_threshold

# ============================================================================
# 5. EVALUATION (Edge Classification)
# ============================================================================
def evaluate_model(model, data, test_idx, threshold):
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATION (Edge Classification)")
    print("=" * 60)

    model.eval()
    data = data.to(DEVICE)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits[test_idx_t], dim=1)[:, 1].cpu().numpy()
        preds = (probs >= threshold).astype(int)

    inference_time = time.time() - start_time
    latency = (inference_time * 1000) / len(test_idx)
    true = data.edge_y[test_idx_t].cpu().numpy()

    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, zero_division=0)
    rec = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)
    
    try:
        fpr, tpr, _ = roc_curve(true, probs)
        auc_score = auc(fpr, tpr)
        ap = average_precision_score(true, probs)
    except:
        auc_score = 0.0
        ap = 0.0

    cm = confusion_matrix(true, preds)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\nTest Results (threshold={threshold:.4f}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc_score:.4f}")
    print(f"  AP:        {ap:.4f}")
    print(f"  FAR:       {far:.4f}")
    print(f"  Latency:   {latency:.4f} ms/flow")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'auc': auc_score, 'ap': ap, 'far': far,
        'latency_ms': latency, 'threshold': threshold,
        'confusion_matrix': cm.tolist()
    }, true, preds, probs

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
def save_results(model, history, results, scaler, feature_cols, y_true, y_pred, y_probs):
    print("\n" + "=" * 60)
    print("STEP 5: SAVING RESULTS")
    print("=" * 60)

    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_dim': len(feature_cols),
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'best_threshold': results['threshold']
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

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Attack'], digits=4)
    with open(f"{OUTPUT_DIR}/results/classification_report.txt", 'w') as f:
        f.write("E-GraphSAGE Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Score'); axes[1].set_title('Validation F1')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['val_acc'], 'purple', label='Val Acc', linewidth=2)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy'); axes[2].set_title('Validation Accuracy')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/training_history.png", dpi=150)
    plt.close()

    # Plot confusion matrix
    cm = np.array(results['confusion_matrix'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                annot_kws={'size': 14})
    axes[0].set_title('Confusion Matrix - E-GraphSAGE', fontweight='bold')
    axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                annot_kws={'size': 14}, vmin=0, vmax=1)
    axes[1].set_title('Normalized Confusion Matrix', fontweight='bold')
    axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/confusion_matrix.png", dpi=150)
    plt.close()

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='#4CAF50', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#4CAF50')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - E-GraphSAGE', fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/roc_curve.png", dpi=150)
    plt.close()

    # Plot PR curve
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    ax.plot(recall_vals, precision_vals, color='#FF9800', lw=2.5, label=f'PR (AP = {ap:.4f})')
    ax.fill_between(recall_vals, precision_vals, alpha=0.2, color='#FF9800')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - E-GraphSAGE', fontweight='bold')
    ax.legend(loc='lower left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results/pr_curve.png", dpi=150)
    plt.close()

    # Plot metrics bar
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'AP']
    values = [results['accuracy'], results['precision'], results['recall'], 
              results['f1_score'], results['auc'], results['ap']]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1976D2', '#388E3C', '#FBC02D', '#D32F2F', '#7B1FA2', '#00796B']
    bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title('Performance Metrics - E-GraphSAGE (Edge Classification)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
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
    print("E-GRAPHSAGE TRAINING - EDGE CLASSIFICATION")
    print("CICIDS2018 Binary Classification (Benign vs Attack Flow)")
    print("=" * 60 + "\n")

    X, y, src_idx, dst_idx, feature_cols, scaler = preprocess_data()
    data, train_idx, val_idx, test_idx = build_endpoint_graph(X, y, src_idx, dst_idx)
    
    del X, y, src_idx, dst_idx
    gc.collect()

    model, history, threshold = train_model(data, train_idx, val_idx)
    results, y_true, y_pred, y_probs = evaluate_model(model, data, test_idx, threshold)
    save_results(model, history, results, scaler, feature_cols, y_true, y_pred, y_probs)

    print("\n" + "=" * 60)
    print("COMPLETED")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"FAR:       {results['far']:.4f}")
    print(f"Threshold: {results['threshold']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

