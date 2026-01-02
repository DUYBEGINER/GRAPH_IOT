# %%capture
# !pip install torch-geometric pandas numpy scikit-learn tqdm matplotlib seaborn pandas

"""
E-GraphSAGE (Edge Classification)"
Dataset: CICIDS2018 - Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
"""

import os
import gc
import time
import json
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, classification_report
)

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "/kaggle/input/cicids2018"
OUTPUT_DIR = "/kaggle/working/output"
TARGET_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

SRC_IP_COL = "Src IP"
DST_IP_COL = "Dst IP"
SRC_PORT_COL = "Src Port"
DST_PORT_COL = "Dst Port"
LABEL_COL = "Label"

SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42

COLS_TO_DROP = {
    "Timestamp", "Flow ID",
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd URG Flags", "CWE Flag Count",
}

SPLIT_STRATEGY = "group_src_endpoint"

ANTI_LEAKAGE_ENABLED = True
ANTI_LEAKAGE_SCOPE = "src_endpoint_only"

# Model/training
NODE_FEAT_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.2
AGGR = "mean"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 13
PATIENCE = 7

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

SAVE_PLOTS = True
MAKE_ZIP = True

SANITY_SHUFFLE_LABELS_ONCE = True   
SANITY_EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Utils: splitting
# =============================================================================
def split_edges(y: np.ndarray, src_groups: np.ndarray):
    n = len(y)
    idx = np.arange(n)

    if SPLIT_STRATEGY == "random_stratified":
        train_idx, temp_idx = train_test_split(
            idx, test_size=1 - TRAIN_RATIO, stratify=y, random_state=RANDOM_STATE
        )
        val_ratio_adj = VAL_RATIO / (1 - TRAIN_RATIO)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=1 - val_ratio_adj, stratify=y[temp_idx], random_state=RANDOM_STATE
        )
        return train_idx, val_idx, test_idx

    if SPLIT_STRATEGY == "group_src_endpoint":
        gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
        train_idx, temp_idx = next(gss1.split(idx, y, groups=src_groups))

        # split temp into val/test by groups again
        temp_groups = src_groups[temp_idx]
        gss2 = GroupShuffleSplit(
            n_splits=1,
            train_size=VAL_RATIO / (1 - TRAIN_RATIO),
            random_state=RANDOM_STATE
        )
        val_rel, test_rel = next(gss2.split(temp_idx, y[temp_idx], groups=temp_groups))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
        return train_idx, val_idx, test_idx

    raise ValueError(f"Unknown SPLIT_STRATEGY: {SPLIT_STRATEGY}")

def preprocess_data():
    print("=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)

    file_path = os.path.join(DATA_DIR, TARGET_FILE)
    print("Loading:", file_path)

    def _usecols(c: str) -> bool:
        return c not in COLS_TO_DROP

    df = pd.read_csv(file_path, low_memory=True, usecols=_usecols)

    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL] != "Label"]

    print(f"Loaded rows: {len(df):,}")

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE:,} rows ...")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # Ports compact
    df[SRC_PORT_COL] = pd.to_numeric(df[SRC_PORT_COL], errors="coerce").fillna(0).astype(np.uint16)
    df[DST_PORT_COL] = pd.to_numeric(df[DST_PORT_COL], errors="coerce").fillna(0).astype(np.uint16)

    # Binary label
    df["binary_label"] = (df[LABEL_COL] != "Benign").astype(np.int8)
    y = df["binary_label"].to_numpy(np.int64, copy=True)

    benign = int((y == 0).sum()); attack = int((y == 1).sum())
    print(f"Benign: {benign:,} ({benign/len(y)*100:.1f}%) | Attack: {attack:,} ({attack/len(y)*100:.1f}%)")

    # Features
    exclude = {LABEL_COL, "binary_label", SRC_IP_COL, DST_IP_COL, SRC_PORT_COL, DST_PORT_COL}
    feature_cols = [c for c in df.columns if c not in exclude]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = (
        df[feature_cols]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .astype(np.float32)
    )
    X_raw = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

    # Endpoint ids without strings:
    n = len(df)
    all_ips = pd.concat(
        [df[SRC_IP_COL].astype("category"), df[DST_IP_COL].astype("category")],
        ignore_index=True
    )
    ip_codes, _ = pd.factorize(all_ips, sort=True)

    src_ip_code = ip_codes[:n].astype(np.uint32, copy=False)
    dst_ip_code = ip_codes[n:].astype(np.uint32, copy=False)

    src_port = df[SRC_PORT_COL].to_numpy(np.uint16, copy=False)
    dst_port = df[DST_PORT_COL].to_numpy(np.uint16, copy=False)

    src_key = (src_ip_code.astype(np.uint64) << 16) | src_port.astype(np.uint64)
    dst_key = (dst_ip_code.astype(np.uint64) << 16) | dst_port.astype(np.uint64)

    endpoint_all = np.concatenate([src_key, dst_key])
    node_codes, uniques = pd.factorize(endpoint_all, sort=True)

    src_nodes = node_codes[:n].astype(np.int32, copy=False)
    dst_nodes = node_codes[n:].astype(np.int32, copy=False)
    num_nodes = int(len(uniques))

    print(f"Unique endpoints (IP:Port): {num_nodes:,}")
    print(f"Features: {len(feature_cols)}")

    del df, all_ips, ip_codes, endpoint_all, node_codes, uniques
    gc.collect()

    return X_raw, y, src_nodes, dst_nodes, feature_cols


# =============================================================================
# 2) BUILD GRAPH (LEAKAGE SAFE + STRONGER SPLIT)
# =============================================================================
def build_graph(X_raw, y, src_nodes, dst_nodes):
    print("\n" + "=" * 70)
    print("STEP 2: GRAPH CONSTRUCTION (LEAKAGE SAFE + STRONGER SPLIT)")
    print(f"Split strategy: {SPLIT_STRATEGY}")
    print("- Message passing edges: TRAIN only")
    print("- Classifier can use edge_attr of query edges (val/test)")
    print("=" * 70)

    # ---- split indices (possibly group split) ----
    train_idx, val_idx, test_idx = split_edges(y, src_groups=src_nodes)

    # ---- anti-leak mapping on node ids (optional) ----
    if ANTI_LEAKAGE_ENABLED:
        if ANTI_LEAKAGE_SCOPE == "all_endpoints":
            keep = np.unique(np.concatenate([src_nodes, dst_nodes]))
        else:
            keep = np.unique(src_nodes)

        max_old = int(max(src_nodes.max(), dst_nodes.max()))
        remap = np.full(max_old + 1, -1, dtype=np.int32)
        remap[keep] = np.arange(len(keep), dtype=np.int32)
        next_id = len(keep)

        if ANTI_LEAKAGE_SCOPE != "all_endpoints":
            dst_u = np.unique(dst_nodes)
            missing = dst_u[remap[dst_u] < 0]
            remap[missing] = np.arange(next_id, next_id + len(missing), dtype=np.int32)
            num_nodes = next_id + len(missing)
        else:
            num_nodes = len(keep)

        src_nodes = remap[src_nodes]
        dst_nodes = remap[dst_nodes]
    else:
        num_nodes = int(max(src_nodes.max(), dst_nodes.max())) + 1

    # ---- scale: fit on TRAIN only; transform train/val/test subsets (no full X_scaled) ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx]).astype(np.float32, copy=False)
    X_val   = scaler.transform(X_raw[val_idx]).astype(np.float32, copy=False)
    X_test  = scaler.transform(X_raw[test_idx]).astype(np.float32, copy=False)

    # free raw features to reduce RAM
    del X_raw
    gc.collect()

    # ---- tensors ----
    node_x = torch.ones((num_nodes, NODE_FEAT_DIM), dtype=torch.float32)

    # Train message passing graph
    train_edge_index = torch.from_numpy(
        np.stack([src_nodes[train_idx], dst_nodes[train_idx]], axis=0)
    ).long()
    train_edge_attr = torch.from_numpy(X_train).float()
    train_edge_y = torch.from_numpy(y[train_idx]).long()

    # Query edges + their edge_attr
    val_edge_label_index = torch.from_numpy(
        np.stack([src_nodes[val_idx], dst_nodes[val_idx]], axis=0)
    ).long()
    test_edge_label_index = torch.from_numpy(
        np.stack([src_nodes[test_idx], dst_nodes[test_idx]], axis=0)
    ).long()

    val_edge_attr = torch.from_numpy(X_val).float()
    test_edge_attr = torch.from_numpy(X_test).float()

    val_edge_y = torch.from_numpy(y[val_idx]).long()
    test_edge_y = torch.from_numpy(y[test_idx]).long()

    data = Data(
        x=node_x,
        edge_index=train_edge_index,   # TRAIN ONLY
        edge_attr=train_edge_attr,     # TRAIN ONLY
        num_nodes=num_nodes
    )

    # attach extras
    data.train_edge_y = train_edge_y

    data.val_edge_label_index = val_edge_label_index
    data.val_edge_attr = val_edge_attr
    data.val_edge_y = val_edge_y

    data.test_edge_label_index = test_edge_label_index
    data.test_edge_attr = test_edge_attr
    data.test_edge_y = test_edge_y

    print(f"Nodes: {num_nodes:,}")
    print(f"Edges total: {len(y):,}")
    print(f"Train/Val/Test edges: {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
    print(f"Train edge_attr kept: {tuple(data.edge_attr.shape)}")
    print(f"Val/Test edge_attr kept: {tuple(val_edge_attr.shape)} / {tuple(test_edge_attr.shape)}")

    # quick overlap check for group split
    if SPLIT_STRATEGY == "group_src_endpoint":
        tr_src = np.unique(src_nodes[train_idx])
        va_src = np.unique(src_nodes[val_idx])
        te_src = np.unique(src_nodes[test_idx])
        ov1 = np.intersect1d(tr_src, va_src).size
        ov2 = np.intersect1d(tr_src, te_src).size
        print(f"[Check] Overlap SRC endpoints (train∩val/train∩test) = {ov1} / {ov2} (expect 0)")

    return data, scaler, train_idx, val_idx, test_idx


# =============================================================================
# 3) MODEL: use (src_emb, dst_emb, edge_attr) in classifier
# =============================================================================
class EdgeFeatureSAGEConv(nn.Module):
    def __init__(self, node_in_dim, out_dim, edge_dim, aggr="mean"):
        super().__init__()
        self.out_dim = out_dim
        self.aggr = aggr

        self.lin_self = nn.Linear(node_in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(out_dim, out_dim, bias=False)
        self.lin_final = nn.Linear(2 * out_dim, out_dim, bias=True)

    def forward(self, x, edge_index, edge_attr):
        _, dst = edge_index
        num_nodes = x.size(0)

        out_self = self.lin_self(x)
        edge_proj = self.lin_edge(edge_attr)

        aggregated = torch.zeros(num_nodes, self.out_dim, device=x.device)

        if self.aggr == "mean":
            ones = torch.ones(edge_index.size(1), device=x.device)
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, ones)
            deg = deg.clamp(min=1)

            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_proj), edge_proj)
            aggregated = aggregated / deg.unsqueeze(1)
        elif self.aggr == "sum":
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_proj), edge_proj)
        else:
            raise ValueError(f"Unsupported aggr: {self.aggr}")

        out_neigh = self.lin_neigh(aggregated)
        h = torch.cat([out_self, out_neigh], dim=1)
        return self.lin_final(h)


class EGraphSAGE(nn.Module):
    def __init__(self, node_in_dim, edge_dim, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.2, aggr="mean"):
        super().__init__()
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(EdgeFeatureSAGEConv(node_in_dim, hidden_dim, edge_dim=edge_dim, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(EdgeFeatureSAGEConv(hidden_dim, hidden_dim, edge_dim=edge_dim, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # project edge_attr -> hidden
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # classifier on concat(src_emb, dst_emb, edge_attr_emb)
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def encode_nodes(self, x, edge_index, edge_attr):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index, edge_attr, edge_label_index, edge_label_attr):
        # message passing happens ONLY on (edge_index, edge_attr) which is TRAIN graph
        h = self.encode_nodes(x, edge_index, edge_attr)

        src_emb = h[edge_label_index[0]]
        dst_emb = h[edge_label_index[1]]
        e_emb = F.relu(self.edge_proj(edge_label_attr))

        out = self.edge_classifier(torch.cat([src_emb, dst_emb, e_emb], dim=1))
        return out


# =============================================================================
# 4) TRAINING
# =============================================================================
def train_one(model, data, train_edge_y, train_edge_label_index, train_edge_label_attr,
              val_edge_y, val_edge_label_index, val_edge_label_attr, max_epochs):
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # weights from train only
    pos = (train_edge_y == 1).sum().item()
    neg = (train_edge_y == 0).sum().item()
    total = pos + neg
    class_weights = torch.tensor([total / (2 * neg), total / (2 * pos)], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    pbar = tqdm(range(1, max_epochs + 1), desc="Training", unit="epoch", ncols=100)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        logits_train = model(
            data.x, data.edge_index, data.edge_attr,
            edge_label_index=train_edge_label_index,
            edge_label_attr=train_edge_label_attr
        )
        loss = criterion(logits_train, train_edge_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(
                data.x, data.edge_index, data.edge_attr,
                edge_label_index=val_edge_label_index,
                edge_label_attr=val_edge_label_attr
            )
            val_loss = criterion(logits_val, val_edge_y).item()

            probs = F.softmax(logits_val, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            true = val_edge_y.detach().cpu().numpy()

            val_acc = accuracy_score(true, preds)
            val_f1 = f1_score(true, preds, zero_division=0)

        scheduler.step(val_loss)

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss))
        history["val_f1"].append(float(val_f1))
        history["val_acc"].append(float(val_acc))

        pbar.set_postfix(loss=f"{loss.item():.4f}", val_f1=f"{val_f1:.4f}", val_acc=f"{val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and max_epochs == NUM_EPOCHS:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_f1


def tune_threshold(model, data, val_edge_y, val_edge_label_index, val_edge_label_attr):
    model.eval()
    with torch.no_grad():
        logits_val = model(
            data.x, data.edge_index, data.edge_attr,
            edge_label_index=val_edge_label_index,
            edge_label_attr=val_edge_label_attr
        )
        probs = F.softmax(logits_val, dim=1)[:, 1].detach().cpu().numpy()
        true = val_edge_y.detach().cpu().numpy()

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (probs >= t).astype(int)
        f1v = f1_score(true, pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_t = float(t)
    return best_t, best_f1


def train_model(data):
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING")
    print("=" * 70)
    print("Device:", DEVICE)

    data = data.to(DEVICE)

    # move query tensors to device once
    val_edge_label_index = data.val_edge_label_index.to(DEVICE)
    val_edge_attr = data.val_edge_attr.to(DEVICE)
    val_edge_y = data.val_edge_y.to(DEVICE)

    test_edge_label_index = data.test_edge_label_index.to(DEVICE)
    test_edge_attr = data.test_edge_attr.to(DEVICE)
    test_edge_y = data.test_edge_y.to(DEVICE)

    # train edges are exactly the message passing train edges
    train_edge_label_index = data.edge_index  # TRAIN edges
    train_edge_label_attr = data.edge_attr
    train_edge_y = data.train_edge_y.to(DEVICE)

    model = EGraphSAGE(
        node_in_dim=data.x.size(1),
        edge_dim=data.edge_attr.size(1),
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        aggr=AGGR
    ).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- OPTIONAL sanity: shuffled labels should DROP ----
    if SANITY_SHUFFLE_LABELS_ONCE:
        print("\n[Sanity] Training with SHUFFLED train labels for quick check ...")
        y_shuf = train_edge_y[torch.randperm(train_edge_y.numel(), device=DEVICE)]
        model_sanity = EGraphSAGE(
            node_in_dim=data.x.size(1),
            edge_dim=data.edge_attr.size(1),
            hidden_dim=HIDDEN_DIM,
            num_classes=2,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            aggr=AGGR
        ).to(DEVICE)

        model_sanity, _, best_f1 = train_one(
            model_sanity, data,
            y_shuf, train_edge_label_index, train_edge_label_attr,
            val_edge_y, val_edge_label_index, val_edge_attr,
            max_epochs=SANITY_EPOCHS
        )
        t_sanity, f1_sanity = tune_threshold(model_sanity, data, val_edge_y, val_edge_label_index, val_edge_attr)
        print(f"[Sanity] Shuffled-label best VAL F1 ~ {best_f1:.4f}, tuned F1 ~ {f1_sanity:.4f} (should NOT be ~1.0)\n")
        del model_sanity
        torch.cuda.empty_cache()

    # ---- real training ----
    model, history, best_val_f1 = train_one(
        model, data,
        train_edge_y, train_edge_label_index, train_edge_label_attr,
        val_edge_y, val_edge_label_index, val_edge_attr,
        max_epochs=NUM_EPOCHS
    )
    print(f"Best Val F1: {best_val_f1:.4f}")

    threshold, t_f1 = tune_threshold(model, data, val_edge_y, val_edge_label_index, val_edge_attr)
    print(f"Optimal threshold: {threshold:.4f} (VAL F1={t_f1:.4f})")

    pack = {
        "val_edge_label_index": val_edge_label_index,
        "val_edge_attr": val_edge_attr,
        "val_edge_y": val_edge_y,
        "test_edge_label_index": test_edge_label_index,
        "test_edge_attr": test_edge_attr,
        "test_edge_y": test_edge_y,
    }
    return model, history, threshold, pack


# =============================================================================
# 5) EVAL
# =============================================================================
def evaluate_model(model, data, threshold, pack):
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATION")
    print("=" * 70)

    model.eval()

    test_edge_label_index = pack["test_edge_label_index"]
    test_edge_attr = pack["test_edge_attr"]
    test_edge_y = pack["test_edge_y"]

    start = time.time()
    with torch.no_grad():
        logits = model(
            data.x, data.edge_index, data.edge_attr,
            edge_label_index=test_edge_label_index,
            edge_label_attr=test_edge_attr
        )
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = (probs >= threshold).astype(int)

    latency = (time.time() - start) * 1000.0 / len(preds)
    true = test_edge_y.detach().cpu().numpy()

    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, zero_division=0)
    rec = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)

    # AUC/AP safe
    try:
        fpr, tpr, _ = roc_curve(true, probs)
        auc_score = auc(fpr, tpr)
        ap = average_precision_score(true, probs)
    except Exception:
        auc_score, ap = 0.0, 0.0

    cm = confusion_matrix(true, preds)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"Test (t={threshold:.4f}) | acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc_score:.4f} ap={ap:.4f} far={far:.6f}")
    print(f"Latency: {latency:.4f} ms/edge")
    print("Confusion matrix:", cm.tolist())

    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc_score),
        "ap": float(ap),
        "far": float(far),
        "latency_ms": float(latency),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }
    return results, true, preds, probs


# =============================================================================
# 6) SAVE
# =============================================================================
def save_results(model, history, results, scaler, feature_cols, y_true, y_pred, y_probs):
    print("\n" + "=" * 70)
    print("STEP 5: SAVING RESULTS")
    print("=" * 70)

    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "node_feat_dim": NODE_FEAT_DIM,
        "edge_dim": len(feature_cols),
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "best_threshold": results["threshold"],
        "split_strategy": SPLIT_STRATEGY,
    }, f"{OUTPUT_DIR}/models/best_model.pt")

    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    results["timestamp"] = datetime.now().isoformat()
    results["history"] = history
    results["split_strategy"] = SPLIT_STRATEGY
    with open(f"{OUTPUT_DIR}/results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    report = classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=4)
    with open(f"{OUTPUT_DIR}/results/classification_report.txt", "w") as f:
        f.write("E-GraphSAGE STRICT SPLIT Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)

    if SAVE_PLOTS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2)
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["val_f1"], label="Val F1", linewidth=2)
        axes[1].set_title("Validation F1"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
        axes[2].set_title("Validation Accuracy"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Acc")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/results/training_history.png", dpi=150)
        plt.close()

        # Confusion Matrix: sklearn format is [[TN, FP], [FN, TP]]
        # Row = True label (0=Benign, 1=Attack), Col = Predicted label
        cm = np.array(results["confusion_matrix"])
        labels = ["Benign", "Attack"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap with correct orientation: True label on Y-axis, Predicted on X-axis
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2%", ax=axes[1], cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    vmin=0, vmax=1)
        axes[1].set_title("Normalized Confusion Matrix")
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("True Label")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/results/confusion_matrix.png", dpi=150)
        plt.close()
        
        # ========== AUC & Latency Summary Plot ==========
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # AUC Bar Chart
        metrics_names = ["AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
        metrics_values = [
            results["auc"], 
            results["accuracy"], 
            results["precision"], 
            results["recall"], 
            results["f1_score"]
        ]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
        
        bars = axes[0].bar(metrics_names, metrics_values, color=colors, edgecolor='black')
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Model Performance Metrics (including AUC)", fontsize=12)
        axes[0].set_ylabel("Score")
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, metrics_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Latency Display
        latency_ms = results["latency_ms"]
        axes[1].bar(["Latency"], [latency_ms], color="#FF5722", edgecolor='black', width=0.4)
        axes[1].set_title("Inference Latency", fontsize=12)
        axes[1].set_ylabel("Time (ms/edge)")
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].text(0, latency_ms + latency_ms*0.05, f'{latency_ms:.4f} ms', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/results/auc_latency_metrics.png", dpi=150)
        plt.close()

        try:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
            ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/results/roc_curve.png", dpi=150)
            plt.close()
        except Exception:
            pass

        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
            ap = average_precision_score(y_true, y_probs)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall_vals, precision_vals, lw=2, label=f"PR (AP={ap:.4f})")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curve")
            ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/results/pr_curve.png", dpi=150)
            plt.close()
        except Exception:
            pass

        # ========== Model Performance Bar Chart ==========
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        metrics_values = [
            results["accuracy"], 
            results["precision"], 
            results["recall"], 
            results["f1_score"],
            results["auc"]
        ]
        colors = ["#5DADE2", "#58D68D", "#EC7063", "#AF7AC5", "#F4D03F"]
        
        bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='none', width=0.6)
        ax.set_ylim(0, 1.0)
        ax.set_title("Model Performance", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score", fontsize=12)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/results/model_performance.png", dpi=150)
        plt.close()

    if MAKE_ZIP:
        shutil.make_archive("/kaggle/working/gnn_ip_output", "zip", OUTPUT_DIR)
        print("Created: /kaggle/working/gnn_ip_output.zip")

    print(f"Saved to: {OUTPUT_DIR}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("E-GRAPHSAGE EDGE CLASSIFICATION (STRICTER SPLIT)")
    print("=" * 70)

    X_raw, y, src_nodes, dst_nodes, feature_cols = preprocess_data()
    data, scaler, train_idx, val_idx, test_idx = build_graph(X_raw, y, src_nodes, dst_nodes)

    del y, src_nodes, dst_nodes
    gc.collect()

    model, history, threshold, pack = train_model(data)
    results, y_true, y_pred, y_probs = evaluate_model(model, data, threshold, pack)
    save_results(model, history, results, scaler, feature_cols, y_true, y_pred, y_probs)

    print("\n" + "=" * 70)
    print("COMPLETED")
    print(f"Split strategy: {SPLIT_STRATEGY}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"FAR:       {results['far']:.6f}")
    print(f"Threshold: {results['threshold']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
