"""
Graph Construction for IP-based GNN
Nodes: IP addresses
Edges: Communication flows between IPs
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle
import os
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = "/kaggle/working/processed_ip"
OUTPUT_DIR = "/kaggle/working/graph_ip"
MAX_SAMPLES = 2000000  # Limit samples for memory
RANDOM_STATE = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def load_processed_data():
    """Load preprocessed data."""
    print("Loading processed data...")

    X = np.load(os.path.join(INPUT_DIR, "X.npy"))
    y = np.load(os.path.join(INPUT_DIR, "y.npy"))
    src_idx = np.load(os.path.join(INPUT_DIR, "src_idx.npy"))
    dst_idx = np.load(os.path.join(INPUT_DIR, "dst_idx.npy"))

    with open(os.path.join(INPUT_DIR, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    print(f"  Flows: {len(X):,}, Features: {X.shape[1]}")
    print(f"  Unique IPs: {metadata['n_ips']:,}")

    return X, y, src_idx, dst_idx, metadata


def sample_data(X, y, src_idx, dst_idx, max_samples):
    """Sample data if too large."""
    if len(X) <= max_samples:
        return X, y, src_idx, dst_idx

    print(f"Sampling {max_samples:,} from {len(X):,}...")

    indices = np.arange(len(X))
    _, sample_idx = train_test_split(indices, test_size=max_samples, stratify=y, random_state=RANDOM_STATE)

    return X[sample_idx], y[sample_idx], src_idx[sample_idx], dst_idx[sample_idx]


def build_ip_graph(X, y, src_idx, dst_idx, n_ips):
    """
    Build IP-based graph.
    Nodes: IP addresses
    Edges: Flows between IPs
    Node features: Aggregated flow features per IP
    Node labels: Majority vote of flow labels
    """
    print("\nBuilding IP graph...")

    # Aggregate features per IP
    print("  Aggregating features per IP...")
    ip_features = defaultdict(list)
    ip_labels = defaultdict(list)

    for i in range(len(X)):
        src = src_idx[i]
        dst = dst_idx[i]

        # Add flow features to both src and dst IPs
        ip_features[src].append(X[i])
        ip_features[dst].append(X[i])
        ip_labels[src].append(y[i])
        ip_labels[dst].append(y[i])

    # Create node features and labels
    print("  Creating node features...")
    n_features = X.shape[1]
    node_features = np.zeros((n_ips, n_features * 3))  # mean, std, max
    node_labels = np.zeros(n_ips, dtype=np.int64)

    for ip in range(n_ips):
        if ip in ip_features:
            feats = np.array(ip_features[ip])
            node_features[ip, :n_features] = feats.mean(axis=0)
            node_features[ip, n_features:n_features*2] = feats.std(axis=0)
            node_features[ip, n_features*2:] = feats.max(axis=0)

            # Label: 1 if any attack flow, else 0
            labels = ip_labels[ip]
            node_labels[ip] = 1 if sum(labels) > 0 else 0

    # Handle NaN/Inf
    node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    del ip_features, ip_labels
    gc.collect()

    # Create edges (unique flow pairs)
    print("  Creating edges...")
    edge_set = set()
    for i in range(len(src_idx)):
        src = src_idx[i]
        dst = dst_idx[i]
        if src != dst:
            edge_set.add((src, dst))
            edge_set.add((dst, src))  # Undirected

    edges = list(edge_set)
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)

    print(f"  Nodes: {n_ips:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Features per node: {node_features.shape[1]}")

    return node_features, node_labels, edge_index


def create_data_splits(n_nodes, node_labels):
    """Create train/val/test masks."""
    print("\nCreating data splits...")

    # Filter nodes with at least one flow
    valid_mask = (node_labels >= 0)  # All nodes are valid

    indices = np.arange(n_nodes)

    # Stratified split
    train_idx, temp_idx = train_test_split(
        indices, test_size=1-TRAIN_RATIO, stratify=node_labels, random_state=RANDOM_STATE
    )
    val_ratio_adjusted = VAL_RATIO / (1 - TRAIN_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1-val_ratio_adjusted, stratify=node_labels[temp_idx], random_state=RANDOM_STATE
    )

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"  Train: {train_mask.sum().item():,}")
    print(f"  Val: {val_mask.sum().item():,}")
    print(f"  Test: {test_mask.sum().item():,}")

    return train_mask, val_mask, test_mask


def save_graph(node_features, node_labels, edge_index, train_mask, val_mask, test_mask):
    """Save graph data."""
    print(f"\nSaving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(node_labels, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    torch.save(data, os.path.join(OUTPUT_DIR, "graph.pt"))

    metadata = {
        'n_nodes': data.num_nodes,
        'n_edges': data.num_edges,
        'n_features': data.num_features,
        'n_train': int(train_mask.sum()),
        'n_val': int(val_mask.sum()),
        'n_test': int(test_mask.sum()),
        'n_benign': int((node_labels == 0).sum()),
        'n_attack': int((node_labels == 1).sum())
    }

    with open(os.path.join(OUTPUT_DIR, "graph_metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    print("  Saved: graph.pt, graph_metadata.pkl")
    return data, metadata


def main():
    """Main graph building pipeline."""
    print("=" * 60)
    print("GRAPH CONSTRUCTION (IP-based)")
    print("=" * 60)

    X, y, src_idx, dst_idx, metadata = load_processed_data()
    X, y, src_idx, dst_idx = sample_data(X, y, src_idx, dst_idx, MAX_SAMPLES)

    # Reindex IPs
    unique_ips = np.unique(np.concatenate([src_idx, dst_idx]))
    ip_map = {old: new for new, old in enumerate(unique_ips)}
    src_idx = np.array([ip_map[i] for i in src_idx])
    dst_idx = np.array([ip_map[i] for i in dst_idx])
    n_ips = len(unique_ips)

    print(f"Active IPs: {n_ips:,}")

    node_features, node_labels, edge_index = build_ip_graph(X, y, src_idx, dst_idx, n_ips)
    train_mask, val_mask, test_mask = create_data_splits(n_ips, node_labels)
    data, graph_metadata = save_graph(node_features, node_labels, edge_index, train_mask, val_mask, test_mask)

    del X, y
    gc.collect()

    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION COMPLETED")
    print(f"Nodes (IPs): {graph_metadata['n_nodes']:,}")
    print(f"Edges: {graph_metadata['n_edges']:,}")
    print(f"Benign IPs: {graph_metadata['n_benign']:,}")
    print(f"Attack IPs: {graph_metadata['n_attack']:,}")
    print("=" * 60)

    return data


if __name__ == "__main__":
    data = main()

