"""
Graph Construction for Flow-based GNN using FAISS
Each flow is a node, connected via K-Nearest Neighbors
Optimized for Kaggle Notebook with GPU support
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import pickle
import os
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = "/kaggle/working/processed_flow"
OUTPUT_DIR = "/kaggle/working/graph_flow"
K_NEIGHBORS = 5
MAX_SAMPLES = 500000  # Limit for memory efficiency
RANDOM_STATE = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_processed_data():
    """Load preprocessed data."""
    print("Loading processed data...")
    X = np.load(os.path.join(INPUT_DIR, "X.npy"))
    y = np.load(os.path.join(INPUT_DIR, "y.npy"))
    print(f"  Loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    return X, y


def sample_data(X, y, max_samples):
    """Sample data if too large."""
    if len(X) <= max_samples:
        return X, y

    print(f"Sampling {max_samples:,} from {len(X):,} samples...")

    # Stratified sampling
    indices = np.arange(len(X))
    _, sample_idx = train_test_split(
        indices, test_size=max_samples, stratify=y, random_state=RANDOM_STATE
    )

    X_sampled = X[sample_idx]
    y_sampled = y[sample_idx]

    print(f"  Sampled: {len(X_sampled):,} samples")
    print(f"  Benign: {(y_sampled == 0).sum():,}, Attack: {(y_sampled == 1).sum():,}")
    return X_sampled, y_sampled


def build_knn_graph_faiss(X, k):
    """Build KNN graph using FAISS for efficiency."""
    print(f"\nBuilding KNN graph (k={k}) using FAISS...")

    try:
        import faiss

        n_samples, n_features = X.shape
        X_float32 = X.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(X_float32)

        # Use GPU if available
        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, n_features)
            print("  Using FAISS GPU")
        except:
            index = faiss.IndexFlatIP(n_features)
            print("  Using FAISS CPU")

        index.add(X_float32)

        # Search k+1 neighbors (first is self)
        print("  Searching neighbors...")
        _, indices = index.search(X_float32, k + 1)

        # Build edge list
        print("  Building edge list...")
        edges_src = []
        edges_dst = []

        for i in range(n_samples):
            for j in indices[i][1:]:  # Skip self
                if j >= 0:
                    edges_src.append(i)
                    edges_dst.append(j)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        del X_float32, indices
        gc.collect()

    except ImportError:
        print("  FAISS not available, using sklearn...")
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        _, indices = nbrs.kneighbors(X)

        edges_src = []
        edges_dst = []

        for i in range(len(X)):
            for j in indices[i][1:]:
                edges_src.append(i)
                edges_dst.append(j)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        del indices
        gc.collect()

    print(f"  Nodes: {len(X):,}, Edges: {edge_index.shape[1]:,}")
    return edge_index


def create_data_splits(n_samples):
    """Create train/val/test masks."""
    print("\nCreating data splits...")

    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * TRAIN_RATIO)
    val_size = int(n_samples * VAL_RATIO)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"  Train: {train_mask.sum().item():,}")
    print(f"  Val: {val_mask.sum().item():,}")
    print(f"  Test: {test_mask.sum().item():,}")

    return train_mask, val_mask, test_mask


def create_graph_data(X, y, edge_index, train_mask, val_mask, test_mask):
    """Create PyTorch Geometric Data object."""
    print("\nCreating graph data object...")

    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print(f"  Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    return data


def save_graph(data, X, y):
    """Save graph data."""
    print(f"\nSaving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(data, os.path.join(OUTPUT_DIR, "graph.pt"))

    metadata = {
        'n_nodes': data.num_nodes,
        'n_edges': data.num_edges,
        'n_features': data.num_features,
        'n_train': int(data.train_mask.sum()),
        'n_val': int(data.val_mask.sum()),
        'n_test': int(data.test_mask.sum()),
        'n_benign': int((y == 0).sum()),
        'n_attack': int((y == 1).sum()),
        'k_neighbors': K_NEIGHBORS
    }

    with open(os.path.join(OUTPUT_DIR, "graph_metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    print("  Saved: graph.pt, graph_metadata.pkl")
    return metadata


def main():
    """Main graph building pipeline."""
    print("=" * 60)
    print("GRAPH CONSTRUCTION (Flow-based)")
    print("=" * 60)

    X, y = load_processed_data()
    X, y = sample_data(X, y, MAX_SAMPLES)

    edge_index = build_knn_graph_faiss(X, K_NEIGHBORS)
    train_mask, val_mask, test_mask = create_data_splits(len(X))
    data = create_graph_data(X, y, edge_index, train_mask, val_mask, test_mask)

    del X, edge_index
    gc.collect()

    metadata = save_graph(data, data.x.numpy(), y)

    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION COMPLETED")
    print(f"Nodes: {metadata['n_nodes']:,}")
    print(f"Edges: {metadata['n_edges']:,}")
    print("=" * 60)

    return data


if __name__ == "__main__":
    data = main()

