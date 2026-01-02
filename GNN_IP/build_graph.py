"""
Graph Construction for E-GraphSAGE (Edge Classification)
Nodes: IP:Port endpoints (source and destination)
Edges: Communication flows between endpoints
Edge Features: Flow features
Edge Labels: Benign/Attack

Node Definition: Each node represents a unique IP:Port combination,
providing finer granularity than IP-only nodes.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = "/kaggle/working/processed_ip"
OUTPUT_DIR = "/kaggle/working/graph_ip"
RANDOM_STATE = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Anti-leakage settings (applied to endpoints = IP:Port)
ANTI_LEAKAGE_ENABLED = True
ANTI_LEAKAGE_SCOPE = "src_endpoint_only"  # all_endpoints or src_endpoint_only


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


def build_endpoint_graph(X, y, src_idx, dst_idx):
    """
    Build endpoint-based graph for E-GraphSAGE.
    
    Graph structure:
    - Nodes: endpoints (IP:Port pairs for finer granularity)
    - Edges: flow records
    - Edge features: flow features
    - Edge labels: benign/attack
    """
    print("\nBuilding Endpoint Graph for E-GraphSAGE...")
    print("Nodes: IP:Port endpoints")
    
    with tqdm(total=5, desc="Graph construction", ncols=100) as pbar:
        # Step 1: Apply anti-leakage endpoint mapping if enabled
        if ANTI_LEAKAGE_ENABLED:
            rng = np.random.RandomState(RANDOM_STATE)
            
            if ANTI_LEAKAGE_SCOPE == "all_endpoints":
                unique_endpoints = np.unique(np.concatenate([src_idx, dst_idx]))
            else:  # src_endpoint_only
                unique_endpoints = np.unique(src_idx)
            
            endpoint_map = {old: new for new, old in enumerate(unique_endpoints)}
            
            # Remap indices
            new_src_idx = np.array([endpoint_map.get(i, i) for i in src_idx])
            if ANTI_LEAKAGE_SCOPE == "all_endpoints":
                new_dst_idx = np.array([endpoint_map.get(i, i) for i in dst_idx])
            else:
                # For dst, we need to create new indices for endpoints not in src
                max_idx = len(unique_endpoints)
                dst_unique = np.unique(dst_idx)
                for dst_ep in dst_unique:
                    if dst_ep not in endpoint_map:
                        endpoint_map[dst_ep] = max_idx
                        max_idx += 1
                new_dst_idx = np.array([endpoint_map[i] for i in dst_idx])
            
            src_idx = new_src_idx
            dst_idx = new_dst_idx
            pbar.update(1)
            pbar.set_postfix_str("Applied anti-leakage mapping")
        else:
            pbar.update(1)
            pbar.set_postfix_str("Skipped anti-leakage")
        
        # Step 2: Build endpoint mapping
        unique_endpoints = np.unique(np.concatenate([src_idx, dst_idx]))
        endpoint_to_idx = {ep: idx for idx, ep in enumerate(unique_endpoints)}
        
        src_indices = np.array([endpoint_to_idx[ep] for ep in src_idx])
        dst_indices = np.array([endpoint_to_idx[ep] for ep in dst_idx])
        
        num_nodes = len(endpoint_to_idx)
        num_edges = len(X)
        pbar.update(1)
        pbar.set_postfix_str("Built endpoint mapping")
        
        # Step 3: Build edge_index
        edge_index = torch.tensor(
            np.stack([src_indices, dst_indices], axis=0),
            dtype=torch.long
        )
        pbar.update(1)
        pbar.set_postfix_str("Built edge index")
        
        # Step 4: Extract edge features
        edge_attr = torch.tensor(X, dtype=torch.float)
        pbar.update(1)
        pbar.set_postfix_str("Extracted edge features")
        
        # Step 5: Create edge labels
        edge_y = torch.tensor(y, dtype=torch.long)
        pbar.update(1)
        pbar.set_postfix_str("Created edge labels")
    
    # Node features: ones vector (as per E-GraphSAGE paper)
    num_edge_features = edge_attr.shape[1]
    node_x = torch.ones((num_nodes, num_edge_features), dtype=torch.float)
    
    # Create PyG Data
    data = Data(
        x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
        num_nodes=num_nodes
    )
    
    # Statistics
    benign = (edge_y == 0).sum().item()
    attack = (edge_y == 1).sum().item()
    
    print(f"\nEndpoint graph built:")
    print(f"  Nodes (IP:Port endpoints): {num_nodes:,}")
    print(f"  Edges (flows):             {num_edges:,}")
    print(f"  Edge features:             {num_edge_features}")
    print(f"  Benign edges:              {benign:,} ({benign/num_edges*100:.1f}%)")
    print(f"  Attack edges:              {attack:,} ({attack/num_edges*100:.1f}%)")
    
    return data


def create_edge_splits(num_edges, edge_labels):
    """Create train/val/test splits for edges."""
    print("\nCreating edge splits...")
    
    indices = np.arange(num_edges)
    labels = edge_labels.numpy()
    
    # Stratified split
    train_idx, temp_idx = train_test_split(
        indices, test_size=1-TRAIN_RATIO, stratify=labels, random_state=RANDOM_STATE
    )
    val_ratio_adjusted = VAL_RATIO / (1 - TRAIN_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1-val_ratio_adjusted, stratify=labels[temp_idx], random_state=RANDOM_STATE
    )
    
    print(f"  Train edges: {len(train_idx):,}")
    print(f"  Val edges:   {len(val_idx):,}")
    print(f"  Test edges:  {len(test_idx):,}")
    
    return train_idx, val_idx, test_idx


def save_graph(data, train_idx, val_idx, test_idx):
    """Save graph data and split indices."""
    print(f"\nSaving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save graph
    torch.save(data, os.path.join(OUTPUT_DIR, "graph.pt"))
    
    # Save split indices
    split_indices = {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    with open(os.path.join(OUTPUT_DIR, "split_indices.pkl"), 'wb') as f:
        pickle.dump(split_indices, f)
    
    # Save metadata
    benign = (data.edge_y == 0).sum().item()
    attack = (data.edge_y == 1).sum().item()
    
    metadata = {
        'n_nodes': data.num_nodes,
        'n_edges': data.edge_index.shape[1],
        'n_edge_features': data.edge_attr.shape[1],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'n_benign': benign,
        'n_attack': attack,
        'anti_leakage': ANTI_LEAKAGE_ENABLED,
        'anti_leakage_scope': ANTI_LEAKAGE_SCOPE
    }
    
    with open(os.path.join(OUTPUT_DIR, "graph_metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print("  Saved: graph.pt, split_indices.pkl, graph_metadata.pkl")
    return metadata


def main():
    """Main graph building pipeline."""
    print("=" * 60)
    print("GRAPH CONSTRUCTION (E-GraphSAGE Edge Classification)")
    print("Nodes: IP:Port endpoints")
    print("=" * 60)
    print(f"Anti-leakage: {ANTI_LEAKAGE_ENABLED} ({ANTI_LEAKAGE_SCOPE})")
    
    X, y, src_idx, dst_idx, metadata = load_processed_data()
    
    # Build graph
    data = build_endpoint_graph(X, y, src_idx, dst_idx)
    
    # Create edge splits
    train_idx, val_idx, test_idx = create_edge_splits(data.edge_index.shape[1], data.edge_y)
    
    # Save
    graph_metadata = save_graph(data, train_idx, val_idx, test_idx)
    
    del X, y
    gc.collect()
    
    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION COMPLETED")
    print(f"Nodes (IP:Port endpoints): {graph_metadata['n_nodes']:,}")
    print(f"Edges (Flows):             {graph_metadata['n_edges']:,}")
    print(f"Edge Features:             {graph_metadata['n_edge_features']}")
    print(f"Benign edges:              {graph_metadata['n_benign']:,}")
    print(f"Attack edges:              {graph_metadata['n_attack']:,}")
    print("=" * 60)
    
    return data


if __name__ == "__main__":
    data = main()

