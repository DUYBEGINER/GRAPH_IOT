"""Build graph structures for GNN models."""

import json
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import knn_graph
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logging.warning("PyTorch Geometric not installed. Graph building will be limited.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_graph(
    data_dir: str,
    output_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build graph based on mode (flow_gnn or ip_gnn).
    
    Args:
        data_dir: Directory with processed data (X.npy, y.npy, etc.)
        output_path: Path to save graph (.pt file)
        config: Configuration dictionary
        
    Returns:
        Metadata dictionary
    """
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "PyTorch Geometric is required for graph building. "
            "Install with: pip install torch-geometric"
        )
    
    data_path = Path(data_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("GRAPH BUILDING PROCESS")
    logger.info("="*80)
    logger.info(f"Mode: {config['mode']}")
    logger.info(f"Data directory: {data_path}")
    logger.info(f"Output: {output_file}")
    
    if config['mode'] == "flow_gnn":
        metadata = build_flow_gnn_graph(data_path, output_file, config)
    elif config['mode'] == "ip_gnn":
        metadata = build_ip_gnn_graph(data_path, output_file, config)
    else:
        raise ValueError(f"Unsupported mode for graph building: {config['mode']}")
    
    logger.info(f"\n✓ Graph saved to: {output_file}")
    logger.info("="*80 + "\n")
    
    return metadata


def build_flow_gnn_graph(
    data_path: Path,
    output_file: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build graph for flow_gnn mode using KNN.
    
    Each flow is a node. Edges connect similar flows (KNN).
    
    Args:
        data_path: Path to processed data directory
        output_file: Output file path
        config: Configuration dictionary
        
    Returns:
        Metadata dictionary
    """
    logger.info("\nBuilding flow_gnn graph...")
    knn_k = config['graph']['knn_k']
    knn_metric = config['graph']['knn_metric']
    logger.info(f"KNN parameters: k={knn_k}, metric={knn_metric}")
    
    # Load data
    logger.info("Loading data...")
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    idx_train = np.load(data_path / "idx_train.npy")
    idx_val = np.load(data_path / "idx_val.npy")
    idx_test = np.load(data_path / "idx_test.npy")
    
    logger.info(f"Loaded: X={X.shape}, y={y.shape}")
    logger.info(f"Splits: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    
    # Convert to tensors
    x = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Build KNN graph ONLY on training set to prevent data leakage
    knn_k = config['graph']['knn_k']
    knn_metric = config['graph']['knn_metric']
    logger.info(f"Building KNN graph ONLY on training set with k={knn_k}...")
    logger.info(f"This prevents information leakage from val/test sets")
    
    # Extract training features only
    x_train = x[idx_train]
    
    # Use cosine similarity if specified, otherwise euclidean
    if knn_metric == "cosine":
        # Normalize features for cosine similarity
        x_train_norm = x_train / (x_train.norm(dim=1, keepdim=True) + 1e-8)
        edge_index_train = knn_graph(x_train_norm, k=knn_k, loop=False)
    else:
        edge_index_train = knn_graph(x_train, k=knn_k, loop=False)
    
    # Map edges back to original node indices
    edge_index = idx_train[edge_index_train]
    
    logger.info(f"Edges created on training set: {edge_index.shape[1]:,}")
    logger.info(f"Val/test nodes are isolated (no edges) to prevent leakage")
    
    # Create masks
    num_nodes = len(y)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Save
    torch.save(data, output_file)
    
    # Metadata
    metadata = {
        "mode": "flow_gnn",
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "num_features": X.shape[1],
        "num_classes": 2,
        "knn_k": config['graph']['knn_k'],
        "knn_metric": config['graph']['knn_metric'],
        "train_nodes": len(idx_train),
        "val_nodes": len(idx_val),
        "test_nodes": len(idx_test),
    }
    
    logger.info("\nGraph statistics:")
    logger.info(f"  Nodes: {metadata['num_nodes']:,}")
    logger.info(f"  Edges: {metadata['num_edges']:,}")
    logger.info(f"  Features: {metadata['num_features']}")
    logger.info(f"  Avg degree: {metadata['num_edges'] / metadata['num_nodes']:.2f}")
    
    return metadata


def build_ip_gnn_graph(
    data_path: Path,
    output_file: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build graph for ip_gnn mode.
    
    Each IP is a node. Edges represent flows between IPs.
    Node features: aggregated flow features per IP.
    
    Args:
        data_path: Path to processed data directory
        output_file: Output file path
        config: Configuration dictionary
        
    Returns:
        Metadata dictionary
    """
    logger.info("\nBuilding ip_gnn graph...")
    
    # Load data
    logger.info("Loading data...")
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    idx_train = np.load(data_path / "idx_train.npy")
    idx_val = np.load(data_path / "idx_val.npy")
    idx_test = np.load(data_path / "idx_test.npy")
    
    # Load IP data
    ip_data_file = data_path / "ip_data.npz"
    if not ip_data_file.exists():
        raise FileNotFoundError(
            f"IP data not found: {ip_data_file}. "
            "Make sure to run split_scale.py with mode=ip_gnn"
        )
    
    ip_data = np.load(ip_data_file)
    src_ips = ip_data["Src_IP"]
    dst_ips = ip_data["Dst_IP"]
    
    logger.info(f"Loaded: X={X.shape}, y={y.shape}")
    logger.info(f"Loaded: {len(src_ips)} flow records with IP data")
    
    # Map IPs to node IDs
    logger.info("Mapping IPs to node IDs...")
    ip_to_id, id_to_ip = create_ip_mapping(src_ips, dst_ips)
    num_ips = len(ip_to_id)
    
    logger.info(f"Unique IPs: {num_ips:,}")
    
    # Create edge list from flows
    logger.info("Creating edge list from flows...")
    edge_list, edge_labels = create_edge_list(src_ips, dst_ips, y, ip_to_id)
    
    logger.info(f"Edges created: {len(edge_list):,}")
    
    # Aggregate node features per IP (using training data only)
    logger.info("Aggregating node features per IP...")
    node_features = aggregate_ip_features(
        X, src_ips, dst_ips, idx_train, ip_to_id, num_ips, config
    )
    
    logger.info(f"Node features: {node_features.shape}")
    
    # Create node labels (majority vote from flows)
    logger.info("Creating node labels...")
    node_labels = create_node_labels(src_ips, dst_ips, y, ip_to_id, num_ips)
    
    # Create node splits based on flow splits
    logger.info("Creating node splits...")
    node_masks = create_node_masks(
        src_ips, dst_ips, idx_train, idx_val, idx_test, ip_to_id, num_ips
    )
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    y_tensor = torch.tensor(node_labels, dtype=torch.long)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    train_mask = torch.tensor(node_masks["train"], dtype=torch.bool)
    val_mask = torch.tensor(node_masks["val"], dtype=torch.bool)
    test_mask = torch.tensor(node_masks["test"], dtype=torch.bool)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Save
    torch.save(data, output_file)
    
    # Save IP mapping
    mapping_file = output_file.parent / "ip_mapping.pkl"
    with open(mapping_file, 'wb') as f:
        pickle.dump({"ip_to_id": ip_to_id, "id_to_ip": id_to_ip}, f)
    
    logger.info(f"IP mapping saved to: {mapping_file}")
    
    # Metadata
    metadata = {
        "mode": "ip_gnn",
        "num_nodes": num_ips,
        "num_edges": edge_index.shape[1],
        "num_features": node_features.shape[1],
        "num_classes": 2,
        "train_nodes": train_mask.sum().item(),
        "val_nodes": val_mask.sum().item(),
        "test_nodes": test_mask.sum().item(),
        "aggregation_features": config['graph']['ip_agg_features'],
    }
    
    logger.info("\nGraph statistics:")
    logger.info(f"  Nodes (IPs): {metadata['num_nodes']:,}")
    logger.info(f"  Edges (flows): {metadata['num_edges']:,}")
    logger.info(f"  Features per node: {metadata['num_features']}")
    logger.info(f"  Avg degree: {metadata['num_edges'] / metadata['num_nodes']:.2f}")
    logger.info(f"  Train nodes: {metadata['train_nodes']:,}")
    logger.info(f"  Val nodes: {metadata['val_nodes']:,}")
    logger.info(f"  Test nodes: {metadata['test_nodes']:,}")
    
    return metadata


def create_ip_mapping(src_ips: np.ndarray, dst_ips: np.ndarray) -> Tuple[Dict, Dict]:
    """Create bidirectional IP <-> ID mapping."""
    unique_ips = set(src_ips) | set(dst_ips)
    
    ip_to_id = {ip: idx for idx, ip in enumerate(sorted(unique_ips))}
    id_to_ip = {idx: ip for ip, idx in ip_to_id.items()}
    
    return ip_to_id, id_to_ip


def create_edge_list(
    src_ips: np.ndarray,
    dst_ips: np.ndarray,
    labels: np.ndarray,
    ip_to_id: Dict
) -> Tuple[list, list]:
    """Create edge list from flow records."""
    edges = []
    edge_labels = []
    
    for src, dst, label in zip(src_ips, dst_ips, labels):
        src_id = ip_to_id.get(src)
        dst_id = ip_to_id.get(dst)
        
        if src_id is not None and dst_id is not None:
            edges.append([src_id, dst_id])
            edge_labels.append(label)
    
    return edges, edge_labels


def aggregate_ip_features(
    X: np.ndarray,
    src_ips: np.ndarray,
    dst_ips: np.ndarray,
    train_idx: np.ndarray,
    ip_to_id: Dict,
    num_ips: int,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Aggregate flow features per IP address.
    
    Uses only training data to avoid data leakage.
    """
    # Collect flows per IP (from training set only)
    ip_flows = defaultdict(list)
    
    for idx in train_idx:
        src = src_ips[idx]
        dst = dst_ips[idx]
        
        src_id = ip_to_id.get(src)
        dst_id = ip_to_id.get(dst)
        
        if src_id is not None:
            ip_flows[src_id].append(X[idx])
        if dst_id is not None:
            ip_flows[dst_id].append(X[idx])
    
    # Aggregate features
    num_features = X.shape[1]
    agg_methods = config['graph']['ip_agg_features']
    feature_dim = num_features * len(agg_methods)
    
    node_features = np.zeros((num_ips, feature_dim), dtype=np.float32)
    
    for ip_id, flows in ip_flows.items():
        flows_array = np.array(flows)
        
        aggregated = []
        for method in agg_methods:
            if method == "mean":
                aggregated.append(flows_array.mean(axis=0))
            elif method == "std":
                aggregated.append(flows_array.std(axis=0))
            elif method == "sum":
                aggregated.append(flows_array.sum(axis=0))
            elif method == "max":
                aggregated.append(flows_array.max(axis=0))
            elif method == "min":
                aggregated.append(flows_array.min(axis=0))
        
        node_features[ip_id] = np.concatenate(aggregated)
    
    # For IPs not seen in training, features remain zero (could also use embeddings)
    
    return node_features


def create_node_labels(
    src_ips: np.ndarray,
    dst_ips: np.ndarray,
    labels: np.ndarray,
    ip_to_id: Dict,
    num_ips: int
) -> np.ndarray:
    """
    Create node labels based on majority vote from flows.
    
    If any flow involving an IP is malicious, label it as 1 (attack).
    """
    ip_labels = np.zeros(num_ips, dtype=np.int64)
    ip_label_votes = defaultdict(list)
    
    for src, dst, label in zip(src_ips, dst_ips, labels):
        src_id = ip_to_id.get(src)
        dst_id = ip_to_id.get(dst)
        
        if src_id is not None:
            ip_label_votes[src_id].append(label)
        if dst_id is not None:
            ip_label_votes[dst_id].append(label)
    
    # Majority vote (or max - any attack label makes IP malicious)
    for ip_id, votes in ip_label_votes.items():
        ip_labels[ip_id] = int(max(votes))  # 1 if any attack, else 0
    
    return ip_labels


def create_node_masks(
    src_ips: np.ndarray,
    dst_ips: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    ip_to_id: Dict,
    num_ips: int
) -> Dict[str, np.ndarray]:
    """
    Create node masks based on which flows they appear in.
    """
    train_ips = set()
    val_ips = set()
    test_ips = set()
    
    for idx in idx_train:
        src_id = ip_to_id.get(src_ips[idx])
        dst_id = ip_to_id.get(dst_ips[idx])
        if src_id is not None:
            train_ips.add(src_id)
        if dst_id is not None:
            train_ips.add(dst_id)
    
    for idx in idx_val:
        src_id = ip_to_id.get(src_ips[idx])
        dst_id = ip_to_id.get(dst_ips[idx])
        if src_id is not None:
            val_ips.add(src_id)
        if dst_id is not None:
            val_ips.add(dst_id)
    
    for idx in idx_test:
        src_id = ip_to_id.get(src_ips[idx])
        dst_id = ip_to_id.get(dst_ips[idx])
        if src_id is not None:
            test_ips.add(src_id)
        if dst_id is not None:
            test_ips.add(dst_id)
    
    # Create masks
    train_mask = np.zeros(num_ips, dtype=bool)
    val_mask = np.zeros(num_ips, dtype=bool)
    test_mask = np.zeros(num_ips, dtype=bool)
    
    for ip_id in train_ips:
        train_mask[ip_id] = True
    for ip_id in val_ips:
        val_mask[ip_id] = True
    for ip_id in test_ips:
        test_mask[ip_id] = True
    
    return {"train": train_mask, "val": val_mask, "test": test_mask}


if __name__ == "__main__":
    import argparse
    from config import load_config
    
    parser = argparse.ArgumentParser(description="Build graph for GNN")
    parser.add_argument("--config", type=str, default="preprocess/config.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with processed data")
    parser.add_argument("--output", type=str, default=None,
                        help="Output graph file path")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set paths
    data_dir = args.data_dir if args.data_dir else config.output_dir
    
    if args.output:
        output_path = args.output
    else:
        output_path = f"{config.output_dir}/data_{config.mode}.pt"
    
    logger.info(f"Configuration loaded: mode={config.mode}")
    
    # Build graph
    try:
        metadata = build_graph(data_dir, output_path, config)
        logger.info("✓ Graph building completed successfully!")
    except Exception as e:
        logger.error(f"✗ Graph building failed: {e}")
        raise
