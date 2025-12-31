"""Build endpoint-based graph for E-GraphSAGE.

Graph structure:
- Nodes: endpoints (IP or IP:Port)
- Edges: flow records
- Edge features: flow features
- Edge labels: benign/attack
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple
from torch_geometric.data import Data
from tqdm import tqdm

from . import config as cfg


def create_endpoint_graph(
    df: pd.DataFrame,
    feature_cols: list,
    src_ip_col: str = None,
    dst_ip_col: str = None,
    label_col: str = None,
    mapping_mode: str = None,
    anti_leakage: bool = None
) -> Tuple[Data, int]:
    """Create PyG Data object for endpoint-based graph.
    
    Args:
        df: DataFrame with flow records
        feature_cols: List of feature column names
        src_ip_col: Source IP column (default from config)
        dst_ip_col: Destination IP column (default from config)
        label_col: Label column (default from config)
        mapping_mode: "ip_port" or "ip_only" (default from config)
        anti_leakage: Enable IP random mapping (default from config)
        
    Returns:
        data: PyG Data object
        num_nodes: Number of nodes
    """
    # Defaults from config
    if src_ip_col is None:
        src_ip_col = cfg.SRC_IP_COL
    if dst_ip_col is None:
        dst_ip_col = cfg.DST_IP_COL
    if label_col is None:
        label_col = cfg.LABEL_COL
    if mapping_mode is None:
        mapping_mode = cfg.MAPPING_MODE
    if anti_leakage is None:
        anti_leakage = cfg.ANTI_LEAKAGE_ENABLED
    
    print(f"\n🔨 Building Endpoint Graph (E-GraphSAGE)...")
    print(f"   Mapping mode: {mapping_mode}")
    print(f"   Anti-leakage: {anti_leakage}")
    
    # Build endpoint mapping with progress
    with tqdm(total=4, desc="📍 Graph construction", ncols=100) as pbar:
        # Step 1: Build endpoint mapping
        endpoint_to_idx, src_indices, dst_indices = _build_endpoint_mapping(
            df, src_ip_col, dst_ip_col, mapping_mode, anti_leakage
        )
        pbar.update(1)
        pbar.set_postfix_str("Built endpoint mapping")
        
        num_nodes = len(endpoint_to_idx)
        num_edges = len(df)
        
        # Step 2: Build edge_index
        edge_index = torch.tensor(
            np.stack([src_indices, dst_indices], axis=0),
            dtype=torch.long
        )
        pbar.update(1)
        pbar.set_postfix_str("Built edge index")
        
        # Step 3: Extract edge features
        edge_attr = df[feature_cols].values.astype(np.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        pbar.update(1)
        pbar.set_postfix_str("Extracted edge features")
        
        # Step 4: Create edge labels
        edge_y = (df[label_col] != 0).astype(int).values
        edge_y = torch.tensor(edge_y, dtype=torch.long)
        pbar.update(1)
        pbar.set_postfix_str("Created edge labels")
    
    # Node features: ones vector (as per E-GraphSAGE)
    num_edge_features = edge_attr.shape[1]
    x = torch.ones((num_nodes, num_edge_features), dtype=torch.float)
    
    # Create PyG Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
        num_nodes=num_nodes
    )
    
    # Statistics
    benign = (edge_y == 0).sum().item()
    attack = (edge_y == 1).sum().item()
    
    print(f"\n✅ Endpoint graph built:")
    print(f"   Nodes (endpoints): {num_nodes:,}")
    print(f"   Edges (flows):     {num_edges:,}")
    print(f"   Edge features:     {num_edge_features}")
    print(f"   Benign edges:      {benign:,} ({benign/num_edges*100:.1f}%)")
    print(f"   Attack edges:      {attack:,} ({attack/num_edges*100:.1f}%)")
    
    return data, num_nodes


def _build_endpoint_mapping(
    df, src_ip_col, dst_ip_col, mapping_mode, anti_leakage
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Build endpoint mapping and edge indices."""
    
    # Extract IP columns
    src_ips = df[src_ip_col].astype(str).values
    dst_ips = df[dst_ip_col].astype(str).values
    
    # Apply IP random mapping if enabled (anti-leakage)
    if anti_leakage:
        rng = np.random.RandomState(cfg.SEED)
        
        if cfg.ANTI_LEAKAGE_SCOPE == "all_ips":
            unique_ips = np.unique(np.concatenate([src_ips, dst_ips]))
        else:  # src_ip_only
            unique_ips = np.unique(src_ips)
        
        ip_map = {ip: f"IP_{i:06d}" for i, ip in enumerate(unique_ips)}
        
        src_ips = np.array([ip_map.get(ip, ip) for ip in src_ips])
        if cfg.ANTI_LEAKAGE_SCOPE == "all_ips":
            dst_ips = np.array([ip_map.get(ip, ip) for ip in dst_ips])
    
    # Build endpoint strings
    if mapping_mode == "ip_port":
        src_port_col = cfg.SRC_PORT_COL
        dst_port_col = cfg.DST_PORT_COL
        src_ports = df[src_port_col].astype(str).values
        dst_ports = df[dst_port_col].astype(str).values
        src_endpoints = [f"{ip}:{port}" for ip, port in zip(src_ips, src_ports)]
        dst_endpoints = [f"{ip}:{port}" for ip, port in zip(dst_ips, dst_ports)]
    else:  # ip_only
        src_endpoints = src_ips.tolist()
        dst_endpoints = dst_ips.tolist()
    
    # Create mapping
    unique_endpoints = sorted(set(src_endpoints + dst_endpoints))
    endpoint_to_idx = {ep: idx for idx, ep in enumerate(unique_endpoints)}
    
    # Convert to indices
    src_indices = np.array([endpoint_to_idx[ep] for ep in src_endpoints], dtype=np.int64)
    dst_indices = np.array([endpoint_to_idx[ep] for ep in dst_endpoints], dtype=np.int64)
    
    return endpoint_to_idx, src_indices, dst_indices
