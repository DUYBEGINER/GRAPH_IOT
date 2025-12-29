"""Build endpoint-based graph for E-GraphSAGE.

Graph structure:
- Nodes: endpoints (IP or IP:Port)
- Edges: flow records
- Edge features: flow features
- Edge labels: benign/attack
"""

import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def create_endpoint_graph(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict
) -> Tuple[Data, int]:
    """
    Create PyG Data object for endpoint-based graph.
    
    Args:
        df: DataFrame with flow records
        feature_cols: List of feature column names  
        config: Configuration dict with keys:
            - src_ip_col, src_port_col, dst_ip_col, dst_port_col
            - mapping_mode: "ip_port" or "ip_only"
            - anti_leakage settings
            
    Returns:
        data: PyG Data object
        num_nodes: Number of nodes
    """
    logger.info("Creating endpoint-based graph for E-GraphSAGE")
    
    # Extract columns from config
    src_ip_col = config['data']['src_ip_col']
    dst_ip_col = config['data']['dst_ip_col']
    src_port_col = config['data'].get('src_port_col', 'Src Port')
    dst_port_col = config['data'].get('dst_port_col', 'Dst Port')
    mapping_mode = config['graph']['mapping_mode']
    label_col = config['data']['label_col']
    
    # Anti-leakage settings
    ip_random_map = config['graph']['anti_leakage']['enabled']
    map_scope = config['graph']['anti_leakage']['map_scope']
    seed = config['project']['seed']
    
    # Build endpoint mapping
    endpoint_to_idx, src_indices, dst_indices = _build_endpoint_mapping(
        df, src_ip_col, src_port_col, dst_ip_col, dst_port_col,
        mapping_mode, ip_random_map, map_scope, seed
    )
    
    num_nodes = len(endpoint_to_idx)
    num_edges = len(df)
    
    # Build edge_index [2, num_edges]
    edge_index = torch.tensor(
        np.stack([src_indices, dst_indices], axis=0),
        dtype=torch.long
    )
    
    # Extract edge features
    edge_attr = df[feature_cols].values.astype(np.float32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create edge labels (binary)
    edge_y = (df[label_col] != "Benign").astype(int).values
    edge_y = torch.tensor(edge_y, dtype=torch.long)
    
    # Node features: all-ones vector (as per E-GraphSAGE)
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
    
    benign = (edge_y == 0).sum().item()
    attack = (edge_y == 1).sum().item()
    
    logger.info(f"Graph: {num_nodes} nodes, {num_edges} edges")
    logger.info(f"Benign: {benign} ({benign/num_edges:.2%}), Attack: {attack} ({attack/num_edges:.2%})")
    
    return data, num_nodes


def _build_endpoint_mapping(
    df, src_ip_col, src_port_col, dst_ip_col, dst_port_col,
    mapping_mode, ip_random_map, map_scope, seed
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Build endpoint mapping and edge indices."""
    
    # Extract columns
    src_ips = df[src_ip_col].astype(str).values
    dst_ips = df[dst_ip_col].astype(str).values
    
    if mapping_mode == "ip_port":
        src_ports = df[src_port_col].astype(str).values
        dst_ports = df[dst_port_col].astype(str).values
    
    # Apply IP random mapping if enabled
    if ip_random_map:
        logger.info(f"Applying IP random mapping (scope={map_scope})")
        rng = np.random.RandomState(seed)
        
        if map_scope == "all_ips":
            unique_ips = np.unique(np.concatenate([src_ips, dst_ips]))
        else:  # src_ip_only
            unique_ips = np.unique(src_ips)
        
        ip_map = {ip: f"IP_{i:06d}" for i, ip in enumerate(unique_ips)}
        
        src_ips = np.array([ip_map.get(ip, ip) for ip in src_ips])
        if map_scope == "all_ips":
            dst_ips = np.array([ip_map.get(ip, ip) for ip in dst_ips])
    
    # Build endpoint strings
    if mapping_mode == "ip_port":
        src_endpoints = [f"{ip}:{port}" for ip, port in zip(src_ips, src_ports)]
        dst_endpoints = [f"{ip}:{port}" for ip, port in zip(dst_ips, dst_ports)]
    else:  # ip_only
        src_endpoints = src_ips.tolist()
        dst_endpoints = dst_ips.tolist()
    
    # Create mapping
    unique_endpoints = sorted(set(src_endpoints + dst_endpoints))
    endpoint_to_idx = {ep: idx for idx, ep in enumerate(unique_endpoints)}
    
    logger.info(f"Unique endpoints: {len(unique_endpoints)}")
    
    # Convert to indices
    src_indices = np.array([endpoint_to_idx[ep] for ep in src_endpoints], dtype=np.int64)
    dst_indices = np.array([endpoint_to_idx[ep] for ep in dst_endpoints], dtype=np.int64)
    
    return endpoint_to_idx, src_indices, dst_indices
