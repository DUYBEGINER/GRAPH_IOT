"""Graph construction utilities using KNN."""

import logging
import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from typing import Tuple

logger = logging.getLogger(__name__)


def build_knn_graph(
    X_scaled: np.ndarray, 
    k: int, 
    metric: str = "cosine"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build KNN graph from feature matrix.
    
    Args:
        X_scaled: Scaled feature matrix (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges] (similarity scores)
        
    Note:
        - Graph is made undirected by symmetrization
        - Edge weights are similarities (1 - distance) for cosine metric
    """
    try:
        logger.info(f"Building KNN graph with k={k}, metric={metric}...")
        
        # Build KNN graph
        adj = kneighbors_graph(
            X_scaled,
            n_neighbors=k,
            metric=metric,
            mode="distance",
            include_self=False,
            n_jobs=-1
        )
        
        # Symmetrize to make graph undirected
        adj = adj.maximum(adj.T)
        
        # Extract edges
        row, col = adj.nonzero()
        edge_index = torch.tensor([row, col], dtype=torch.long)
        
        # Convert distance to similarity
        dist = np.array(adj[row, col]).flatten()
        if metric == "cosine":
            # Cosine distance in [0, 2], convert to similarity
            edge_weight = torch.tensor(1.0 - dist, dtype=torch.float)
        else:
            # For other metrics, use inverse distance as weight
            # Add small epsilon to avoid division by zero
            edge_weight = torch.tensor(1.0 / (dist + 1e-8), dtype=torch.float)
        
        num_nodes = X_scaled.shape[0]
        num_edges = edge_index.shape[1]
        avg_degree = num_edges / num_nodes
        
        logger.info(f"Graph constructed: {num_nodes} nodes, {num_edges} edges")
        logger.info(f"Average node degree: {avg_degree:.2f}")
        logger.info(f"Edge weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")
        
        return edge_index, edge_weight
        
    except Exception as e:
        logger.error(f"Error building KNN graph: {e}")
        raise
