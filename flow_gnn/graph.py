"""Build KNN graph from features using FAISS for efficient ANN."""

import logging
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import Tuple

def build_knn_graph(X_scaled: np.ndarray, k: int = 10) -> torch.Tensor:
    """Build KNN graph with progress tracking.
    
    Args:
        X_scaled: Scaled feature matrix (n_samples, n_features)
        k: Number of neighbors
        
    Returns:
        edge_index: Graph edges as [2, num_edges] tensor
    """
    
    logging.info(f"🔨 Building KNN graph (k={k})...")
    
    # Prepare data for FAISS - ensure contiguous float32
    X = np.ascontiguousarray(X_scaled, dtype=np.float32)
    n_samples, n_features = X.shape
    
    logging.info(f"   Data shape: {n_samples:,} samples × {n_features} features")
    
    # Normalize vectors for cosine similarity
    with tqdm(total=1, desc="Normalizing vectors", ncols=100) as pbar:
        faiss.normalize_L2(X)
        pbar.update(1)
    
    # Build FAISS index for cosine similarity
    with tqdm(total=1, desc="Building FAISS index", ncols=100) as pbar:
        index = faiss.IndexFlatIP(n_features)  # Inner Product (cosine after normalization)
        index.add(X)
        pbar.update(1)
    
    # Search for k+1 neighbors (including self) with progress
    logging.info(f"   Searching for {k} nearest neighbors...")
    with tqdm(total=n_samples, desc="KNN search", unit="samples", ncols=100) as pbar:
        batch_size = 10000
        all_indices = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            _, indices = index.search(X[i:end_idx], k + 1)
            all_indices.append(indices)
            pbar.update(end_idx - i)
        
        indices = np.vstack(all_indices)
    
    # Remove self-loops (first column is self)
    indices = indices[:, 1:]
    
    # Build edge list
    with tqdm(total=1, desc="Building edges", ncols=100) as pbar:
        row = np.repeat(np.arange(n_samples), k)
        col = indices.flatten()
        
        # Symmetrize to make graph undirected
        edges = np.vstack([
            np.concatenate([row, col]),
            np.concatenate([col, row])
        ])
        pbar.update(1)
    
    # Remove duplicates
    with tqdm(total=1, desc="Removing duplicates", ncols=100) as pbar:
        edges = np.unique(edges, axis=1)
        pbar.update(1)
    
    edge_index = torch.tensor(edges, dtype=torch.long)
    
    num_edges = edge_index.shape[1]
    avg_degree = num_edges / n_samples
    
    logging.info(f"✅ KNN graph built: {num_edges:,} edges, avg degree: {avg_degree:.2f}")
        
    return edge_index
