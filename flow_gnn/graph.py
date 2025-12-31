"""Build KNN graph from features using FAISS for efficient ANN."""

import logging
import torch
import numpy as np
import faiss
from tqdm import tqdm

from . import config as cfg


def build_knn_graph(X_scaled: np.ndarray, k: int = None) -> torch.Tensor:
    """Build KNN graph with progress tracking.
    
    Args:
        X_scaled: Scaled feature matrix (n_samples, n_features)
        k: Number of neighbors (default from config)
        
    Returns:
        edge_index: Graph edges as [2, num_edges] tensor
    """
    if k is None:
        k = cfg.K_NEIGHBORS
    
    print(f"\n🔨 Building KNN graph (k={k})...")
    
    # Prepare data for FAISS
    X = np.ascontiguousarray(X_scaled, dtype=np.float32)
    n_samples, n_features = X.shape
    print(f"   Data shape: {n_samples:,} samples × {n_features} features")
    
    # Step 1: Normalize vectors for cosine similarity
    with tqdm(total=3, desc="📍 Graph construction", ncols=100) as main_pbar:
        faiss.normalize_L2(X)
        main_pbar.update(1)
        main_pbar.set_postfix_str("Normalized vectors")
        
        # Step 2: Build FAISS index
        index = faiss.IndexFlatIP(n_features)
        index.add(X)
        main_pbar.update(1)
        main_pbar.set_postfix_str("Built FAISS index")
        
        # Step 3: KNN search with batch processing
        batch_size = 10000
        all_indices = []
        
        for i in tqdm(range(0, n_samples, batch_size), 
                      desc="   🔍 KNN search", unit="batch", ncols=100, leave=False):
            end_idx = min(i + batch_size, n_samples)
            _, indices = index.search(X[i:end_idx], k + 1)
            all_indices.append(indices)
        
        indices = np.vstack(all_indices)
        main_pbar.update(1)
        main_pbar.set_postfix_str("KNN search complete")
    
    # Remove self-loops (first column is self)
    indices = indices[:, 1:]
    
    # Build edge list
    print("   🔗 Building edge list...")
    row = np.repeat(np.arange(n_samples), k)
    col = indices.flatten()
    
    # Symmetrize to make graph undirected
    edges = np.vstack([
        np.concatenate([row, col]),
        np.concatenate([col, row])
    ])
    
    # Remove duplicates
    edges = np.unique(edges, axis=1)
    edge_index = torch.tensor(edges, dtype=torch.long)
    
    num_edges = edge_index.shape[1]
    avg_degree = num_edges / n_samples
    
    print(f"✅ KNN graph built: {num_edges:,} edges, avg degree: {avg_degree:.2f}")
        
    return edge_index
