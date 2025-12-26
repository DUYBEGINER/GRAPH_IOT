"""
Graph Construction for Network Traffic Data
Xây dựng đồ thị từ dữ liệu network traffic để sử dụng với GNN
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
PROCESSED_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\processed_data"
GRAPH_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\graph_data"

# Graph construction parameters
K_NEIGHBORS = 10  # Số lượng neighbors cho KNN graph
SIMILARITY_THRESHOLD = 0.5  # Ngưỡng similarity để tạo edge
GRAPH_TYPE = 'similar'  # 'knn' hoặc 'similarity'

# ============================================================================
# GRAPH CONSTRUCTION CLASS
# ============================================================================

class NetworkTrafficGraphBuilder:
    """Class để xây dựng đồ thị từ network traffic data"""

    def __init__(self, k_neighbors=10, similarity_threshold=0.5, graph_type='knn'):
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.graph_type = graph_type

    def build_knn_graph(self, X, batch_size=5000):
        """
        Xây dựng KNN graph từ features

        Args:
            X: Feature matrix (n_samples, n_features)
            batch_size: Batch size để xử lý (tránh memory overflow)

        Returns:
            edge_index: Edge indices (2, num_edges)
        """
        print(f"\nXây dựng KNN graph với k={self.k_neighbors}...")

        n_samples = X.shape[0]

        if n_samples <= batch_size:
            # Xử lý trực tiếp nếu dữ liệu nhỏ
            adjacency = kneighbors_graph(
                X,
                n_neighbors=self.k_neighbors,
                mode='connectivity',
                include_self=False
            )
        else:
            # Xử lý theo batch nếu dữ liệu lớn - tạo edge list thay vì adjacency matrix
            print(f"  Dữ liệu lớn, xử lý theo batch ({batch_size} samples/batch)...")
            edges = []

            for i in tqdm(range(0, n_samples, batch_size), desc="  Building KNN graph"):
                end_idx = min(i + batch_size, n_samples)
                batch_X = X[i:end_idx]

                # Tìm k nearest neighbors trong toàn bộ dataset
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto').fit(X)
                distances, indices = nbrs.kneighbors(batch_X)

                # Tạo edges (bỏ qua neighbor đầu tiên vì đó là chính nó)
                for local_idx in range(len(batch_X)):
                    global_idx = i + local_idx
                    for neighbor_idx in indices[local_idx][1:]:  # Skip first (itself)
                        edges.append([global_idx, neighbor_idx])

            # Chuyển edge list sang COO format
            edges = np.array(edges).T
            edge_index = torch.tensor(edges, dtype=torch.long)
            print(f"✓ Graph created: {n_samples} nodes, {edge_index.shape[1]} edges")
            return edge_index

        # Chuyển sang edge_index format (cho trường hợp không batch)
        adjacency = adjacency.tocoo()
        edge_index = torch.tensor(
            np.vstack([adjacency.row, adjacency.col]),
            dtype=torch.long
        )

        print(f"✓ Graph created: {n_samples} nodes, {edge_index.shape[1]} edges")

        return edge_index

    def build_similarity_graph(self, X, batch_size=1000):
        """
        Xây dựng graph dựa trên cosine similarity

        Args:
            X: Feature matrix
            batch_size: Batch size

        Returns:
            edge_index: Edge indices
        """
        print(f"\nXây dựng Similarity graph (threshold={self.similarity_threshold})...")

        n_samples = X.shape[0]
        edges = []

        # Xử lý theo batch
        for i in tqdm(range(0, n_samples, batch_size), desc="  Computing similarity"):
            end_idx = min(i + batch_size, n_samples)

            # Tính similarity cho batch hiện tại với tất cả samples
            similarities = cosine_similarity(X[i:end_idx], X)

            # Tìm các edges có similarity > threshold
            for local_idx in range(end_idx - i):
                global_idx = i + local_idx
                # Lấy indices có similarity > threshold (không bao gồm chính nó)
                similar_indices = np.where(
                    (similarities[local_idx] > self.similarity_threshold) &
                    (np.arange(n_samples) != global_idx)
                )[0]

                # Thêm edges
                for j in similar_indices:
                    edges.append([global_idx, j])

        if len(edges) == 0:
            print("  ⚠ Không tìm thấy edges, giảm threshold hoặc dùng KNN graph")
            # Fallback to KNN
            return self.build_knn_graph(X, batch_size)

        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)

        print(f"✓ Graph created: {n_samples} nodes, {edge_index.shape[1]} edges")

        return edge_index

    def create_graph_data(self, X, y, edge_index):
        """
        Tạo PyTorch Geometric Data object

        Args:
            X: Node features
            y: Node labels
            edge_index: Edge indices

        Returns:
            Data object
        """
        # Chuyển đổi sang tensor
        x = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        # Tạo Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        return data


class CICIDSGraphDataset(InMemoryDataset):
    """Custom Dataset cho CICIDS2018 Graph Data"""

    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.data_list is not None:
            data_list = self.data_list

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def build_graph_dataset():
    """Xây dựng graph dataset từ processed data"""

    print("=" * 80)
    print("BUILDING GRAPH DATASET FOR GNN")
    print("=" * 80)

    # Tạo output directory
    os.makedirs(GRAPH_DATA_DIR, exist_ok=True)

    # Load processed data
    print("\nLoading processed data...")
    X = np.load(os.path.join(PROCESSED_DATA_DIR, "X_features.npy"))
    y_binary = np.load(os.path.join(PROCESSED_DATA_DIR, "y_binary.npy"))
    y_multi = np.load(os.path.join(PROCESSED_DATA_DIR, "y_multi.npy"))

    with open(os.path.join(PROCESSED_DATA_DIR, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    print(f"✓ Loaded data: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"✓ Binary classes: {len(np.unique(y_binary))}")
    print(f"✓ Multi classes: {len(np.unique(y_multi))}")

    # Giảm kích thước nếu quá lớn (optional - để training nhanh hơn)
    MAX_SAMPLES = 200000  # Giới hạn số samples để training nhanh
    if X.shape[0] > MAX_SAMPLES:
        print(f"\n⚠ Dataset lớn ({X.shape[0]:,} samples), sampling {MAX_SAMPLES:,} samples...")
        indices = np.random.choice(X.shape[0], MAX_SAMPLES, replace=False)
        X = X[indices]
        y_binary = y_binary[indices]
        y_multi = y_multi[indices]
        print(f"✓ Sampled data: {X.shape[0]:,} samples")

    # Build graph
    builder = NetworkTrafficGraphBuilder(
        k_neighbors=K_NEIGHBORS,
        similarity_threshold=SIMILARITY_THRESHOLD,
        graph_type=GRAPH_TYPE
    )

    if GRAPH_TYPE == 'knn':
        edge_index = builder.build_knn_graph(X)
    else:
        edge_index = builder.build_similarity_graph(X)

    # Create graph data objects
    print("\nCreating graph data objects...")

    # Binary classification graph
    print("  Creating binary classification graph...")
    graph_binary = builder.create_graph_data(X, y_binary, edge_index)

    # Multi-class classification graph
    print("  Creating multi-class classification graph...")
    graph_multi = builder.create_graph_data(X, y_multi, edge_index)

    # Save graphs
    print("\nSaving graph data...")
    torch.save(graph_binary, os.path.join(GRAPH_DATA_DIR, "graph_binary.pt"))
    torch.save(graph_multi, os.path.join(GRAPH_DATA_DIR, "graph_multi.pt"))

    # Save edge_index separately
    torch.save(edge_index, os.path.join(GRAPH_DATA_DIR, "edge_index.pt"))

    # Save graph metadata
    graph_metadata = {
        'n_nodes': X.shape[0],
        'n_features': X.shape[1],
        'n_edges': edge_index.shape[1],
        'k_neighbors': K_NEIGHBORS,
        'graph_type': GRAPH_TYPE,
        'avg_degree': edge_index.shape[1] / X.shape[0]
    }

    with open(os.path.join(GRAPH_DATA_DIR, "graph_metadata.pkl"), 'wb') as f:
        pickle.dump(graph_metadata, f)

    print("\n" + "=" * 80)
    print("GRAPH CONSTRUCTION COMPLETED!")
    print("=" * 80)
    print(f"Nodes: {graph_metadata['n_nodes']:,}")
    print(f"Features per node: {graph_metadata['n_features']}")
    print(f"Edges: {graph_metadata['n_edges']:,}")
    print(f"Average degree: {graph_metadata['avg_degree']:.2f}")
    print(f"Output directory: {GRAPH_DATA_DIR}")
    print("=" * 80)

    return graph_binary, graph_multi, graph_metadata


if __name__ == "__main__":
    graph_binary, graph_multi, metadata = build_graph_dataset()

