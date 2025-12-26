"""
Graph Construction Module for GNN Anomaly Detection
====================================================
Module này xây dựng graph từ network flow data.

Cách xây dựng Graph:
--------------------
1. NODE: Mỗi network flow (traffic record) là một node trong graph
   - Node features: Các đặc trưng của flow (packet size, duration, flags, etc.)

2. EDGE: Kết nối giữa các flows dựa trên sự tương đồng
   - Phương pháp KNN: Mỗi node kết nối với K nodes có features tương tự nhất
   - Sử dụng cosine similarity hoặc euclidean distance để đo độ tương đồng

3. LABELS: Binary classification
   - 0: Normal traffic (Benign)
   - 1: Anomaly traffic (Attack)

Lý do sử dụng KNN Graph:
------------------------
- Các flows có đặc trưng tương tự thường có cùng nhãn (normal/anomaly)
- GNN có thể học từ cấu trúc neighborhood để phân loại
- Phù hợp với bài toán không có cấu trúc graph tự nhiên
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib
import os
from tqdm import tqdm

import config


def build_knn_graph(features: np.ndarray, k: int = None, metric: str = 'cosine') -> tuple:
    """
    Xây dựng KNN graph từ features.

    Mỗi node (flow) được kết nối với K nodes có features tương tự nhất.

    Args:
        features: Ma trận features (num_samples, num_features)
        k: Số lượng neighbors cho mỗi node
        metric: Metric để tính khoảng cách ('cosine', 'euclidean', 'manhattan')

    Returns:
        Tuple (edge_index, edge_weights)
        - edge_index: [2, num_edges] tensor chứa các cặp (source, target)
        - edge_weights: [num_edges] tensor chứa trọng số của edges
    """
    if k is None:
        k = config.K_NEIGHBORS

    print(f"\n[INFO] Building KNN graph with k={k}, metric={metric}...")
    num_nodes = features.shape[0]
    print(f"[INFO] Number of nodes: {num_nodes}")

    # Khởi tạo NearestNeighbors
    if metric == 'cosine':
        # Với cosine, dùng brute force vì sklearn không hỗ trợ trực tiếp
        nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
    else:
        nn = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')

    # Fit và tìm neighbors
    print("[INFO] Finding nearest neighbors...")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    # Xây dựng edge list (bỏ self-loop - neighbor đầu tiên là chính nó)
    print("[INFO] Building edge list...")
    edge_list = []
    edge_weights = []

    for i in tqdm(range(num_nodes), desc="Building edges"):
        for j_idx in range(1, k+1):  # Bỏ index 0 (self)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]

            # Chuyển distance thành similarity
            if metric == 'cosine':
                # Cosine distance = 1 - cosine similarity
                weight = 1 - dist
            else:
                # Chuyển euclidean distance thành similarity
                weight = 1 / (1 + dist)

            edge_list.append([i, j])
            edge_weights.append(weight)

    # Chuyển thành numpy arrays
    edge_index = np.array(edge_list).T  # Shape: [2, num_edges]
    edge_weights = np.array(edge_weights)

    print(f"[INFO] Created {len(edge_weights)} edges")
    print(f"[INFO] Average edge weight: {edge_weights.mean():.4f}")

    return edge_index, edge_weights


def build_threshold_graph(features: np.ndarray, threshold: float = None) -> tuple:
    """
    Xây dựng graph dựa trên threshold similarity.

    Hai nodes được kết nối nếu cosine similarity > threshold.

    Args:
        features: Ma trận features
        threshold: Ngưỡng similarity để tạo edge

    Returns:
        Tuple (edge_index, edge_weights)
    """
    if threshold is None:
        threshold = config.SIMILARITY_THRESHOLD

    print(f"\n[INFO] Building threshold graph with threshold={threshold}...")

    # Tính cosine similarity matrix
    print("[INFO] Computing similarity matrix...")
    sim_matrix = cosine_similarity(features)

    # Lấy các cặp có similarity > threshold
    print("[INFO] Finding edges above threshold...")
    rows, cols = np.where(sim_matrix > threshold)

    # Loại bỏ self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    weights = sim_matrix[rows, cols]

    edge_index = np.stack([rows, cols])

    print(f"[INFO] Created {len(weights)} edges")

    return edge_index, weights


def build_batch_knn_graph(features: np.ndarray, k: int = None, batch_size: int = 10000) -> tuple:
    """
    Xây dựng KNN graph theo batch để tiết kiệm memory.

    Phù hợp với dataset lớn không fit vào memory.

    Args:
        features: Ma trận features
        k: Số lượng neighbors
        batch_size: Kích thước batch

    Returns:
        Tuple (edge_index, edge_weights)
    """
    if k is None:
        k = config.K_NEIGHBORS

    print(f"\n[INFO] Building batch KNN graph with k={k}, batch_size={batch_size}...")
    num_nodes = features.shape[0]

    # Fit NearestNeighbors trên toàn bộ data
    print("[INFO] Fitting NearestNeighbors model...")
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
    nn.fit(features)

    # Query theo batch
    all_edges = []
    all_weights = []

    num_batches = (num_nodes + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_nodes)

        batch_features = features[start_idx:end_idx]
        distances, indices = nn.kneighbors(batch_features)

        for i in range(len(batch_features)):
            global_i = start_idx + i
            for j_idx in range(1, k+1):
                j = indices[i, j_idx]
                dist = distances[i, j_idx]
                weight = 1 - dist  # Cosine similarity

                all_edges.append([global_i, j])
                all_weights.append(weight)

    edge_index = np.array(all_edges).T
    edge_weights = np.array(all_weights)

    print(f"[INFO] Created {len(edge_weights)} edges")

    return edge_index, edge_weights


def create_pyg_data(features: np.ndarray, labels: np.ndarray,
                    edge_index: np.ndarray, edge_weights: np.ndarray = None,
                    train_mask: np.ndarray = None, val_mask: np.ndarray = None,
                    test_mask: np.ndarray = None) -> Data:
    """
    Tạo PyTorch Geometric Data object.

    Args:
        features: Node features [num_nodes, num_features]
        labels: Node labels [num_nodes]
        edge_index: Edge connectivity [2, num_edges]
        edge_weights: Edge weights [num_edges]
        train_mask, val_mask, test_mask: Boolean masks cho split

    Returns:
        PyTorch Geometric Data object
    """
    print("\n[INFO] Creating PyTorch Geometric Data object...")

    # Chuyển sang tensors
    x = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    edge_index_tensor = torch.LongTensor(edge_index)

    # Tạo Data object
    data = Data(x=x, y=y, edge_index=edge_index_tensor)

    # Thêm edge weights nếu có
    if edge_weights is not None:
        data.edge_attr = torch.FloatTensor(edge_weights)

    # Thêm masks nếu có
    if train_mask is not None:
        data.train_mask = torch.BoolTensor(train_mask)
    if val_mask is not None:
        data.val_mask = torch.BoolTensor(val_mask)
    if test_mask is not None:
        data.test_mask = torch.BoolTensor(test_mask)

    print(f"[INFO] Data object created:")
    print(f"  - Nodes: {data.num_nodes}")
    print(f"  - Edges: {data.num_edges}")
    print(f"  - Features per node: {data.num_node_features}")
    print(f"  - Classes: {len(torch.unique(y))}")

    return data


def build_graph_from_splits(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                            k: int = None, graph_method: str = None) -> Data:
    """
    Xây dựng graph từ các splits đã có.

    Gộp tất cả data thành một graph lớn với train/val/test masks.

    Args:
        X_train, X_val, X_test: Features cho các splits
        y_train, y_val, y_test: Labels cho các splits
        k: Số neighbors cho KNN graph
        graph_method: Phương pháp xây dựng graph

    Returns:
        PyTorch Geometric Data object với masks
    """
    if k is None:
        k = config.K_NEIGHBORS
    if graph_method is None:
        graph_method = config.GRAPH_METHOD

    print("\n" + "="*60)
    print("GRAPH CONSTRUCTION")
    print("="*60)

    # 1. Gộp tất cả data
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    print(f"[INFO] Total samples: {len(y_all)}")

    # 2. Tạo masks
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    n_total = n_train + n_val + n_test

    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:n_train] = True

    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[n_train:n_train+n_val] = True

    test_mask = np.zeros(n_total, dtype=bool)
    test_mask[n_train+n_val:] = True

    print(f"[INFO] Train mask: {train_mask.sum()} nodes")
    print(f"[INFO] Val mask: {val_mask.sum()} nodes")
    print(f"[INFO] Test mask: {test_mask.sum()} nodes")

    # 3. Xây dựng graph
    if graph_method == 'knn':
        if len(y_all) > 50000:
            edge_index, edge_weights = build_batch_knn_graph(X_all, k=k)
        else:
            edge_index, edge_weights = build_knn_graph(X_all, k=k)
    elif graph_method == 'threshold':
        edge_index, edge_weights = build_threshold_graph(X_all)
    else:
        raise ValueError(f"Unknown graph method: {graph_method}")

    # 4. Tạo PyG Data object
    data = create_pyg_data(
        features=X_all,
        labels=y_all,
        edge_index=edge_index,
        edge_weights=edge_weights,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data


def save_graph(data: Data, file_path: str = None):
    """
    Lưu graph data.

    Args:
        data: PyTorch Geometric Data object
        file_path: Đường dẫn để lưu
    """
    if file_path is None:
        file_path = os.path.join(config.OUTPUT_DIR, 'graph_data.pt')

    torch.save(data, file_path)
    print(f"[INFO] Saved graph data to: {file_path}")


def load_graph(file_path: str = None) -> Data:
    """
    Load graph data.

    Args:
        file_path: Đường dẫn đến file

    Returns:
        PyTorch Geometric Data object
    """
    if file_path is None:
        file_path = os.path.join(config.OUTPUT_DIR, 'graph_data.pt')

    data = torch.load(file_path)
    print(f"[INFO] Loaded graph data from: {file_path}")
    print(f"[INFO] Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    return data


def get_graph_statistics(data: Data) -> dict:
    """
    Tính các thống kê của graph.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Dictionary chứa các thống kê
    """
    edge_index = data.edge_index.numpy()

    # Tính degree cho mỗi node
    degrees = np.bincount(edge_index[0], minlength=data.num_nodes)

    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_node_features,
        'avg_degree': degrees.mean(),
        'max_degree': degrees.max(),
        'min_degree': degrees.min(),
        'density': data.num_edges / (data.num_nodes * (data.num_nodes - 1)),
    }

    if hasattr(data, 'y'):
        labels = data.y.numpy()
        stats['class_distribution'] = {
            'normal': (labels == 0).sum(),
            'anomaly': (labels == 1).sum()
        }

    return stats


def print_graph_info(data: Data):
    """
    In thông tin chi tiết về graph.

    Args:
        data: PyTorch Geometric Data object
    """
    stats = get_graph_statistics(data)

    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    print(f"Number of nodes: {stats['num_nodes']:,}")
    print(f"Number of edges: {stats['num_edges']:,}")
    print(f"Number of features: {stats['num_features']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Max degree: {stats['max_degree']}")
    print(f"Min degree: {stats['min_degree']}")
    print(f"Graph density: {stats['density']:.6f}")

    if 'class_distribution' in stats:
        print(f"\nClass distribution:")
        print(f"  Normal: {stats['class_distribution']['normal']:,}")
        print(f"  Anomaly: {stats['class_distribution']['anomaly']:,}")


if __name__ == "__main__":
    # Test graph construction
    from data_preprocessing import preprocess_pipeline

    # Preprocess data
    preprocessed = preprocess_pipeline()

    # Build graph
    data = build_graph_from_splits(
        preprocessed['X_train'], preprocessed['X_val'], preprocessed['X_test'],
        preprocessed['y_train'], preprocessed['y_val'], preprocessed['y_test']
    )

    # Print statistics
    print_graph_info(data)

    # Save graph
    save_graph(data)

