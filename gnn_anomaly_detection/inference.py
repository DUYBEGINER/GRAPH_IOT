"""
Inference Module for GNN Anomaly Detection
==========================================
Module này dùng để inference (dự đoán) trên dữ liệu mới.

Usage:
    python inference.py --input new_data.csv
    python inference.py --input new_data.csv --output predictions.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import joblib
from torch_geometric.data import Data

import config
from data_preprocessing import clean_data, normalize_features
from graph_construction import build_knn_graph, create_pyg_data
from models import get_model


def load_trained_model(model_path: str = None, model_type: str = 'GCN',
                       input_dim: int = None) -> torch.nn.Module:
    """
    Load trained model từ checkpoint.

    Args:
        model_path: Đường dẫn đến model checkpoint
        model_type: Loại model (GCN, GAT, GraphSAGE)
        input_dim: Số chiều input features

    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, 'best_model.pt')

    print(f"[INFO] Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Khởi tạo model
    model = get_model(model_type, input_dim)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[INFO] Model loaded (trained at epoch {checkpoint.get('epoch', 'N/A')})")

    return model


def load_preprocessing_artifacts():
    """
    Load các artifacts cần thiết cho preprocessing.

    Returns:
        Tuple (scaler, feature_names)
    """
    scaler_path = os.path.join(config.OUTPUT_DIR, 'scaler.pkl')
    features_path = os.path.join(config.OUTPUT_DIR, 'feature_names.txt')

    # Load scaler
    scaler = joblib.load(scaler_path)
    print(f"[INFO] Loaded scaler from: {scaler_path}")

    # Load feature names
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"[INFO] Loaded {len(feature_names)} feature names")

    return scaler, feature_names


def preprocess_new_data(df: pd.DataFrame, scaler, feature_names: list) -> np.ndarray:
    """
    Preprocess dữ liệu mới cho inference.

    Args:
        df: DataFrame chứa dữ liệu mới
        scaler: Fitted scaler
        feature_names: Danh sách feature names

    Returns:
        Normalized features array
    """
    print("[INFO] Preprocessing new data...")

    # Clean data
    df_clean = df.copy()

    # Drop columns không cần thiết (nếu có)
    cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    # Đảm bảo có đúng các features cần thiết
    for col in feature_names:
        if col not in df_clean.columns:
            print(f"[WARNING] Missing feature: {col}, filling with 0")
            df_clean[col] = 0

    # Chọn đúng các features theo thứ tự
    df_clean = df_clean[feature_names]

    # Chuyển sang numeric
    for col in feature_names:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Xử lý missing và infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)

    # Normalize
    X = df_clean.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_normalized = scaler.transform(X)

    print(f"[INFO] Preprocessed data shape: {X_normalized.shape}")

    return X_normalized


def build_inference_graph(features: np.ndarray, k: int = None) -> Data:
    """
    Xây dựng graph cho inference.

    Args:
        features: Normalized features
        k: Số neighbors cho KNN

    Returns:
        PyTorch Geometric Data object
    """
    if k is None:
        k = config.K_NEIGHBORS

    print(f"[INFO] Building inference graph with k={k}...")

    # Build KNN graph
    from graph_construction import build_knn_graph
    edge_index, edge_weights = build_knn_graph(features, k=k)

    # Create Data object (không cần labels cho inference)
    x = torch.FloatTensor(features)
    edge_index_tensor = torch.LongTensor(edge_index)

    data = Data(x=x, edge_index=edge_index_tensor)

    if edge_weights is not None:
        data.edge_attr = torch.FloatTensor(edge_weights)

    print(f"[INFO] Graph created: {data.num_nodes} nodes, {data.num_edges} edges")

    return data


@torch.no_grad()
def predict(model: torch.nn.Module, data: Data,
            device: str = None) -> tuple:
    """
    Dự đoán trên dữ liệu mới.

    Args:
        model: Trained GNN model
        data: Graph data
        device: Device để inference

    Returns:
        Tuple (predictions, probabilities)
    """
    if device is None:
        device = config.DEVICE

    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)
    model.eval()

    print("[INFO] Running inference...")

    # Forward pass
    out = model(data)
    probs = torch.exp(out)

    # Get predictions
    predictions = out.argmax(dim=1).cpu().numpy()
    probabilities = probs.cpu().numpy()

    # Statistics
    num_normal = (predictions == 0).sum()
    num_anomaly = (predictions == 1).sum()

    print(f"[INFO] Predictions:")
    print(f"  Normal: {num_normal} ({100*num_normal/len(predictions):.2f}%)")
    print(f"  Anomaly: {num_anomaly} ({100*num_anomaly/len(predictions):.2f}%)")

    return predictions, probabilities


def inference_pipeline(input_file: str, output_file: str = None,
                       model_type: str = 'GCN', k: int = None) -> pd.DataFrame:
    """
    Pipeline hoàn chỉnh cho inference.

    Args:
        input_file: Đường dẫn đến file CSV input
        output_file: Đường dẫn để lưu kết quả
        model_type: Loại model
        k: Số neighbors cho graph

    Returns:
        DataFrame với predictions
    """
    print("\n" + "="*60)
    print("INFERENCE PIPELINE")
    print("="*60)

    # 1. Load data
    print(f"\n[INFO] Loading data from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"[INFO] Loaded {len(df)} samples")

    # 2. Load preprocessing artifacts
    scaler, feature_names = load_preprocessing_artifacts()

    # 3. Preprocess data
    X = preprocess_new_data(df, scaler, feature_names)

    # 4. Build graph
    data = build_inference_graph(X, k=k)

    # 5. Load model
    model = load_trained_model(model_type=model_type, input_dim=len(feature_names))

    # 6. Predict
    predictions, probabilities = predict(model, data)

    # 7. Create result DataFrame
    result_df = df.copy()
    result_df['Predicted_Label'] = predictions
    result_df['Predicted_Class'] = ['Normal' if p == 0 else 'Anomaly' for p in predictions]
    result_df['Prob_Normal'] = probabilities[:, 0]
    result_df['Prob_Anomaly'] = probabilities[:, 1]

    # 8. Save results
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"\n[INFO] Results saved to: {output_file}")

    return result_df


def main():
    """Main function cho inference."""
    parser = argparse.ArgumentParser(description='GNN Anomaly Detection Inference')

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Đường dẫn đến file CSV input')

    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Đường dẫn để lưu predictions (default: input_predictions.csv)')

    parser.add_argument('--model', '-m', type=str, default='GCN',
                       choices=['GCN', 'GAT', 'GraphSAGE'],
                       help='Loại model (default: GCN)')

    parser.add_argument('--k-neighbors', '-k', type=int, default=None,
                       help='Số neighbors cho KNN graph')

    args = parser.parse_args()

    # Default output file
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_predictions.csv"

    # Run inference
    result_df = inference_pipeline(
        input_file=args.input,
        output_file=args.output,
        model_type=args.model,
        k=args.k_neighbors
    )

    print("\n[INFO] Inference completed!")


if __name__ == "__main__":
    main()

