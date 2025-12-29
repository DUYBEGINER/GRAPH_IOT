"""
GNN Inference/Demo Script for IoT Network Anomaly Detection
Script để demo model GNN đã được huấn luyện với best_model_binary.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GNN MODEL DEFINITIONS (copy từ train_gnn.py)
# ============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_channels * heads))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                  heads=1, concat=False, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, dropout=0.5, aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class HybridGNN(nn.Module):
    """Hybrid GNN combining GCN and GAT"""
    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, heads=4, dropout=0.5):
        super(HybridGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.gcn_convs = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        self.gat_bns = nn.ModuleList()

        self.gcn_convs.append(GCNConv(in_channels, hidden_channels))
        self.gcn_bns.append(BatchNorm(hidden_channels))
        self.gat_convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.gat_bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gcn_bns.append(BatchNorm(hidden_channels))
            self.gat_convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
            self.gat_bns.append(BatchNorm(hidden_channels))

        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_bn = BatchNorm(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x_gcn = x
        for i in range(self.num_layers - 1):
            x_gcn = self.gcn_convs[i](x_gcn, edge_index)
            x_gcn = self.gcn_bns[i](x_gcn)
            x_gcn = F.relu(x_gcn)
            x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)

        x_gat = x
        for i in range(self.num_layers - 1):
            x_gat = self.gat_convs[i](x_gat, edge_index)
            x_gat = self.gat_bns[i](x_gat)
            x_gat = F.elu(x_gat)
            x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)

        x = torch.cat([x_gcn, x_gat], dim=1)
        x = self.fusion(x)
        x = self.fusion_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class GNNInference:
    """Class để thực hiện inference với GNN model"""

    def __init__(self, model_path, processed_data_dir, model_name='GAT',
                 hidden_channels=128, num_layers=3, heads=4, dropout=0.3,
                 k_neighbors=8, device=None):
        """
        Initialize GNN Inference

        Args:
            model_path: Đường dẫn tới file model (.pt)
            processed_data_dir: Đường dẫn tới thư mục chứa scaler, metadata
            model_name: Tên model ('GCN', 'GAT', 'GraphSAGE', 'Hybrid')
            hidden_channels: Số hidden channels
            num_layers: Số layers
            heads: Số attention heads (cho GAT)
            dropout: Dropout rate
            k_neighbors: Số neighbors cho KNN graph
            device: Device để chạy model
        """
        self.model_path = model_path
        self.processed_data_dir = processed_data_dir
        self.model_name = model_name
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.k_neighbors = k_neighbors
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔥 Using device: {self.device}")

        # Load metadata và preprocessing objects
        self._load_preprocessing_objects()

        # Load model
        self._load_model()

    def _load_preprocessing_objects(self):
        """Load scaler, metadata, feature names"""
        print("\n📦 Loading preprocessing objects...")

        # Load scaler
        scaler_path = os.path.join(self.processed_data_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"  ✓ Loaded scaler from {scaler_path}")

        # Load metadata
        metadata_path = os.path.join(self.processed_data_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"  ✓ Loaded metadata from {metadata_path}")

        self.num_features = self.metadata['n_features']
        self.num_classes = 2  # Binary classification (Benign vs Attack)
        self.feature_names = self.metadata['feature_names']
        self.class_names = ['Benign', 'Attack']

        print(f"  ✓ Num features: {self.num_features}")
        print(f"  ✓ Num classes: {self.num_classes}")

    def _detect_model_config(self, checkpoint):
        """Auto-detect model configuration from checkpoint"""
        state_dict = checkpoint['model_state_dict']

        # Detect hidden channels
        if 'classifier.weight' in state_dict:
            hidden_channels = state_dict['classifier.weight'].shape[1]
        else:
            hidden_channels = self.hidden_channels

        # Detect num_layers
        num_layers = 0
        for key in state_dict.keys():
            if key.startswith('convs.'):
                layer_num = int(key.split('.')[1])
                num_layers = max(num_layers, layer_num + 1)

        if num_layers == 0:
            num_layers = self.num_layers

        # Detect model type
        model_name = self.model_name
        if 'lin_l' in str(state_dict.keys()):
            model_name = 'GraphSAGE'
        elif 'att_src' in str(state_dict.keys()) or 'att_dst' in str(state_dict.keys()):
            model_name = 'GAT'
        elif 'fusion' in str(state_dict.keys()):
            model_name = 'Hybrid'
        elif any('gcn' in key.lower() for key in state_dict.keys()):
            model_name = 'GCN'

        return model_name, hidden_channels, num_layers

    def _load_model(self):
        """Load trained model"""
        print(f"\n🤖 Loading model...")

        # Load checkpoint first
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Auto-detect configuration
        detected_model, detected_hidden, detected_layers = self._detect_model_config(checkpoint)

        print(f"  🔍 Auto-detected configuration:")
        print(f"     Model type: {detected_model}")
        print(f"     Hidden channels: {detected_hidden}")
        print(f"     Num layers: {detected_layers}")

        # Use detected config
        self.model_name = detected_model
        self.hidden_channels = detected_hidden
        self.num_layers = detected_layers

        # Create model
        models = {
            'GCN': GCN,
            'GAT': GAT,
            'GraphSAGE': GraphSAGE,
            'Hybrid': HybridGNN
        }

        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not supported. Choose: {list(models.keys())}")

        model_class = models[self.model_name]

        if self.model_name == 'GAT' or self.model_name == 'Hybrid':
            self.model = model_class(
                in_channels=self.num_features,
                hidden_channels=self.hidden_channels,
                num_classes=self.num_classes,
                num_layers=self.num_layers,
                heads=self.heads,
                dropout=self.dropout
            )
        else:
            self.model = model_class(
                in_channels=self.num_features,
                hidden_channels=self.hidden_channels,
                num_classes=self.num_classes,
                num_layers=self.num_layers,
                dropout=self.dropout
            )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"  ✓ Model loaded from {self.model_path}")
        print(f"  ✓ Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
        print(f"  ✓ Best val acc: {checkpoint.get('best_val_acc', 'N/A'):.4f}")

    def _preprocess_data(self, data):
        """
        Preprocess raw data

        Args:
            data: DataFrame hoặc numpy array

        Returns:
            Preprocessed numpy array
        """
        if isinstance(data, pd.DataFrame):
            # Đảm bảo có đủ các features
            if len(data.columns) != self.num_features:
                print(f"⚠️  Warning: Expected {self.num_features} features, got {len(data.columns)}")
            data = data.values

        # Scale data
        data_scaled = self.scaler.transform(data)

        return data_scaled

    def _build_knn_graph(self, X):
        """
        Build KNN graph from features

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            edge_index: Edge indices (2, num_edges)
        """
        print(f"\n🔗 Building KNN graph (k={self.k_neighbors})...")

        n_samples = X.shape[0]

        # Special case: single sample - create self-loop
        if n_samples == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            print(f"  ✓ Created self-loop for single sample")
            return edge_index

        k = min(self.k_neighbors, n_samples - 1)

        # Use NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Create edges
        edges = []
        for i in range(n_samples):
            for j in indices[i][1:]:  # Skip first (itself)
                edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        print(f"  ✓ Created {edge_index.shape[1]} edges")

        return edge_index

    def predict(self, data, return_proba=False):
        """
        Predict trên dữ liệu mới

        Args:
            data: DataFrame hoặc numpy array
            return_proba: Có trả về probability không

        Returns:
            predictions: Class predictions
            probabilities: Class probabilities (nếu return_proba=True)
        """
        print("\n" + "="*80)
        print("🔮 STARTING INFERENCE")
        print("="*80)

        # Preprocess
        X = self._preprocess_data(data)
        print(f"✓ Preprocessed {X.shape[0]} samples with {X.shape[1]} features")

        # Build graph
        edge_index = self._build_knn_graph(X)

        # Create PyG Data object
        x = torch.tensor(X, dtype=torch.float).to(self.device)
        edge_index = edge_index.to(self.device)

        graph_data = Data(x=x, edge_index=edge_index)

        # Predict
        print("\n🎯 Running inference...")
        with torch.no_grad():
            output = self.model(graph_data.x, graph_data.edge_index)
            probabilities = F.softmax(output, dim=1).cpu().numpy()
            predictions = output.argmax(dim=1).cpu().numpy()

        print(f"✓ Inference completed for {len(predictions)} samples")

        if return_proba:
            return predictions, probabilities
        return predictions

    def predict_single(self, sample):
        """
        Predict cho 1 sample

        Args:
            sample: Dictionary hoặc array of features

        Returns:
            prediction: Class prediction
            probability: Class probabilities
            class_name: Tên class
        """
        if isinstance(sample, dict):
            # Convert dict to array theo thứ tự feature_names
            sample_array = np.array([sample[feat] for feat in self.feature_names])
        else:
            sample_array = np.array(sample)

        # Reshape to 2D
        sample_array = sample_array.reshape(1, -1)

        # Predict
        pred, proba = self.predict(sample_array, return_proba=True)

        pred_class = pred[0]
        pred_proba = proba[0]
        class_name = self.class_names[pred_class]

        return pred_class, pred_proba, class_name

    def predict_batch(self, data, batch_size=1000):
        """
        Predict theo batch cho dữ liệu lớn

        Args:
            data: DataFrame hoặc numpy array
            batch_size: Batch size

        Returns:
            predictions: All predictions
            probabilities: All probabilities
        """
        n_samples = len(data)
        all_predictions = []
        all_probabilities = []

        print(f"\n📊 Processing {n_samples} samples in batches of {batch_size}")

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            if isinstance(data, pd.DataFrame):
                batch_data = data.iloc[i:end_idx]
            else:
                batch_data = data[i:end_idx]

            print(f"  Processing batch {i//batch_size + 1}: samples {i}-{end_idx}")

            pred, proba = self.predict(batch_data, return_proba=True)
            all_predictions.extend(pred)
            all_probabilities.extend(proba)

        return np.array(all_predictions), np.array(all_probabilities)

    def demo_summary(self, predictions, probabilities):
        """In ra summary của predictions"""
        print("\n" + "="*80)
        print("📊 PREDICTION SUMMARY")
        print("="*80)

        # Count by class
        unique, counts = np.unique(predictions, return_counts=True)

        for class_idx, count in zip(unique, counts):
            class_name = self.class_names[class_idx]
            percentage = (count / len(predictions)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.2f}%)")

        # Average confidence
        avg_confidence = np.max(probabilities, axis=1).mean()
        print(f"\n  Average Confidence: {avg_confidence:.4f}")

        print("="*80)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_with_test_data(model_path, processed_data_dir, num_samples=100):
    """
    Demo với test data từ processed data

    Args:
        model_path: Đường dẫn tới model
        processed_data_dir: Đường dẫn tới processed data
        num_samples: Số samples để test
    """
    print("\n" + "="*80)
    print("🎬 DEMO WITH TEST DATA")
    print("="*80)

    # Initialize inference
    inferencer = GNNInference(
        model_path=model_path,
        processed_data_dir=processed_data_dir,
        model_name='GAT',  # Thay đổi theo model đã train
        hidden_channels=128,
        num_layers=3,
        heads=4,
        dropout=0.3,
        k_neighbors=8
    )

    # Load test data
    print(f"\n📂 Loading test data...")
    X_path = os.path.join(processed_data_dir, 'X_features.npy')
    y_path = os.path.join(processed_data_dir, 'y_binary.npy')

    X = np.load(X_path)
    y = np.load(y_path)

    # Random sample
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    X_test = X[indices]
    y_test = y[indices]

    print(f"✓ Loaded {len(X_test)} test samples")

    # Predict
    predictions, probabilities = inferencer.predict(X_test, return_proba=True)

    # Summary
    inferencer.demo_summary(predictions, probabilities)

    # Accuracy
    accuracy = (predictions == y_test).mean()
    print(f"\n🎯 Accuracy on test samples: {accuracy:.4f}")

    # Show some examples
    print("\n" + "="*80)
    print("📝 SAMPLE PREDICTIONS (First 10)")
    print("="*80)
    for i in range(min(10, len(predictions))):
        true_label = inferencer.class_names[y_test[i]]
        pred_label = inferencer.class_names[predictions[i]]
        confidence = probabilities[i][predictions[i]]
        status = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"  {status} Sample {i+1}: True={true_label:10s} | Pred={pred_label:10s} | Confidence={confidence:.4f}")

    return inferencer, predictions, probabilities


def demo_with_csv(model_path, processed_data_dir, csv_path, num_samples=100):
    """
    Demo với CSV file mới

    Args:
        model_path: Đường dẫn tới model
        processed_data_dir: Đường dẫn tới processed data
        csv_path: Đường dẫn tới CSV file
        num_samples: Số samples để test
    """
    print("\n" + "="*80)
    print("🎬 DEMO WITH NEW CSV DATA")
    print("="*80)

    # Initialize inference
    inferencer = GNNInference(
        model_path=model_path,
        processed_data_dir=processed_data_dir,
        model_name='GAT',
        hidden_channels=128,
        num_layers=3,
        heads=4,
        dropout=0.3,
        k_neighbors=8
    )

    # Load CSV
    print(f"\n📂 Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Get only feature columns
    feature_cols = inferencer.feature_names
    df_features = df[feature_cols]

    # Sample
    if len(df_features) > num_samples:
        df_features = df_features.sample(n=num_samples, random_state=42)

    print(f"✓ Loaded {len(df_features)} samples")

    # Predict
    predictions, probabilities = inferencer.predict(df_features, return_proba=True)

    # Summary
    inferencer.demo_summary(predictions, probabilities)

    # Show some examples
    print("\n" + "="*80)
    print("📝 SAMPLE PREDICTIONS (First 10)")
    print("="*80)
    for i in range(min(10, len(predictions))):
        pred_label = inferencer.class_names[predictions[i]]
        confidence = probabilities[i][predictions[i]]
        print(f"  Sample {i+1}: Predicted={pred_label:10s} | Confidence={confidence:.4f}")

    return inferencer, predictions, probabilities


def demo_interactive(model_path, processed_data_dir):
    """
    Demo interactive - nhập features thủ công

    Args:
        model_path: Đường dẫn tới model
        processed_data_dir: Đường dẫn tới processed data
    """
    print("\n" + "="*80)
    print("🎬 INTERACTIVE DEMO")
    print("="*80)

    # Initialize inference
    inferencer = GNNInference(
        model_path=model_path,
        processed_data_dir=processed_data_dir,
        model_name='GAT',
        hidden_channels=128,
        num_layers=3,
        heads=4,
        dropout=0.3,
        k_neighbors=8
    )

    print("\nNote: Để demo nhanh, hãy tạo random sample từ test data")
    print("      Hoặc nhập features thủ công (70 features)")

    # Load sample from test data
    X_path = os.path.join(processed_data_dir, 'X_features.npy')
    X = np.load(X_path)

    # Random sample
    sample_idx = np.random.randint(0, len(X))
    sample = X[sample_idx]

    print(f"\n📊 Using random sample #{sample_idx}")

    # Predict
    pred_class, pred_proba, class_name = inferencer.predict_single(sample)

    print("\n" + "="*80)
    print("🎯 PREDICTION RESULT")
    print("="*80)
    print(f"  Predicted Class: {class_name}")
    print(f"  Confidence: {pred_proba[pred_class]:.4f}")
    print(f"\n  Class Probabilities:")
    for i, prob in enumerate(pred_proba):
        print(f"    {inferencer.class_names[i]}: {prob:.4f}")
    print("="*80)

    return inferencer


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function để chạy demo"""

    # Cấu hình
    MODEL_PATH = "/GNN/models/best_model_binary.pt"
    PROCESSED_DATA_DIR = "/processed_data"

    # Kiểm tra files
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return

    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"❌ Processed data directory not found: {PROCESSED_DATA_DIR}")
        return

    print("="*80)
    print("🚀 GNN INFERENCE DEMO")
    print("="*80)
    print("\nChọn chế độ demo:")
    print("  1. Demo với test data (từ processed data)")
    print("  2. Demo với CSV file mới")
    print("  3. Demo interactive (random sample)")
    print("  4. Chạy tất cả")

    choice = input("\nNhập lựa chọn (1-4): ").strip()

    if choice == "1":
        demo_with_test_data(MODEL_PATH, PROCESSED_DATA_DIR, num_samples=10000)

    elif choice == "2":
        csv_path = input("Nhập đường dẫn tới CSV file: ").strip()
        if os.path.exists(csv_path):
            demo_with_csv(MODEL_PATH, PROCESSED_DATA_DIR, csv_path, num_samples=100)
        else:
            print(f"❌ CSV file not found: {csv_path}")

    elif choice == "3":
        demo_interactive(MODEL_PATH, PROCESSED_DATA_DIR)

    elif choice == "4":
        # Demo with test data
        print("\n\n" + "="*80)
        print("DEMO 1: TEST DATA")
        print("="*80)
        demo_with_test_data(MODEL_PATH, PROCESSED_DATA_DIR, num_samples=100)

        # Demo interactive
        print("\n\n" + "="*80)
        print("DEMO 2: INTERACTIVE")
        print("="*80)
        demo_interactive(MODEL_PATH, PROCESSED_DATA_DIR)

    else:
        print("❌ Lựa chọn không hợp lệ!")

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()

