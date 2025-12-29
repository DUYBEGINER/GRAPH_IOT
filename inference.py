"""
Inference Script for GNN-based IoT Anomaly Detection
Script để sử dụng model đã train để dự đoán
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from gnn_models import create_model

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_DIR = r"D:\PROJECT\Machine Learning\IOT\models"
PROCESSED_DATA_DIR = r"processed_data"
GRAPH_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\graph_data"
RESULTS_DIR = r"D:\PROJECT\Machine Learning\IOT\results"

TASK = 'binary'  # 'binary' hoặc 'multi'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# INFERENCE CLASS
# ============================================================================

class GNNPredictor:
    """Class để thực hiện inference với GNN model"""

    def __init__(self, model_path, config_path, device='cpu'):
        self.device = device

        # Load configuration
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)

        # Load metadata
        metadata_path = os.path.join(PROCESSED_DATA_DIR, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Create model
        model_kwargs = {
            'num_layers': self.config['num_layers'],
            'dropout': self.config['dropout']
        }

        if self.config['model_name'] in ['GAT', 'Hybrid']:
            model_kwargs['heads'] = 4  # Default heads

        if self.config['task'] == 'binary':
            num_classes = 2
        else:
            num_classes = self.metadata['n_classes']

        self.model = create_model(
            self.config['model_name'],
            in_channels=self.metadata['n_features'],
            hidden_channels=self.config['hidden_channels'],
            num_classes=num_classes,
            **model_kwargs
        )

        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f"✓ Model loaded: {self.config['model_name']}")
        print(f"✓ Task: {self.config['task']}")
        print(f"✓ Best validation accuracy: {self.config['best_val_acc']:.4f}")

    @torch.no_grad()
    def predict(self, data):
        """
        Dự đoán cho toàn bộ graph

        Args:
            data: PyTorch Geometric Data object

        Returns:
            predictions: Array of predictions
            probabilities: Array of probabilities
        """
        data = data.to(self.device)

        # Forward pass
        out = self.model(data.x, data.edge_index)

        # Get predictions
        pred = out.argmax(dim=1).cpu().numpy()

        # Get probabilities
        probs = F.softmax(out, dim=1).cpu().numpy()

        return pred, probs

    @torch.no_grad()
    def predict_nodes(self, data, node_indices):
        """
        Dự đoán cho các nodes cụ thể

        Args:
            data: PyTorch Geometric Data object
            node_indices: Indices của nodes cần dự đoán

        Returns:
            predictions: Predictions for specified nodes
            probabilities: Probabilities for specified nodes
        """
        data = data.to(self.device)

        # Forward pass
        out = self.model(data.x, data.edge_index)

        # Get predictions for specified nodes
        pred = out[node_indices].argmax(dim=1).cpu().numpy()
        probs = F.softmax(out[node_indices], dim=1).cpu().numpy()

        return pred, probs

    def interpret_predictions(self, predictions, probabilities):
        """
        Giải thích kết quả dự đoán

        Args:
            predictions: Array of class predictions
            probabilities: Array of class probabilities

        Returns:
            results: List of interpretation dictionaries
        """
        results = []

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if self.config['task'] == 'binary':
                label = 'Attack' if pred == 1 else 'Benign'
                confidence = prob[pred]
            else:
                label = self.metadata['class_names'][pred]
                confidence = prob[pred]

            results.append({
                'node_id': i,
                'prediction': label,
                'confidence': confidence,
                'probabilities': prob
            })

        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_inference():
    """Example: Thực hiện inference trên test data"""

    print("=" * 80)
    print("GNN INFERENCE EXAMPLE")
    print("=" * 80)

    # Load graph data
    print("\nLoading graph data...")
    graph_file = f"graph_{TASK}.pt"
    data = torch.load(os.path.join(GRAPH_DATA_DIR, graph_file))
    print(f"✓ Graph loaded: {data.num_nodes:,} nodes")

    # Load predictor
    print("\nLoading model...")
    predictor = GNNPredictor(
        model_path=os.path.join(MODEL_DIR, f'best_model_{TASK}.pt'),
        config_path=os.path.join(RESULTS_DIR, f'config_{TASK}.pkl'),
        device=DEVICE
    )

    # Predict for all nodes
    print("\nMaking predictions...")
    predictions, probabilities = predictor.predict(data)

    print(f"✓ Predictions made for {len(predictions):,} nodes")

    # Analyze predictions
    print("\n" + "-" * 80)
    print("PREDICTION SUMMARY")
    print("-" * 80)

    if TASK == 'binary':
        benign_count = (predictions == 0).sum()
        attack_count = (predictions == 1).sum()

        print(f"Benign (Normal): {benign_count:,} ({benign_count/len(predictions)*100:.2f}%)")
        print(f"Attack (Anomaly): {attack_count:,} ({attack_count/len(predictions)*100:.2f}%)")
    else:
        unique, counts = np.unique(predictions, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = predictor.metadata['class_names'][class_id]
            print(f"{class_name}: {count:,} ({count/len(predictions)*100:.2f}%)")

    # Show some example predictions
    print("\n" + "-" * 80)
    print("EXAMPLE PREDICTIONS (First 10 nodes)")
    print("-" * 80)

    sample_indices = range(min(10, len(predictions)))
    results = predictor.interpret_predictions(
        predictions[sample_indices],
        probabilities[sample_indices]
    )

    for result in results:
        print(f"\nNode {result['node_id']}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")

    # Compare with ground truth
    if hasattr(data, 'y'):
        print("\n" + "-" * 80)
        print("ACCURACY ON FULL GRAPH")
        print("-" * 80)

        true_labels = data.y.cpu().numpy()
        accuracy = (predictions == true_labels).mean()

        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Count correct predictions by class
        if TASK == 'binary':
            benign_correct = ((predictions == 0) & (true_labels == 0)).sum()
            benign_total = (true_labels == 0).sum()
            attack_correct = ((predictions == 1) & (true_labels == 1)).sum()
            attack_total = (true_labels == 1).sum()

            print(f"\nBenign Detection Rate: {benign_correct/benign_total:.4f} ({benign_correct}/{benign_total})")
            print(f"Attack Detection Rate: {attack_correct/attack_total:.4f} ({attack_correct}/{attack_total})")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80 + "\n")

    return predictions, probabilities, results


def predict_new_samples(X_new, edge_index_new):
    """
    Dự đoán cho dữ liệu mới

    Args:
        X_new: New node features (n_nodes, n_features)
        edge_index_new: New edge indices (2, n_edges)

    Returns:
        predictions, probabilities
    """

    # Load predictor
    predictor = GNNPredictor(
        model_path=os.path.join(MODEL_DIR, f'best_model_{TASK}.pt'),
        config_path=os.path.join(RESULTS_DIR, f'config_{TASK}.pkl'),
        device=DEVICE
    )

    # Create Data object
    from torch_geometric.data import Data

    x = torch.tensor(X_new, dtype=torch.float)
    edge_index = torch.tensor(edge_index_new, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    # Predict
    predictions, probabilities = predictor.predict(data)

    # Interpret
    results = predictor.interpret_predictions(predictions, probabilities)

    return predictions, probabilities, results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run example inference
    predictions, probabilities, results = example_inference()

