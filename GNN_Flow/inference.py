"""
Demo/Inference Script for Flow-based GraphSAGE Model
Run locally with test dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
import numpy as np
import pickle
import os
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "models/best_model.pt"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
K_NEIGHBORS = 5


# ============================================================================
# MODEL DEFINITION (same as training)
# ============================================================================
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class FlowGNNInference:
    """Inference class for Flow-based GNN model."""

    def __init__(self, model_dir, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None

        self._load_model()
        self._load_preprocessors()

    def _load_model(self):
        """Load trained model."""
        model_path = os.path.join(self.model_dir, MODEL_PATH)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get model config from checkpoint or use defaults
        in_channels = checkpoint.get('in_channels', 76)  # Default from CICIDS2018
        hidden_channels = checkpoint.get('hidden_channels', 128)
        num_layers = checkpoint.get('num_layers', 3)

        self.model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=2,
            num_layers=num_layers
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")

    def _load_preprocessors(self):
        """Load scaler and feature names."""
        scaler_path = os.path.join(self.model_dir, SCALER_PATH)
        features_path = os.path.join(self.model_dir, FEATURE_NAMES_PATH)

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded")

        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"Features: {len(self.feature_names)}")

    def preprocess_csv(self, csv_path):
        """Preprocess CSV file for inference."""
        print(f"\nLoading: {csv_path}")

        try:
            df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, low_memory=False, encoding='latin-1')

        print(f"Rows: {len(df):,}")

        # Filter header rows
        if 'Label' in df.columns:
            df = df[df['Label'] != 'Label'].copy()

        # Get labels if available
        labels = None
        if 'Label' in df.columns:
            labels = (df['Label'] != 'Benign').astype(int).values

        # Select features
        if self.feature_names:
            available = [f for f in self.feature_names if f in df.columns]
            df = df[available].copy()

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing/inf values
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        X = df.values

        # Scale
        if self.scaler:
            X = self.scaler.transform(X)

        return X, labels

    def build_graph(self, X):
        """Build KNN graph for inference."""
        print(f"Building KNN graph (k={K_NEIGHBORS})...")

        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, algorithm='auto').fit(X)
        _, indices = nbrs.kneighbors(X)

        edges_src = []
        edges_dst = []

        for i in range(len(X)):
            for j in indices[i][1:]:
                edges_src.append(i)
                edges_dst.append(j)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        print(f"Nodes: {len(X):,}, Edges: {edge_index.shape[1]:,}")
        return edge_index

    @torch.no_grad()
    def predict(self, X, edge_index):
        """Run inference."""
        x = torch.tensor(X, dtype=torch.float).to(self.device)
        edge_index = edge_index.to(self.device)

        start_time = time.time()
        out = self.model(x, edge_index)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        inference_time = time.time() - start_time

        return preds.cpu().numpy(), probs.cpu().numpy(), inference_time

    def evaluate(self, csv_path):
        """Full evaluation pipeline."""
        X, labels = self.preprocess_csv(csv_path)
        edge_index = self.build_graph(X)
        preds, probs, inference_time = self.predict(X, edge_index)

        print(f"\nInference time: {inference_time*1000:.2f} ms ({inference_time*1000/len(X):.4f} ms/sample)")

        # Predictions summary
        benign_pred = (preds == 0).sum()
        attack_pred = (preds == 1).sum()
        print(f"\nPredictions:")
        print(f"  Benign: {benign_pred:,} ({benign_pred/len(preds)*100:.1f}%)")
        print(f"  Attack: {attack_pred:,} ({attack_pred/len(preds)*100:.1f}%)")

        results = {
            'predictions': preds,
            'probabilities': probs,
            'inference_time_ms': inference_time * 1000,
            'n_samples': len(preds)
        }

        # Calculate metrics if labels available
        if labels is not None:
            acc = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)

            try:
                auc = roc_auc_score(labels, probs[:, 1])
            except:
                auc = 0.0

            cm = confusion_matrix(labels, preds)

            print(f"\nMetrics:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC:       {auc:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
            print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

            results.update({
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'auc': auc,
                'confusion_matrix': cm.tolist()
            })

        return results


def main():
    """Demo main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Flow-based GNN Inference')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing model and preprocessors')
    parser.add_argument('--csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output predictions file')

    args = parser.parse_args()

    print("=" * 60)
    print("FLOW-BASED GNN INFERENCE")
    print("=" * 60)

    # Initialize inference
    inferencer = FlowGNNInference(args.model_dir)

    # Run evaluation
    results = inferencer.evaluate(args.csv)

    # Save predictions
    df_out = pd.DataFrame({
        'prediction': results['predictions'],
        'prob_benign': results['probabilities'][:, 0],
        'prob_attack': results['probabilities'][:, 1],
        'label': ['Benign' if p == 0 else 'Attack' for p in results['predictions']]
    })
    df_out.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

