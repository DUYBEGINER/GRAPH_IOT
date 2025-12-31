"""
Demo/Inference Script for IP-based GraphSAGE Model
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
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "models/best_model.pt"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
IP_ENCODER_PATH = "ip_encoder.pkl"


# ============================================================================
# MODEL DEFINITION
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


class IPGNNInference:
    """Inference class for IP-based GNN model."""

    def __init__(self, model_dir, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None

        self._load_model()
        self._load_preprocessors()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, MODEL_PATH)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        in_channels = checkpoint.get('in_channels', 228)  # 76 features * 3 aggregations
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
        scaler_path = os.path.join(self.model_dir, SCALER_PATH)
        features_path = os.path.join(self.model_dir, FEATURE_NAMES_PATH)

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded")

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

        # Get IP columns
        if 'Src IP' not in df.columns or 'Dst IP' not in df.columns:
            raise ValueError("CSV must contain 'Src IP' and 'Dst IP' columns")

        src_ips = df['Src IP'].values
        dst_ips = df['Dst IP'].values

        # Get labels if available
        labels = None
        if 'Label' in df.columns:
            labels = (df['Label'] != 'Benign').astype(int).values

        # Select features
        if self.feature_names:
            available = [f for f in self.feature_names if f in df.columns]
            feature_df = df[available].copy()
        else:
            exclude = ['Label', 'Src IP', 'Dst IP', 'Timestamp', 'Flow ID']
            feature_df = df.drop(columns=[c for c in exclude if c in df.columns])

        # Convert to numeric
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

        # Handle missing/inf
        feature_df = feature_df.fillna(0).replace([np.inf, -np.inf], 0)

        X = feature_df.values

        # Scale
        if self.scaler:
            X = self.scaler.transform(X)

        return X, labels, src_ips, dst_ips

    def build_ip_graph(self, X, src_ips, dst_ips):
        """Build IP graph from flows."""
        print("Building IP graph...")

        # Create IP mapping
        all_ips = np.unique(np.concatenate([src_ips, dst_ips]))
        ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        n_ips = len(all_ips)

        src_idx = np.array([ip_to_idx[ip] for ip in src_ips])
        dst_idx = np.array([ip_to_idx[ip] for ip in dst_ips])

        # Aggregate features per IP
        n_features = X.shape[1]
        ip_features = defaultdict(list)

        for i in range(len(X)):
            ip_features[src_idx[i]].append(X[i])
            ip_features[dst_idx[i]].append(X[i])

        node_features = np.zeros((n_ips, n_features * 3))
        for ip in range(n_ips):
            if ip in ip_features:
                feats = np.array(ip_features[ip])
                node_features[ip, :n_features] = feats.mean(axis=0)
                node_features[ip, n_features:n_features*2] = feats.std(axis=0)
                node_features[ip, n_features*2:] = feats.max(axis=0)

        node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create edges
        edge_set = set()
        for i in range(len(src_idx)):
            if src_idx[i] != dst_idx[i]:
                edge_set.add((src_idx[i], dst_idx[i]))
                edge_set.add((dst_idx[i], src_idx[i]))

        edges = list(edge_set)
        edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)

        print(f"IPs: {n_ips:,}, Edges: {edge_index.shape[1]:,}")

        return node_features, edge_index, ip_to_idx, all_ips

    @torch.no_grad()
    def predict(self, node_features, edge_index):
        x = torch.tensor(node_features, dtype=torch.float).to(self.device)
        edge_index = edge_index.to(self.device)

        start_time = time.time()
        out = self.model(x, edge_index)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        inference_time = time.time() - start_time

        return preds.cpu().numpy(), probs.cpu().numpy(), inference_time

    def evaluate(self, csv_path):
        """Full evaluation pipeline."""
        X, labels, src_ips, dst_ips = self.preprocess_csv(csv_path)
        node_features, edge_index, ip_to_idx, all_ips = self.build_ip_graph(X, src_ips, dst_ips)
        preds, probs, inference_time = self.predict(node_features, edge_index)

        print(f"\nInference time: {inference_time*1000:.2f} ms")

        # Summary
        benign_ips = (preds == 0).sum()
        attack_ips = (preds == 1).sum()
        print(f"\nIP Predictions:")
        print(f"  Benign IPs: {benign_ips:,}")
        print(f"  Attack IPs: {attack_ips:,}")

        results = {
            'ip_predictions': {str(ip): int(preds[ip_to_idx[ip]]) for ip in all_ips},
            'probabilities': probs.tolist(),
            'inference_time_ms': inference_time * 1000,
            'n_ips': len(all_ips)
        }

        # If labels available, compute flow-level metrics
        if labels is not None:
            # Map IP predictions back to flows
            flow_preds = []
            for i in range(len(src_ips)):
                src = ip_to_idx[src_ips[i]]
                dst = ip_to_idx[dst_ips[i]]
                # If either IP is attack, flow is attack
                flow_preds.append(1 if (preds[src] == 1 or preds[dst] == 1) else 0)

            flow_preds = np.array(flow_preds)

            acc = accuracy_score(labels, flow_preds)
            prec = precision_score(labels, flow_preds, zero_division=0)
            rec = recall_score(labels, flow_preds, zero_division=0)
            f1 = f1_score(labels, flow_preds, zero_division=0)

            cm = confusion_matrix(labels, flow_preds)

            print(f"\nFlow-level Metrics (from IP predictions):")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
            print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

            results.update({
                'flow_accuracy': acc,
                'flow_precision': prec,
                'flow_recall': rec,
                'flow_f1_score': f1,
                'confusion_matrix': cm.tolist()
            })

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='IP-based GNN Inference')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing model')
    parser.add_argument('--csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--output', type=str, default='ip_predictions.json', help='Output file')

    args = parser.parse_args()

    print("=" * 60)
    print("IP-BASED GNN INFERENCE")
    print("=" * 60)

    inferencer = IPGNNInference(args.model_dir)
    results = inferencer.evaluate(args.csv)

    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

