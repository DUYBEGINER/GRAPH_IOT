"""
Demo/Inference Script for E-GraphSAGE Model (Edge Classification)
Run locally with test dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "models/best_model.pt"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
IP_ENCODER_PATH = "ip_encoder.pkl"

# Column names
SRC_IP_COL = "Src IP"
DST_IP_COL = "Dst IP"
LABEL_COL = "Label"


# ============================================================================
# MODEL DEFINITION: E-GraphSAGE
# ============================================================================
class EdgeFeatureSAGEConv(nn.Module):
    """SAGEConv layer that incorporates edge features during aggregation."""
    
    def __init__(self, in_dim, out_dim, in_edge_dim, aggr="mean"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_edge_dim = in_edge_dim
        self.aggr = aggr
        
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(out_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(in_edge_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        num_nodes = x.size(0)
        
        out_self = self.lin_self(x)
        edge_projected = self.lin_edge(edge_attr)
        
        aggregated = torch.zeros(num_nodes, self.out_dim, device=x.device)
        
        if self.aggr == "mean":
            ones = torch.ones(edge_index.size(1), device=x.device)
            degree = torch.zeros(num_nodes, device=x.device)
            degree.scatter_add_(0, dst, ones)
            degree = degree.clamp(min=1)
            
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_projected), edge_projected)
            aggregated = aggregated / degree.unsqueeze(1)
        elif self.aggr == "sum":
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_projected), edge_projected)
        
        out_neigh = self.lin_neigh(aggregated)
        out = out_self + out_neigh + self.bias
        
        return out


class EGraphSAGE(nn.Module):
    """E-GraphSAGE for edge classification."""
    
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.2, aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.in_edge_dim = in_dim
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(EdgeFeatureSAGEConv(in_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(EdgeFeatureSAGEConv(hidden_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, edge_attr, edge_label_index=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        if edge_label_index is None:
            edge_label_index = edge_index
        
        src_emb = h[edge_label_index[0]]
        dst_emb = h[edge_label_index[1]]
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        
        return self.edge_classifier(edge_emb)


class EGraphSAGEInference:
    """Inference class for E-GraphSAGE model (Edge Classification)."""

    def __init__(self, model_dir, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5

        self._load_model()
        self._load_preprocessors()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, MODEL_PATH)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        in_dim = checkpoint.get('in_dim', 76)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_layers = checkpoint.get('num_layers', 2)

        self.model = EGraphSAGE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = checkpoint.get('best_threshold', 0.5)

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Threshold: {self.threshold:.4f}")

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
        if LABEL_COL in df.columns:
            df = df[df[LABEL_COL] != 'Label'].copy()

        # Get IP columns
        if SRC_IP_COL not in df.columns or DST_IP_COL not in df.columns:
            raise ValueError(f"CSV must contain '{SRC_IP_COL}' and '{DST_IP_COL}' columns")

        src_ips = df[SRC_IP_COL].values
        dst_ips = df[DST_IP_COL].values

        # Get labels if available
        labels = None
        if LABEL_COL in df.columns:
            labels = (df[LABEL_COL] != 'Benign').astype(int).values

        # Select features
        if self.feature_names:
            available = [f for f in self.feature_names if f in df.columns]
            feature_df = df[available].copy()
        else:
            exclude = [LABEL_COL, SRC_IP_COL, DST_IP_COL, 'Timestamp', 'Flow ID']
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

    def build_endpoint_graph(self, X, src_ips, dst_ips):
        """Build endpoint graph for E-GraphSAGE."""
        print("Building endpoint graph...")

        # Create IP mapping
        all_ips = np.unique(np.concatenate([src_ips, dst_ips]))
        ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        n_nodes = len(all_ips)

        src_idx = np.array([ip_to_idx[ip] for ip in src_ips])
        dst_idx = np.array([ip_to_idx[ip] for ip in dst_ips])

        # Edge index
        edge_index = torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long)
        
        # Edge features
        edge_attr = torch.tensor(X, dtype=torch.float)
        
        # Node features (ones vector as per E-GraphSAGE)
        n_features = X.shape[1]
        node_features = torch.ones((n_nodes, n_features), dtype=torch.float)

        print(f"Nodes (IPs): {n_nodes:,}, Edges (Flows): {edge_index.shape[1]:,}")

        return node_features, edge_index, edge_attr, ip_to_idx, all_ips

    @torch.no_grad()
    def predict(self, node_features, edge_index, edge_attr):
        x = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        start_time = time.time()
        logits = self.model(x, edge_index, edge_attr)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= self.threshold).astype(int)
        inference_time = time.time() - start_time

        return preds, probs, inference_time

    def evaluate(self, csv_path):
        """Full evaluation pipeline for edge classification."""
        X, labels, src_ips, dst_ips = self.preprocess_csv(csv_path)
        node_features, edge_index, edge_attr, ip_to_idx, all_ips = self.build_endpoint_graph(X, src_ips, dst_ips)
        preds, probs, inference_time = self.predict(node_features, edge_index, edge_attr)

        print(f"\nInference time: {inference_time*1000:.2f} ms")
        print(f"Latency: {(inference_time*1000)/len(preds):.4f} ms/flow")

        # Summary
        benign_flows = (preds == 0).sum()
        attack_flows = (preds == 1).sum()
        print(f"\nFlow Predictions:")
        print(f"  Benign flows: {benign_flows:,}")
        print(f"  Attack flows: {attack_flows:,}")

        results = {
            'flow_predictions': preds.tolist(),
            'probabilities': probs.tolist(),
            'inference_time_ms': inference_time * 1000,
            'latency_ms_per_flow': (inference_time * 1000) / len(preds),
            'n_flows': len(preds),
            'n_ips': len(all_ips),
            'threshold': self.threshold
        }

        # If labels available, compute flow-level metrics
        if labels is not None:
            acc = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)

            try:
                auc_score = roc_auc_score(labels, probs)
            except:
                auc_score = 0.0

            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            print(f"\nFlow-level Metrics:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC:       {auc_score:.4f}")
            print(f"  FAR:       {far:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  TN: {tn:,}  FP: {fp:,}")
            print(f"  FN: {fn:,}  TP: {tp:,}")

            results.update({
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'auc': auc_score,
                'far': far,
                'confusion_matrix': cm.tolist()
            })

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='E-GraphSAGE Inference (Edge Classification)')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing model')
    parser.add_argument('--csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--output', type=str, default='flow_predictions.json', help='Output file')

    args = parser.parse_args()

    print("=" * 60)
    print("E-GRAPHSAGE INFERENCE (Edge Classification)")
    print("=" * 60)

    inferencer = EGraphSAGEInference(args.model_dir)
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

