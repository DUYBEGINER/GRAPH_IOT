import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from pathlib import Path
import sys

import streamlit as st
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from sklearn.neighbors import NearestNeighbors

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# adjust based on trained model location
MODEL_DIR = "GNN_Flow/gnn_flow_output"


def normalize_label(series: pd.Series):
    """Convert label column to binary (0=Benign, 1=Attack)."""
    s = series.astype(str).str.strip().str.upper()
    return np.where(s == "BENIGN", 0, 1).astype(np.int64)

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


@torch.no_grad()
def infer_window(model, x_tensor, edge_index, device, threshold=0.5):
    """Run inference on a window of flows (node classification).
    
    Args:
        model: GraphSAGE model
        x_tensor: Node features [num_nodes, in_dim] - each node is a flow
        edge_index: Edge indices [2, num_edges] - KNN connections
        device: Device to run on
        threshold: Classification threshold
    
    Returns:
        pred: Binary predictions [num_nodes]
        prob_attack: Attack probabilities [num_nodes]
    """
    model.eval()
    # Force CPU to avoid device mismatch
    data = Data(x=x_tensor.cpu(), edge_index=edge_index.cpu())
    
    # Forward pass - returns logits [num_nodes, num_classes]
    logits = model(data.x, data.edge_index)
    
    # Softmax to get probabilities, take class 1 (attack) probability
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pred = (probs >= threshold).astype(np.int64)
    
    return pred, probs


def build_knn_graph(X, k=5):
    """Build KNN graph from feature matrix.
    
    Args:
        X: Feature matrix [num_samples, num_features]
        k: Number of nearest neighbors
    
    Returns:
        edge_index: Edge indices [2, num_edges]
    """
    n_samples = len(X)
    
    # Handle small graphs
    if n_samples <= k:
        k = max(1, n_samples - 1)
    
    # Build KNN graph using sklearn
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='auto')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Build edge list
    edges_src = []
    edges_dst = []
    
    for i in range(n_samples):
        for j in indices[i][1:]:  # Skip first (self)
            if j < n_samples:  # Valid index
                edges_src.append(i)
                edges_dst.append(j)
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index


@st.cache_resource
def load_artifacts(model_dir: str, device: str, 
                   hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3):
    """Load model checkpoint, scaler, and features. Cached to prevent reloading."""
    model_path = Path(model_dir)
    
    # Load scaler from model dir
    with open(model_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load features from model dir (try .pkl first, then .json)
    feature_cols = None
    if (model_path / "feature_names.pkl").exists():
        with open(model_path / "feature_names.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    elif (model_path / "feature_names.json").exists():
        with open(model_path / "feature_names.json", "r") as f:
            feature_data = json.load(f)
            if isinstance(feature_data, dict) and "features" in feature_data:
                feature_cols = feature_data["features"]
            else:
                feature_cols = feature_data
    else:
        raise FileNotFoundError("feature_names.pkl or feature_names.json not found in model directory")
    
    # Load checkpoint
    ckpt = torch.load(model_path / "best_model.pt", map_location="cpu", weights_only=False)
    
    # Get state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        in_dim = ckpt.get("in_channels", len(feature_cols))
        hidden_dim = ckpt.get("hidden_channels", hidden_dim)
        num_layers = ckpt.get("num_layers", num_layers)
    else:
        state_dict = ckpt
        # Detect input dimension from first layer
        first_layer_key = "convs.0.lin_l.weight"
        if first_layer_key in state_dict:
            in_dim = state_dict[first_layer_key].shape[1]
        else:
            in_dim = len(feature_cols)
    
    # Adjust feature_cols to match model input dimension
    if len(feature_cols) > in_dim:
        feature_cols = feature_cols[:in_dim]
    elif len(feature_cols) < in_dim:
        # Pad with dummy features
        for i in range(in_dim - len(feature_cols)):
            feature_cols.append(f"feature_{len(feature_cols) + i}")
    
    # Adjust scaler to match model input dimension
    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != in_dim:
        from sklearn.preprocessing import StandardScaler
        new_scaler = StandardScaler()
        
        if hasattr(scaler, 'mean_') and len(scaler.mean_) >= in_dim:
            new_scaler.mean_ = scaler.mean_[:in_dim]
            new_scaler.scale_ = scaler.scale_[:in_dim]
            new_scaler.var_ = scaler.var_[:in_dim] if hasattr(scaler, 'var_') else None
        else:
            # Create default scaler if dimension mismatch
            new_scaler.mean_ = np.zeros(in_dim)
            new_scaler.scale_ = np.ones(in_dim)
            new_scaler.var_ = np.ones(in_dim)
        
        new_scaler.n_features_in_ = in_dim
        new_scaler.n_samples_seen_ = scaler.n_samples_seen_ if hasattr(scaler, 'n_samples_seen_') else in_dim
        scaler = new_scaler
    
    # Create GraphSAGE model - Force CPU for stability
    model = GraphSAGE(
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout
    ).to("cpu")  # Force CPU
    
    # Load weights
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model, scaler, feature_cols, in_dim


def main():
    st.set_page_config(
        page_title="Flow-GNN Realtime Demo", 
        layout="wide"
    )
    st.title("🛡️ IoT Flow-GNN Anomaly Detection – Realtime Demo")
    st.caption("Flow-GNN (GraphSAGE): Node=flow, Edge=KNN similarity, Task=node classification")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Paths
        model_dir = st.text_input("Model directory", value=MODEL_DIR)
        
        # Model params
        st.subheader("Model Parameters")
        hidden_dim = st.number_input("Hidden dim", value=128, min_value=32, max_value=512)
        num_layers = st.number_input("Num layers", value=3, min_value=1, max_value=5)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.3, step=0.05)
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, step=0.05)
        
        # Graph params
        st.subheader("Graph Settings")
        k_neighbors = st.slider("K neighbors", 3, 20, 5, step=1)
        
        # Inference params
        st.subheader("Inference Settings")
        label_col = st.text_input("Label column (optional)", value="Label")
        chunk_size = st.slider("Chunk size", 200, 5000, 2000, step=100)
        window_size = st.slider("Window size", 1000, 30000, 8000, step=1000)
        speed = st.slider("Speed (sleep seconds per chunk)", 0.0, 1.0, 0.2, step=0.05)

        device = "cpu"
        st.caption(f"🔧 Device: {device} (CPU forced for stability)")

    st.info("💡 **Tip:** For large files, upload may take time. Config allows up to 400MB uploads.")
    
    uploaded = st.file_uploader(
        "📁 Upload CSV for testing (e.g., Thuesday-20-02-2018...)", 
        type=["csv"]
    )

    if uploaded is None:
        st.info("📤 Please upload a CSV file to start the demo.")
        return

    try:
        model, scaler, feature_cols, in_dim = load_artifacts(
            model_dir, device, 
            hidden_dim, num_layers, dropout
        )
        st.success(f"✅ Model loaded: {in_dim} features")
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    start = st.button("▶️ Start Realtime Detection", type="primary")
    if not start:
        return

    # UI placeholders
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_acc = st.empty()
    with col2:
        metric_rate = st.empty()
    with col3:
        metric_rows = st.empty()

    st.subheader("📊 Latest Predictions")
    table_ph = st.empty()
    
    st.subheader("📈 Performance Over Time")
    chart_ph = st.empty()

    # Streaming state
    feat_buf = []
    y_buf = []
    has_label = False
    correct = 0
    total = 0
    rows_seen = 0
    history = []
    
    # Attack detection logs
    attack_success_logs = []  # Predicted Attack & Actually Attack (True Positive)
    attack_failed_logs = []   # Predicted Attack but Actually Benign (False Positive)
    missed_attack_logs = []   # Predicted Benign but Actually Attack (False Negative)
    
    # Additional UI for attack logs
    st.subheader("🔴 Attack Detection Logs")
    col_log1, col_log2, col_log3 = st.columns(3)
    with col_log1:
        metric_tp = st.empty()
    with col_log2:
        metric_fp = st.empty()
    with col_log3:
        metric_fn = st.empty()
    
    tab_tp, tab_fp, tab_fn = st.tabs([
        "✅ Detected Attacks (True Positive)", 
        "⚠️ False Alarms (False Positive)", 
        "❌ Missed Attacks (False Negative)"
    ])
    with tab_tp:
        table_tp = st.empty()
    with tab_fp:
        table_fp = st.empty()
    with tab_fn:
        table_fn = st.empty()

    # Stream CSV
    try:
        for chunk in pd.read_csv(uploaded, low_memory=False, chunksize=chunk_size):
            rows_seen += len(chunk)

            # Ensure feature alignment
            for c in feature_cols:
                if c not in chunk.columns:
                    chunk[c] = 0

            X = chunk[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            X_scaled = scaler.transform(X)

            # Get labels if available
            if label_col in chunk.columns and chunk[label_col].notna().any():
                y = normalize_label(chunk[label_col])
                has_label = True
            else:
                y = None

            # Append to window buffer
            feat_buf.extend(list(X_scaled))
            if has_label and y is not None:
                y_buf.extend(list(y))

            # Maintain window size
            if len(feat_buf) > window_size:
                cut = len(feat_buf) - window_size
                feat_buf = feat_buf[cut:]
                if has_label:
                    y_buf = y_buf[cut:]

            # Build graph for Flow-GNN (GraphSAGE)
            # For node classification: each flow is a node, edges are KNN connections
            Xw = np.asarray(feat_buf, dtype=np.float32)
            num_flows = len(Xw)
            
            # Node features: actual flow features
            x_tensor = torch.from_numpy(Xw).float()
            
            # Build KNN graph
            edge_index = build_knn_graph(Xw, k=k_neighbors)

            # Run inference
            pred_w, prob_w = infer_window(model, x_tensor, edge_index, device, threshold)

            # Get predictions for new chunk
            new_n = min(chunk_size, len(pred_w))
            pred_new = pred_w[-new_n:]
            prob_new = prob_w[-new_n:]

            # Calculate accuracy if labels available
            if has_label and len(y_buf) >= new_n:
                y_new = np.asarray(y_buf[-new_n:], dtype=np.int64)
                correct += int((pred_new == y_new).sum())
                total += int(new_n)
                acc = correct / max(total, 1)
                metric_acc.metric("Accuracy (cumulative)", f"{acc:.4f}")
                
                # Log attack detections with full record info
                chunk_tail = chunk.tail(new_n).reset_index(drop=True)
                for i in range(len(pred_new)):
                    predicted_attack = pred_new[i] == 1
                    actual_attack = y_new[i] == 1
                    
                    if predicted_attack or actual_attack:
                        # Build log record with all info
                        record_info = chunk_tail.iloc[i].to_dict()
                        log_entry = {
                            "row_index": rows_seen - new_n + i,
                            "prob_attack": float(prob_new[i]),
                            "predicted": "ATTACK" if predicted_attack else "BENIGN",
                            "actual": "ATTACK" if actual_attack else "BENIGN",
                            **record_info
                        }
                        
                        if predicted_attack and actual_attack:
                            # True Positive: Correctly detected attack
                            attack_success_logs.append(log_entry)
                        elif predicted_attack and not actual_attack:
                            # False Positive: False alarm
                            attack_failed_logs.append(log_entry)
                        elif not predicted_attack and actual_attack:
                            # False Negative: Missed attack
                            missed_attack_logs.append(log_entry)
                
                # Update attack log metrics
                metric_tp.metric("✅ Detected Attacks (TP)", len(attack_success_logs))
                metric_fp.metric("⚠️ False Alarms (FP)", len(attack_failed_logs))
                metric_fn.metric("❌ Missed Attacks (FN)", len(missed_attack_logs))
                
                # Display attack logs (show last 50 entries)
                display_cols_log = ["row_index", "prob_attack", "predicted", "actual"] + \
                                   [c for c in feature_cols[:5]] + ([label_col] if label_col in chunk.columns else [])
                
                if attack_success_logs:
                    df_tp = pd.DataFrame(attack_success_logs[-50:])
                    # Select only existing columns
                    display_cols_tp = [c for c in display_cols_log if c in df_tp.columns]
                    table_tp.dataframe(df_tp[display_cols_tp], use_container_width=True)
                
                if attack_failed_logs:
                    df_fp = pd.DataFrame(attack_failed_logs[-50:])
                    display_cols_fp = [c for c in display_cols_log if c in df_fp.columns]
                    table_fp.dataframe(df_fp[display_cols_fp], use_container_width=True)
                    
                if missed_attack_logs:
                    df_fn = pd.DataFrame(missed_attack_logs[-50:])
                    display_cols_fn = [c for c in display_cols_log if c in df_fn.columns]
                    table_fn.dataframe(df_fn[display_cols_fn], use_container_width=True)
            else:
                metric_acc.metric("Accuracy (cumulative)", "N/A")

            attack_rate = float(pred_new.mean())
            metric_rate.metric("Attack Rate (latest chunk)", f"{attack_rate:.3f}")
            metric_rows.metric("Rows Processed", f"{rows_seen:,}")

            # Show latest predictions
            out = chunk.tail(min(30, len(chunk))).copy()
            out["prob_attack"] = prob_new[-len(out):]
            out["pred"] = np.where(out["prob_attack"] >= threshold, "ATTACK", "BENIGN")
            
            display_cols = ["prob_attack", "pred"]
            if label_col in out.columns:
                display_cols.append(label_col)
            table_ph.dataframe(out[display_cols], use_container_width=True)

            # Update history
            acc_val = acc if (has_label and total > 0) else np.nan
            history.append({
                "rows_seen": rows_seen, 
                "attack_rate": attack_rate, 
                "acc": acc_val
            })
            hist_df = pd.DataFrame(history)
            chart_ph.line_chart(hist_df.set_index("rows_seen")[["attack_rate", "acc"]])

            time.sleep(speed)

        st.success("✅ Done streaming CSV!")
        
        # Save attack logs to CSV files
        st.subheader("💾 Download Attack Logs")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            if attack_success_logs:
                df_tp_final = pd.DataFrame(attack_success_logs)
                csv_tp = df_tp_final.to_csv(index=False)
                st.download_button(
                    label=f"📥 Detected Attacks ({len(attack_success_logs)} records)",
                    data=csv_tp,
                    file_name="attack_detected_true_positive.csv",
                    mime="text/csv"
                )
            else:
                st.info("No True Positive records")
        
        with col_dl2:
            if attack_failed_logs:
                df_fp_final = pd.DataFrame(attack_failed_logs)
                csv_fp = df_fp_final.to_csv(index=False)
                st.download_button(
                    label=f"📥 False Alarms ({len(attack_failed_logs)} records)",
                    data=csv_fp,
                    file_name="attack_false_positive.csv",
                    mime="text/csv"
                )
            else:
                st.info("No False Positive records")
        
        with col_dl3:
            if missed_attack_logs:
                df_fn_final = pd.DataFrame(missed_attack_logs)
                csv_fn = df_fn_final.to_csv(index=False)
                st.download_button(
                    label=f"📥 Missed Attacks ({len(missed_attack_logs)} records)",
                    data=csv_fn,
                    file_name="attack_missed_false_negative.csv",
                    mime="text/csv"
                )
            else:
                st.info("No False Negative records")
        
        # Summary statistics
        st.subheader("📊 Detection Summary")
        total_attacks = len(attack_success_logs) + len(missed_attack_logs)
        total_predicted = len(attack_success_logs) + len(attack_failed_logs)
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            if total_attacks > 0:
                recall = len(attack_success_logs) / total_attacks
                st.metric("Recall (Attack Detection Rate)", f"{recall:.4f}")
            else:
                st.metric("Recall", "N/A")
        
        with col_sum2:
            if total_predicted > 0:
                precision = len(attack_success_logs) / total_predicted
                st.metric("Precision", f"{precision:.4f}")
            else:
                st.metric("Precision", "N/A")
        
        with col_sum3:
            if total_attacks > 0 and total_predicted > 0:
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                st.metric("F1-Score", f"{f1:.4f}")
            else:
                st.metric("F1-Score", "N/A")
        
    except Exception as e:
        st.error(f"❌ Error during inference: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
