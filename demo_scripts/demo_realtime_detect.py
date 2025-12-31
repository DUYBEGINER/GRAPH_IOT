"""
Realtime IP GNN Anomaly Detection Demo
Uses IP-GNN (E-GraphSAGE) model for streaming inference on CSV data
Node = endpoint (IP), Edge = flow, Task = edge classification
"""

import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
import sys

import streamlit as st
from torch_geometric.data import Data

# Add parent directory to path to import ip_gnn module
sys.path.insert(0, str(Path(__file__).parent.parent))
from ip_gnn.model import EGraphSAGE

MODEL_DIR = "ip_gnn/ip_gnn_result"
DATA_DIR = "dataset-processed/ip_gnn"


def normalize_label(series: pd.Series):
    """Convert label column to binary (0=Benign, 1=Attack)."""
    s = series.astype(str).str.strip().str.upper()
    return np.where(s == "BENIGN", 0, 1).astype(np.int64)


@torch.no_grad()
def infer_window(model, x_tensor, edge_index, edge_attr, device, threshold=0.5):
    """Run inference on a window of flows (edge classification).
    
    Args:
        model: EGraphSAGE model
        x_tensor: Node features [num_nodes, in_dim]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge features [num_edges, in_dim]
        device: Device to run on
        threshold: Classification threshold
    
    Returns:
        pred: Binary predictions [num_edges]
        prob_attack: Attack probabilities [num_edges]
    """
    model.eval()
    # Force CPU to avoid device mismatch
    data = Data(x=x_tensor.cpu(), edge_index=edge_index.cpu(), edge_attr=edge_attr.cpu())
    
    # Forward pass - returns logits [num_edges, num_classes]
    logits = model(data.x, data.edge_index, data.edge_attr)
    
    # Softmax to get probabilities, take class 1 (attack) probability
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pred = (probs >= threshold).astype(np.int64)
    
    return pred, probs


@st.cache_resource
def load_artifacts(model_dir: str, data_dir: str, device: str, 
                   hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
    """Load model checkpoint, scaler, and features. Cached to prevent reloading."""
    model_path = Path(model_dir)
    data_path = Path(data_dir)
    
    # Load feature names
    with open(data_path / "feature_names.json", "r", encoding="utf-8") as f:
        feature_data = json.load(f)
        feature_cols = feature_data["features"]
    
    # Load scaler
    with open(data_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load checkpoint
    ckpt = torch.load(model_path / "best_model.pt", map_location="cpu", weights_only=False)
    
    # Get state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    # Detect input dimension from checkpoint (edge features)
    # For EGraphSAGE, first layer is EdgeFeatureSAGEConv with lin_edge
    first_layer_key = "convs.0.lin_edge.weight"
    in_dim = len(feature_cols)  # Default
    
    if first_layer_key in state_dict:
        # Shape is [out_dim, in_edge_dim] for lin_edge
        in_dim = state_dict[first_layer_key].shape[1]
        
        # Only use first in_dim features
        feature_cols = feature_cols[:in_dim]
        
        # Adjust scaler to match model input dimension
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != in_dim:
            from sklearn.preprocessing import StandardScaler
            new_scaler = StandardScaler()
            new_scaler.mean_ = scaler.mean_[:in_dim]
            new_scaler.scale_ = scaler.scale_[:in_dim]
            new_scaler.var_ = scaler.var_[:in_dim] if hasattr(scaler, 'var_') else None
            new_scaler.n_features_in_ = in_dim
            new_scaler.n_samples_seen_ = scaler.n_samples_seen_ if hasattr(scaler, 'n_samples_seen_') else in_dim
            scaler = new_scaler
    
    # Create EGraphSAGE model - ALWAYS use CPU to avoid macOS MPS segfault
    model = EGraphSAGE(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout,
        aggr="mean"
    ).to("cpu")  # Force CPU
    
    # Load weights
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model, scaler, feature_cols, in_dim


def main():
    # Increase upload limit to 400MB
    try:
        import streamlit.config as st_config
        st_config.set_option('server.maxUploadSize', 400)
    except:
        pass
    
    st.set_page_config(
        page_title="IP-GNN Realtime Demo", 
        layout="wide"
    )
    st.title("🛡️ IoT IP-GNN Anomaly Detection – Realtime Demo")
    st.caption("IP-GNN (E-GraphSAGE): Node=endpoint (IP), Edge=flow, Task=edge classification")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Paths
        model_dir = st.text_input("Model directory", value=MODEL_DIR)
        data_dir = st.text_input("Data directory", value=DATA_DIR)
        
        # Model params
        st.subheader("Model Parameters")
        hidden_dim = st.number_input("Hidden dim", value=128, min_value=32, max_value=512)
        num_layers = st.number_input("Num layers", value=2, min_value=1, max_value=5)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.3, step=0.05)
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, step=0.05)
        
        # Inference params
        st.subheader("Inference Settings")
        label_col = st.text_input("Label column (optional)", value="Label")
        chunk_size = st.slider("Chunk size", 200, 5000, 2000, step=100)
        window_size = st.slider("Window size", 1000, 30000, 8000, step=1000)
        speed = st.slider("Speed (sleep seconds per chunk)", 0.0, 1.0, 0.2, step=0.05)

        # Force CPU to avoid macOS MPS segfault issues
        device = "cpu"
        st.caption(f"🔧 Device: {device} (CPU forced for stability)")

    uploaded = st.file_uploader(
        "📁 Upload CSV for testing (e.g., Thuesday-20-02-2018...)", 
        type=["csv"]
    )

    if uploaded is None:
        st.info("📤 Please upload a CSV file to start the demo.")
        return

    try:
        model, scaler, feature_cols, in_dim = load_artifacts(
            model_dir, data_dir, device, 
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

            # Build graph for IP-GNN (E-GraphSAGE)
            # For edge classification: nodes are dummy (ones), edges are flows
            Xw = np.asarray(feat_buf, dtype=np.float32)
            num_flows = len(Xw)
            
            # Create dummy node features (all ones) - nodes don't have real features
            # We need at least num_flows nodes for edge_index
            num_nodes = num_flows
            x_tensor = torch.ones((num_nodes, Xw.shape[1]), dtype=torch.float)
            
            # Edge index: each flow is an edge between dummy nodes
            # Simple mapping: flow i connects node i to node (i+1) % num_nodes
            edge_src = np.arange(num_flows)
            edge_dst = (np.arange(num_flows) + 1) % num_nodes
            edge_index = torch.from_numpy(np.vstack([edge_src, edge_dst])).long()
            
            # Edge attributes: actual flow features
            edge_attr = torch.from_numpy(Xw).float()

            pred_w, prob_w = infer_window(model, x_tensor, edge_index, edge_attr, device, threshold)

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
                                   [c for c in feature_cols[:5]] + [label_col] if label_col in chunk.columns else ["row_index", "prob_attack", "predicted", "actual"]
                
                if attack_success_logs:
                    df_tp = pd.DataFrame(attack_success_logs[-50:])
                    table_tp.dataframe(df_tp, use_container_width=True)
                
                if attack_failed_logs:
                    df_fp = pd.DataFrame(attack_failed_logs[-50:])
                    table_fp.dataframe(df_fp, use_container_width=True)
                    
                if missed_attack_logs:
                    df_fn = pd.DataFrame(missed_attack_logs[-50:])
                    table_fn.dataframe(df_fn, use_container_width=True)
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
        
        if total_attacks > 0:
            recall = len(attack_success_logs) / total_attacks
            st.metric("Recall (Attack Detection Rate)", f"{recall:.4f}")
        
        if total_predicted > 0:
            precision = len(attack_success_logs) / total_predicted
            st.metric("Precision", f"{precision:.4f}")
        
    except Exception as e:
        st.error(f"❌ Error during inference: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
