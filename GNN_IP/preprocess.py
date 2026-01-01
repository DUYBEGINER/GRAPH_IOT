"""
Data Preprocessing for E-GraphSAGE (Edge Classification)
Dataset: CICIDS2018 - Only Thuesday-20-02-2018 file (contains IP info)
Binary Classification: Benign Flow vs Attack Flow
Optimized for Kaggle Notebook
"""

import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = "/kaggle/input/cicids2018-csv"
OUTPUT_DIR = "/kaggle/working/processed_ip"
TARGET_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
SAMPLE_SIZE = None
RANDOM_STATE = 42

# Columns for graph building
SRC_IP_COL = "Src IP"
DST_IP_COL = "Dst IP"
LABEL_COL = "Label"

# Columns to drop (not useful for features)
COLS_TO_DROP = [
    'Timestamp', 'Flow ID', 'Src Port', 'Dst Port',
    'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count'
]


def load_data():
    """Load the IP-containing CSV file."""
    print("Loading CICIDS2018 data (IP-based)...")

    file_path = os.path.join(DATA_DIR, TARGET_FILE)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"  Loading: {TARGET_FILE}...")
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')

    # Filter header rows
    if 'Label' in df.columns:
        df = df[df['Label'] != 'Label'].copy()

    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_data(data):
    """Clean and preprocess data."""
    print("\nCleaning data...")
    initial_rows = len(data)

    # Keep IP columns separately
    ip_data = data[[SRC_IP_COL, DST_IP_COL]].copy()

    # Drop unnecessary columns (except IP which we need for graph)
    drops = [c for c in COLS_TO_DROP if c in data.columns]
    data = data.drop(columns=drops)
    print(f"  Dropped: {drops}")

    # Convert to numeric (except Label and IP)
    non_numeric = [LABEL_COL, SRC_IP_COL, DST_IP_COL]
    for col in data.columns:
        if col not in non_numeric:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle missing and infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], 0)

    # Remove duplicates
    data = data.drop_duplicates()

    gc.collect()
    print(f"  Rows: {initial_rows:,} -> {len(data):,}")

    return data, ip_data.loc[data.index]


def create_labels(data):
    """Create binary labels: 0 = Benign, 1 = Attack."""
    print("\nCreating binary labels...")
    data['binary_label'] = (data[LABEL_COL] != 'Benign').astype(int)

    benign = (data['binary_label'] == 0).sum()
    attack = (data['binary_label'] == 1).sum()
    print(f"  Benign: {benign:,} ({benign/len(data)*100:.1f}%)")
    print(f"  Attack: {attack:,} ({attack/len(data)*100:.1f}%)")

    return data


def create_ip_mapping(ip_data):
    """Create IP to index mapping for graph construction."""
    print("\nCreating IP mapping...")

    # Get unique IPs
    all_ips = pd.concat([ip_data[SRC_IP_COL], ip_data[DST_IP_COL]]).unique()

    # Create mapping
    ip_encoder = LabelEncoder()
    ip_encoder.fit(all_ips)

    src_idx = ip_encoder.transform(ip_data[SRC_IP_COL].values)
    dst_idx = ip_encoder.transform(ip_data[DST_IP_COL].values)

    print(f"  Unique IPs: {len(all_ips):,}")

    return src_idx, dst_idx, ip_encoder


def extract_and_normalize(data):
    """Extract features and normalize."""
    print("\nExtracting and normalizing features...")

    # Get feature columns (exclude labels and IPs)
    exclude = [LABEL_COL, 'binary_label', SRC_IP_COL, DST_IP_COL]
    feature_cols = [c for c in data.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if data[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    # Remove zero-variance features
    variances = data[feature_cols].var()
    feature_cols = variances[variances > 0].index.tolist()
    print(f"  Features: {len(feature_cols)}")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(data[feature_cols])
    y = data['binary_label'].values

    print(f"  X shape: {X.shape}")
    return X, y, feature_cols, scaler


def save_data(X, y, feature_cols, scaler, src_idx, dst_idx, ip_encoder):
    """Save processed data."""
    print(f"\nSaving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "src_idx.npy"), src_idx)
    np.save(os.path.join(OUTPUT_DIR, "dst_idx.npy"), dst_idx)

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(OUTPUT_DIR, "feature_names.pkl"), 'wb') as f:
        pickle.dump(feature_cols, f)

    with open(os.path.join(OUTPUT_DIR, "ip_encoder.pkl"), 'wb') as f:
        pickle.dump(ip_encoder, f)

    metadata = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_ips': len(ip_encoder.classes_),
        'n_benign': int((y == 0).sum()),
        'n_attack': int((y == 1).sum()),
        'feature_names': feature_cols
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    print("  Saved: X.npy, y.npy, src_idx.npy, dst_idx.npy, scaler.pkl, feature_names.pkl, ip_encoder.pkl, metadata.pkl")
    return metadata


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("CICIDS2018 Preprocessing (E-GraphSAGE)")
    print("=" * 60)

    data = load_data()

    if SAMPLE_SIZE and SAMPLE_SIZE < len(data):
        print(f"\nSampling {SAMPLE_SIZE:,} rows...")
        data = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        gc.collect()

    data, ip_data = clean_data(data)
    data = create_labels(data)
    src_idx, dst_idx, ip_encoder = create_ip_mapping(ip_data)
    X, y, feature_cols, scaler = extract_and_normalize(data)

    del data, ip_data
    gc.collect()

    metadata = save_data(X, y, feature_cols, scaler, src_idx, dst_idx, ip_encoder)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print(f"Flows (edges):  {metadata['n_samples']:,}")
    print(f"Features:       {metadata['n_features']}")
    print(f"Unique IPs:     {metadata['n_ips']:,}")
    print(f"Benign flows:   {metadata['n_benign']:,}")
    print(f"Attack flows:   {metadata['n_attack']:,}")
    print("=" * 60)

    return X, y, src_idx, dst_idx


if __name__ == "__main__":
    X, y, src_idx, dst_idx = main()

