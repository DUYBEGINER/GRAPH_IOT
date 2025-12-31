"""
Data Preprocessing for Flow-based GNN (GraphSAGE)
Dataset: CICIDS2018 (excluding Thuesday-20-02-2018 file)
Binary Classification: Benign vs Attack
Optimized for Kaggle Notebook
"""

import pandas as pd
import numpy as np
import pickle
import os
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = "/kaggle/input/cicids2018-csv"  # Kaggle dataset path
OUTPUT_DIR = "/kaggle/working/processed_flow"
EXCLUDED_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
SAMPLE_SIZE = None  # Set to limit samples (e.g., 500000)
RANDOM_STATE = 42

# Columns to drop (not useful for training)
COLS_TO_DROP = [
    'Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port',
    'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count'
]


def load_data():
    """Load all CSV files except the IP-based file."""
    print("Loading CICIDS2018 data (Flow-based)...")
    csv_files = sorted(Path(DATA_DIR).glob("*_TrafficForML_CICFlowMeter.csv"))
    csv_files = [f for f in csv_files if f.name != EXCLUDED_FILE]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    print(f"Found {len(csv_files)} files (excluded: {EXCLUDED_FILE})")

    dfs = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}...", end=" ")
        try:
            df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, low_memory=False, encoding='latin-1')

        # Filter header rows
        if 'Label' in df.columns:
            df = df[df['Label'] != 'Label'].copy()

        print(f"{len(df):,} rows")
        dfs.append(df)
        gc.collect()

    data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print(f"Total: {len(data):,} rows, {len(data.columns)} columns")
    return data


def clean_data(data):
    """Clean and preprocess data."""
    print("\nCleaning data...")
    initial_rows = len(data)

    # Drop unnecessary columns
    existing_drops = [c for c in COLS_TO_DROP if c in data.columns]
    if existing_drops:
        data = data.drop(columns=existing_drops)
        print(f"  Dropped columns: {existing_drops}")

    # Convert to numeric (except Label)
    for col in data.columns:
        if col != 'Label':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle missing and infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], 0)

    # Remove duplicates
    data = data.drop_duplicates()
    gc.collect()

    print(f"  Rows: {initial_rows:,} -> {len(data):,}")
    return data


def create_labels(data):
    """Create binary labels: 0 = Benign, 1 = Attack."""
    print("\nCreating binary labels...")
    data['binary_label'] = (data['Label'] != 'Benign').astype(int)

    benign = (data['binary_label'] == 0).sum()
    attack = (data['binary_label'] == 1).sum()
    print(f"  Benign: {benign:,} ({benign/len(data)*100:.1f}%)")
    print(f"  Attack: {attack:,} ({attack/len(data)*100:.1f}%)")

    return data


def extract_and_normalize(data):
    """Extract features and normalize."""
    print("\nExtracting and normalizing features...")

    feature_cols = [c for c in data.columns if c not in ['Label', 'binary_label']]
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


def save_data(X, y, feature_cols, scaler):
    """Save processed data."""
    print(f"\nSaving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(OUTPUT_DIR, "feature_names.pkl"), 'wb') as f:
        pickle.dump(feature_cols, f)

    metadata = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_benign': int((y == 0).sum()),
        'n_attack': int((y == 1).sum()),
        'feature_names': feature_cols
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    print("  Saved: X.npy, y.npy, scaler.pkl, feature_names.pkl, metadata.pkl")
    return metadata


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("CICIDS2018 Preprocessing (Flow-based)")
    print("=" * 60)

    data = load_data()

    if SAMPLE_SIZE and SAMPLE_SIZE < len(data):
        print(f"\nSampling {SAMPLE_SIZE:,} rows...")
        data = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        gc.collect()

    data = clean_data(data)
    data = create_labels(data)
    X, y, feature_cols, scaler = extract_and_normalize(data)

    del data
    gc.collect()

    metadata = save_data(X, y, feature_cols, scaler)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print(f"Samples: {metadata['n_samples']:,}")
    print(f"Features: {metadata['n_features']}")
    print("=" * 60)

    return X, y


if __name__ == "__main__":
    X, y = main()

