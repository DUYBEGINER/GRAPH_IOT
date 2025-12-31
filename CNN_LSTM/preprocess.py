"""
Shared data preprocessing for CNN and LSTM models
Dataset: CICIDS2018 (Binary Classification)
"""
import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import *

def load_data():
    """Load and merge all CSV files except the excluded one"""
    print("Loading data...")
    csv_files = sorted(Path(DATA_DIR).glob("*_TrafficForML_CICFlowMeter.csv"))
    csv_files = [f for f in csv_files if f.name != EXCLUDED_FILE]
    print(f"  Found {len(csv_files)} files")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False, encoding='utf-8')
        except:
            df = pd.read_csv(f, low_memory=False, encoding='latin-1')
        if 'Label' in df.columns:
            df = df[df['Label'] != 'Label'].copy()
        print(f"  {f.name}: {len(df):,} rows")
        dfs.append(df)
        gc.collect()

    data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"  Total: {len(data):,} rows")
    return data

def clean_data(data):
    """Clean data: drop columns, handle inf/nan, sort by time"""
    print("Cleaning data...")

    # Drop unnecessary columns (keep Timestamp for sorting)
    cols_to_drop = [c for c in COLS_TO_DROP if c in data.columns and c != 'Timestamp']
    data = data.drop(columns=cols_to_drop)

    # Parse and sort by Timestamp
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        data = data.dropna(subset=['Timestamp'])
        data = data.sort_values('Timestamp').reset_index(drop=True)
        print("  Sorted by Timestamp")

    # Convert to numeric
    for col in data.columns:
        if col not in ['Label', 'Timestamp']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle inf/nan
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    data[numeric_cols] = data[numeric_cols].fillna(0)

    # Remove duplicates
    before = len(data)
    data = data.drop_duplicates()
    print(f"  Removed {before - len(data):,} duplicates")

    gc.collect()
    return data

def balance_sample(data):
    """Balance and sample data while preserving time order"""
    print("Balancing data...")

    # Create binary labels
    data['binary_label'] = (data['Label'] != 'Benign').astype(int)

    benign = data[data['binary_label'] == 0]
    attack = data[data['binary_label'] == 1]
    print(f"  Original - Benign: {len(benign):,}, Attack: {len(attack):,}")

    # Calculate target samples
    target_attack = min(int(TOTAL_SAMPLES * ATTACK_RATIO), len(attack))
    target_benign = min(int(target_attack * BENIGN_RATIO / ATTACK_RATIO), len(benign))

    # Sample uniformly across time (not random) to preserve temporal distribution
    benign_idx = np.linspace(0, len(benign)-1, target_benign, dtype=int)
    attack_idx = np.linspace(0, len(attack)-1, target_attack, dtype=int)

    data = pd.concat([benign.iloc[benign_idx], attack.iloc[attack_idx]])

    # Sort again by timestamp
    if 'Timestamp' in data.columns:
        data = data.sort_values('Timestamp').reset_index(drop=True)

    print(f"  Sampled - Benign: {target_benign:,}, Attack: {target_attack:,}")
    print(f"  Total: {len(data):,}")

    del benign, attack
    gc.collect()
    return data

def extract_features(data):
    """Extract features (exclude Timestamp and labels)"""
    print("Extracting features...")

    # Get feature columns (exclude Timestamp, Label, binary_label)
    feature_cols = [c for c in data.columns if c not in ['Label', 'binary_label', 'Timestamp']]
    feature_cols = [c for c in feature_cols if data[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    # Remove zero-variance features
    variances = data[feature_cols].var()
    feature_cols = variances[variances > 0].index.tolist()
    print(f"  Features: {len(feature_cols)}")

    X = data[feature_cols].values
    y = data['binary_label'].values

    return X, y, feature_cols

def split_data(X, y):
    """Split chronologically into train/val/test sets (no shuffle to prevent data leakage)"""
    print("Splitting chronologically...")

    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val, X_test):
    """Fit scaler on train data only (prevent data leakage)"""
    print("Normalizing...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                           feature_cols, scaler, output_dir):
    """Save preprocessed data"""
    print(f"Saving to {output_dir}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(output_dir / 'X_train.npy', X_train.astype(np.float32))
    np.save(output_dir / 'X_val.npy', X_val.astype(np.float32))
    np.save(output_dir / 'X_test.npy', X_test.astype(np.float32))
    np.save(output_dir / 'y_train.npy', y_train.astype(np.int32))
    np.save(output_dir / 'y_val.npy', y_val.astype(np.int32))
    np.save(output_dir / 'y_test.npy', y_test.astype(np.int32))

    # Save scaler and feature names
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_cols, f)

    # Save metadata
    metadata = {
        'n_features': len(feature_cols),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_benign': int((y_train == 0).sum()),
        'train_attack': int((y_train == 1).sum()),
        'val_benign': int((y_val == 0).sum()),
        'val_attack': int((y_val == 1).sum()),
        'test_benign': int((y_test == 0).sum()),
        'test_attack': int((y_test == 1).sum())
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("  Done!")
    return metadata

def main():
    """Main preprocessing pipeline"""
    import os
    print("=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    output_dir = os.path.join(OUTPUT_DIR, "processed_data")

    # Pipeline
    data = load_data()
    data = clean_data(data)
    data = balance_sample(data)
    X, y, feature_cols = extract_features(data)

    del data
    gc.collect()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    del X, y
    gc.collect()

    X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test)

    metadata = save_preprocessed_data(
        X_train, X_val, X_test, y_train, y_val, y_test,
        feature_cols, scaler, output_dir
    )

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Total samples: {metadata['train_samples'] + metadata['val_samples'] + metadata['test_samples']:,}")
    print("=" * 50)

if __name__ == "__main__":
    main()

