"""
Sample and split data from cleaned files
- Tự động xử lý cả Flow-GNN và IP-GNN
- Flow-GNN: Load 8 files, no IP columns
- IP-GNN: Load 1 file (20-02-2018), keep IP columns
- Sample 2M records randomly
- Split 70% train, 10% val, 20% test
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config as cfg


def process_flow_gnn(cleaned_dir=None, output_dir=None):
    """Processing for Flow-GNN (no IP columns)"""
    print("\n" + "="*60)
    print("PROCESSING FLOW-GNN")
    print("="*60)
    
    cleaned_path = Path(cleaned_dir or cfg.OUTPUT_DIR) / "flow_gnn"
    output_path = Path(output_dir or cfg.OUTPUT_DIR) / "flow_gnn"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load specified files from config
    files_to_use = [cleaned_path / f"{Path(f).stem}_cleaned.csv" for f in cfg.FILES_TO_USE]
    
    # Verify files exist
    missing = [f for f in files_to_use if not f.exists()]
    if missing:
        raise ValueError(f"Files not found: {[f.name for f in missing]}")
    
    print(f"\nUsing {len(files_to_use)} files:")
    for f in files_to_use:
        print(f"  - {f.name}")
    
    # Load and concat
    print("\nLoading data...")
    dfs = [pd.read_csv(f) for f in files_to_use]
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df_all):,}")
    
    # Sample 2M records
    if len(df_all) > cfg.SAMPLE_SIZE:
        print(f"Sampling {cfg.SAMPLE_SIZE:,} records...")
        df_all = df_all.sample(n=cfg.SAMPLE_SIZE, random_state=cfg.SEED).reset_index(drop=True)
    else:
        print(f"Using all {len(df_all):,} records")
    
    # Separate features and labels
    feature_cols = [c for c in df_all.columns if c != "Label"]
    X = df_all[feature_cols].values.astype(np.float32)
    y = df_all["Label"].values.astype(np.int64)
    
    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {len(X):,}")
    print(f"Benign: {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"Attack: {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    
    # Split train/val/test
    print("\nSplitting data...")
    idx = np.arange(len(y))
    
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y,
        test_size=cfg.VAL_RATIO + cfg.TEST_RATIO,
        stratify=y,
        random_state=cfg.SEED
    )
    
    val_size_ratio = cfg.VAL_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1 - val_size_ratio,
        stratify=y_temp,
        random_state=cfg.SEED
    )
    
    print(f"Train: {len(idx_train):,} ({len(idx_train)/len(y)*100:.1f}%)")
    print(f"Val:   {len(idx_val):,} ({len(idx_val)/len(y)*100:.1f}%)")
    print(f"Test:  {len(idx_test):,} ({len(idx_test)/len(y)*100:.1f}%)")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    scaler.fit(X[idx_train])
    X_scaled = scaler.transform(X)
    
    # Save
    print(f"\nSaving to {output_path}...")
    np.save(output_path / "X.npy", X_scaled)
    np.save(output_path / "y.npy", y)
    np.save(output_path / "idx_train.npy", idx_train)
    np.save(output_path / "idx_val.npy", idx_val)
    np.save(output_path / "idx_test.npy", idx_test)
    
    with open(output_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_path / "feature_names.json", 'w') as f:
        json.dump({"features": feature_cols}, f, indent=2)
    
    manifest = {
        "total_samples": len(y),
        "n_features": X.shape[1],
        "train_size": len(idx_train),
        "val_size": len(idx_val),
        "test_size": len(idx_test),
        "benign": int((y == 0).sum()),
        "attack": int((y == 1).sum()),
        "files_used": cfg.FILES_TO_USE,
        "files_skipped": cfg.FILES_TO_SKIP,
        "sample_size": cfg.SAMPLE_SIZE,
        "seed": cfg.SEED
    }
    
    with open(output_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("✓ Flow-GNN done!")


def process_ip_gnn(cleaned_dir=None, output_dir=None):
    """Processing for IP-GNN (keep IP columns)"""
    print("\n" + "="*60)
    print("PROCESSING IP-GNN")
    print("="*60)
    
    cleaned_path = Path(cleaned_dir or cfg.OUTPUT_DIR) / "ip_gnn"
    output_path = Path(output_dir or cfg.OUTPUT_DIR) / "ip_gnn"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load IP_GNN_FILE
    file_path = cleaned_path / f"{Path(cfg.IP_GNN_FILE).stem}_cleaned.csv"
    
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path.name}")
    
    print(f"\nUsing file: {file_path.name}")
    
    # Load data
    print("Loading data...")
    df_all = pd.read_csv(file_path)
    print(f"Total rows: {len(df_all):,}")
    
    # Sample 2M records
    if len(df_all) > cfg.SAMPLE_SIZE:
        print(f"Sampling {cfg.SAMPLE_SIZE:,} records...")
        df_all = df_all.sample(n=cfg.SAMPLE_SIZE, random_state=cfg.SEED).reset_index(drop=True)
    else:
        print(f"Using all {len(df_all):,} records")
    
    # Extract IP columns
    ip_cols = [c for c in cfg.IP_COLS if c in df_all.columns]
    if not ip_cols:
        raise ValueError(f"IP columns not found: {cfg.IP_COLS}")
    
    print(f"IP columns: {ip_cols}")
    
    # Separate features, IPs, and labels
    feature_cols = [c for c in df_all.columns if c not in ip_cols and c != "Label"]
    X = df_all[feature_cols].values.astype(np.float32)
    y = df_all["Label"].values.astype(np.int64)
    
    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {len(X):,}")
    print(f"Benign: {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"Attack: {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    
    # Split train/val/test
    print("\nSplitting data...")
    idx = np.arange(len(y))
    
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y,
        test_size=cfg.VAL_RATIO + cfg.TEST_RATIO,
        stratify=y,
        random_state=cfg.SEED
    )
    
    val_size_ratio = cfg.VAL_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1 - val_size_ratio,
        stratify=y_temp,
        random_state=cfg.SEED
    )
    
    print(f"Train: {len(idx_train):,} ({len(idx_train)/len(y)*100:.1f}%)")
    print(f"Val:   {len(idx_val):,} ({len(idx_val)/len(y)*100:.1f}%)")
    print(f"Test:  {len(idx_test):,} ({len(idx_test)/len(y)*100:.1f}%)")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    scaler.fit(X[idx_train])
    X_scaled = scaler.transform(X)
    
    # Save
    print(f"\nSaving to {output_path}...")
    np.save(output_path / "X.npy", X_scaled)
    np.save(output_path / "y.npy", y)
    np.save(output_path / "idx_train.npy", idx_train)
    np.save(output_path / "idx_val.npy", idx_val)
    np.save(output_path / "idx_test.npy", idx_test)
    
    # Save data with IPs for graph building
    df_all[feature_cols] = X_scaled
    df_all.to_csv(output_path / "data_with_ips.csv", index=False)
    
    with open(output_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_path / "feature_names.json", 'w') as f:
        json.dump({
            "features": feature_cols,
            "ip_cols": ip_cols
        }, f, indent=2)
    
    manifest = {
        "total_samples": len(y),
        "n_features": X.shape[1],
        "train_size": len(idx_train),
        "val_size": len(idx_val),
        "test_size": len(idx_test),
        "benign": int((y == 0).sum()),
        "attack": int((y == 1).sum()),
        "file_used": cfg.IP_GNN_FILE,
        "ip_columns": ip_cols,
        "sample_size": cfg.SAMPLE_SIZE,
        "seed": cfg.SEED,
        "note": "IP columns preserved for graph building"
    }
    
    with open(output_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("✓ IP-GNN done!")
    print(f"  → Use data_with_ips.csv for graph building")


def main():
    """Process both Flow-GNN and IP-GNN"""
    print("\n" + "="*60)
    print("SAMPLE & SPLIT PIPELINE")
    print("="*60)
    
    try:
        # Process Flow-GNN
        process_flow_gnn()
        
        # Process IP-GNN
        process_ip_gnn()
        
        print("\n" + "="*60)
        print("ALL DONE! ✅")
        print("="*60)
        print("\nOutput:")
        print("  - dataset-processed/flow_gnn/")
        print("  - dataset-processed/ip_gnn/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
