"""Split data into train/val/test and apply scaling."""

import json
import logging
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def make_split_manifest(
    cleaned_dir: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create train/val/test split from cleaned data and apply scaling.
    
    Supports both random and time-based splitting.
    
    Args:
        cleaned_dir: Directory with cleaned parquet/csv files
        output_dir: Directory to save split data
        config: Configuration dictionary
        
    Returns:
        Manifest dictionary with split statistics
    """
    start_time = time.time()
    
    cleaned_path = Path(cleaned_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 100)
    logger.info("🚀 SPLIT AND SCALE PROCESS")
    logger.info("=" * 100)
    logger.info(f"🎯 Mode:          {config['mode']}")
    logger.info(f"✂️  Split mode:    {config['split']['split_mode']}")
    logger.info(f"📊 Val ratio:     {config['split']['val_ratio']}")
    logger.info(f"📊 Test ratio:    {config['split']['test_ratio']}")
    logger.info(f"🎲 Random seed:   {config['split']['seed']}")
    logger.info("=" * 100)
    
    # Load all cleaned files
    logger.info("\n📂 Loading cleaned data...")
    df_all = load_cleaned_data(cleaned_path)
    
    logger.info(f"✅ Total rows loaded: {len(df_all):,}")
    logger.info(f"📋 Total columns: {len(df_all.columns)}")
    
    # Store original labels if multi-class is enabled
    original_labels = None
    if config['labels'].get('save_original', False):
        label_col = find_original_label_column(df_all)
        if label_col:
            original_labels = df_all[label_col].copy()
            logger.info(f"💾 Saving original labels from column: {label_col}")
    
    # Extract features and labels
    special_cols = config['cleaning'].get('keep_cols_ip_mode', [])
    exclude_cols = ["Label", "source_file"] + special_cols
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]
    
    logger.info(f"🔢 Feature columns: {len(feature_cols)}")
    if special_cols:
        logger.info(f"⚙️  Special columns: {special_cols}")
    
    X = df_all[feature_cols].values.astype(np.float32)
    y = df_all["Label"].values.astype(np.int64)
    
    # Variance filtering (remove constant features)
    if config['labels'].get('variance_filtering', False):
        logger.info("\n🔍 Applying variance filtering...")
        variances = X.var(axis=0)
        variance_threshold = config['labels'].get('variance_threshold', 0.0)
        valid_features = variances > variance_threshold
        
        n_removed = (~valid_features).sum()
        if n_removed > 0:
            logger.info(f"🗑️  Removing {n_removed} features with variance <= {variance_threshold}")
            X = X[:, valid_features]
            feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_features[i]]
            logger.info(f"✅ Remaining features: {len(feature_cols)}")
        else:
            logger.info(f"✅ All features have sufficient variance")
    
    # Get timestamps if available (for time split)
    timestamps = None
    if "Timestamp" in df_all.columns and config['split']['split_mode'] == "time":
        timestamps = pd.to_datetime(df_all["Timestamp"], errors='coerce')
    
    # Perform split
    logger.info(f"\n✂️  Performing {config['split']['split_mode']} split...")
    
    if config['split']['split_mode'] == "time" and timestamps is not None:
        idx_train, idx_val, idx_test = time_based_split(
            len(df_all),
            timestamps,
            config['split']['val_ratio'],
            config['split']['test_ratio']
        )
    else:
        idx_train, idx_val, idx_test = random_stratified_split(
            y,
            config['split']['val_ratio'],
            config['split']['test_ratio'],
            config['split']['seed']
        )
    
    logger.info(f"📊 Train: {len(idx_train):,} ({len(idx_train)/len(y)*100:.1f}%)")
    logger.info(f"📊 Val:   {len(idx_val):,} ({len(idx_val)/len(y)*100:.1f}%)")
    logger.info(f"📊 Test:  {len(idx_test):,} ({len(idx_test)/len(y)*100:.1f}%)")
    
    # Class distribution
    logger.info("\n📈 Class distribution:")
    logger.info(f"   Train - Benign: {(y[idx_train]==0).sum():>7,} | Attack: {(y[idx_train]==1).sum():>7,}")
    logger.info(f"   Val   - Benign: {(y[idx_val]==0).sum():>7,} | Attack: {(y[idx_val]==1).sum():>7,}")
    logger.info(f"   Test  - Benign: {(y[idx_test]==0).sum():>7,} | Attack: {(y[idx_test]==1).sum():>7,}")
    
    # Fit scaler on training data
    logger.info("\n⚙️  Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train = X[idx_train]
    scaler.fit(X_train)
    
    # Transform all data
    logger.info("⚙️  Transforming all data with fitted scaler...")
    X_scaled = scaler.transform(X)
    
    logger.info(f"   Scaled range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    logger.info(f"   Scaled mean: {X_scaled.mean():.3f}, std: {X_scaled.std():.3f}")
    
    # Save data
    logger.info("\n💾 Saving processed data...")
    
    files_saved = []
    
    # Save features and binary labels
    np.save(output_path / "X.npy", X_scaled)
    np.save(output_path / "y.npy", y)
    files_saved.extend(["X.npy", "y.npy"])
    logger.info("   ✅ Saved X.npy and y.npy (binary labels)")
    
    # Save multi-class labels if enabled
    if config['labels'].get('save_multiclass', False) and original_labels is not None:
        label_encoder = LabelEncoder()
        y_multi = label_encoder.fit_transform(original_labels)
        np.save(output_path / "y_multi.npy", y_multi)
        files_saved.append("y_multi.npy")
        
        # Save label encoder classes
        label_classes = {"classes": label_encoder.classes_.tolist()}
        with open(output_path / "label_classes.json", 'w') as f:
            json.dump(label_classes, f, indent=2)
        files_saved.append("label_classes.json")
        
        logger.info(f"   ✅ Saved y_multi.npy with {len(label_encoder.classes_)} classes")
        logger.info(f"   📋 Classes: {label_encoder.classes_.tolist()}")
    
    # Save indices
    np.save(output_path / "idx_train.npy", idx_train)
    np.save(output_path / "idx_val.npy", idx_val)
    np.save(output_path / "idx_test.npy", idx_test)
    files_saved.extend(["idx_train.npy", "idx_val.npy", "idx_test.npy"])
    logger.info("   ✅ Saved train/val/test indices")
    
    # Save scaler
    with open(output_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    files_saved.append("scaler.pkl")
    logger.info("   ✅ Saved StandardScaler")
    
    # Save feature names
    feature_names = {"features": feature_cols, "num_features": len(feature_cols)}
    with open(output_path / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    files_saved.append("feature_names.json")
    logger.info("   ✅ Saved feature names")
    
    # Save IP data if in ip_gnn mode
    if config['mode'] == "ip_gnn":
        save_ip_data(df_all, special_cols, output_path)
        files_saved.append("ip_data.npz")
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Create manifest
    manifest = {
        "mode": config['mode'],
        "split_mode": config['split']['split_mode'],
        "seed": config['split']['seed'],
        "total_samples": len(y),
        "num_features": len(feature_cols),
        "variance_filtering_applied": config['labels'].get('variance_filtering', False),
        "processing_time": processing_time,
        "splits": {
            "train": {
                "size": int(len(idx_train)),
                "benign": int((y[idx_train]==0).sum()),
                "attack": int((y[idx_train]==1).sum())
            },
            "val": {
                "size": int(len(idx_val)),
                "benign": int((y[idx_val]==0).sum()),
                "attack": int((y[idx_val]==1).sum())
            },
            "test": {
                "size": int(len(idx_test)),
                "benign": int((y[idx_test]==0).sum()),
                "attack": int((y[idx_test]==1).sum())
            }
        },
        "scaler_params": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "files_saved": files_saved
    }
    
    # Add multi-class info if available
    if config['labels'].get('save_multiclass', False) and original_labels is not None:
        manifest["multiclass_enabled"] = True
        manifest["num_classes"] = len(label_encoder.classes_)
    
    # Save manifest
    with open(output_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    files_saved.append("manifest.json")
    
    logger.info(f"\n{'='*100}")
    logger.info("✅ SPLIT AND SCALE COMPLETED")
    logger.info(f"{'='*100}")
    logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
    logger.info(f"📁 Output directory: {output_path}")
    logger.info(f"📦 Files saved: {len(files_saved)}")
    logger.info(f"{'='*100}\n")
    
    return manifest


def load_cleaned_data(cleaned_dir: Path) -> pd.DataFrame:
    """Load all cleaned files from directory."""
    
    # Try parquet first, then csv
    parquet_files = sorted(cleaned_dir.glob("*_cleaned.parquet"))
    csv_files = sorted(cleaned_dir.glob("*_cleaned.csv"))
    
    files = parquet_files if parquet_files else csv_files
    
    if not files:
        raise ValueError(f"No cleaned files found in {cleaned_dir}")
    
    dfs = []
    for file in tqdm(files, desc="Loading files", unit="file"):
        if file.suffix == ".parquet":
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenate all
    df_all = pd.concat(dfs, ignore_index=True)
    
    return df_all


def random_stratified_split(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified random split.
    
    Args:
        y: Labels
        val_ratio: Validation ratio
        test_ratio: Test ratio
        seed: Random seed
        
    Returns:
        idx_train, idx_val, idx_test
    """
    idx = np.arange(len(y))
    
    # First split: train vs (val + test)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y,
        test_size=val_ratio + test_ratio,
        stratify=y,
        random_state=seed
    )
    
    # Second split: val vs test
    val_size_ratio = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1 - val_size_ratio,
        stratify=y_temp,
        random_state=seed
    )
    
    return idx_train, idx_val, idx_test


def time_based_split(
    total_size: int,
    timestamps: pd.Series,
    val_ratio: float,
    test_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform time-based split (chronological).
    
    Args:
        total_size: Total number of samples
        timestamps: Timestamp series
        val_ratio: Validation ratio
        test_ratio: Test ratio
        
    Returns:
        idx_train, idx_val, idx_test
    """
    logger.info("  Using time-based split...")
    
    # Sort by timestamp
    sorted_idx = timestamps.argsort().values
    
    # Calculate split points
    train_size = int(total_size * (1 - val_ratio - test_ratio))
    val_size = int(total_size * val_ratio)
    
    # Split chronologically
    idx_train = sorted_idx[:train_size]
    idx_val = sorted_idx[train_size:train_size + val_size]
    idx_test = sorted_idx[train_size + val_size:]
    
    logger.info(f"  Time range - Train: {timestamps.iloc[idx_train].min()} to {timestamps.iloc[idx_train].max()}")
    logger.info(f"  Time range - Val: {timestamps.iloc[idx_val].min()} to {timestamps.iloc[idx_val].max()}")
    logger.info(f"  Time range - Test: {timestamps.iloc[idx_test].min()} to {timestamps.iloc[idx_test].max()}")
    
    return idx_train, idx_val, idx_test


def save_ip_data(df: pd.DataFrame, special_cols: list, output_path: Path):
    """Save IP and timestamp data for ip_gnn mode."""
    
    logger.info("   Saving IP data for ip_gnn mode...")
    
    ip_data = {}
    
    for col in special_cols:
        if col in df.columns:
            ip_data[col] = df[col].values
    
    # Save as npz
    np.savez(output_path / "ip_data.npz", **ip_data)
    
    logger.info(f"   ✅ Saved {len(ip_data)} special columns to ip_data.npz")


def find_original_label_column(df: pd.DataFrame) -> str:
    """Find the original label column (before binary conversion)."""
    # If we added source_file tracking, the original label might be preserved
    # Otherwise, we need to look for alternative label columns
    possible_names = ["original_label", "attack_type", "Label_orig", "Attack"]
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    # If none found, check if we have the source_file column
    # In this case, we need to reload from original CSVs
    # For now, return None
    return None


if __name__ == "__main__":
    import argparse
    from config import load_config
    
    parser = argparse.ArgumentParser(description="Split and scale cleaned data")
    parser.add_argument("--config", type=str, default="preprocess/config.yaml",
                        help="Path to config file")
    parser.add_argument("--cleaned_dir", type=str, default=None,
                        help="Directory with cleaned data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set directories
    cleaned_dir = args.cleaned_dir if args.cleaned_dir else config.output_dir
    output_dir = args.output_dir if args.output_dir else config.output_dir
    
    logger.info(f"Configuration loaded: mode={config.mode}")
    
    # Run split and scale
    try:
        manifest = make_split_manifest(cleaned_dir, output_dir, config)
        logger.info("✓ Split and scale completed successfully!")
    except Exception as e:
        logger.error(f"✗ Split and scale failed: {e}")
        raise
