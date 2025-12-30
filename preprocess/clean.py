import numpy as np
import pandas as pd
from pathlib import Path
import config as cfg


def clean_all(input_dir=None, output_dir=None, mode="flow_gnn"):
    """Clean CSV files
    
    Args:
        mode: "flow_gnn" or "ip_gnn"
            - flow_gnn: process all files, drop IP columns
            - ip_gnn: process only IP_GNN_FILE, keep IP columns
    """
    input_path = Path(input_dir or cfg.INPUT_DIR)
    output_path = Path(output_dir or cfg.OUTPUT_DIR)
    
    # Create subdirectories
    if mode == "flow_gnn":
        output_path = output_path / "flow_gnn"
    elif mode == "ip_gnn":
        output_path = output_path / "ip_gnn"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select files based on mode
    if mode == "ip_gnn":
        csv_files = [input_path / cfg.IP_GNN_FILE]
        print(f"\nCleaning for IP-GNN (keeping IP columns):")
    else:
        csv_files = [input_path / fname for fname in cfg.FILES_TO_USE]
        print(f"\nCleaning for Flow-GNN (dropping IP columns):")
    
    # Verify files exist
    missing = [f for f in csv_files if not f.exists()]
    if missing:
        raise ValueError(f"Files not found: {[f.name for f in missing]}")
    
    print(f"Processing {len(csv_files)} file(s):")
    for csv_file in csv_files:
        clean_single_file(csv_file, output_path, mode)
        print(f"✓ {csv_file.name}")


def clean_single_file(csv_path, output_dir, mode="flow_gnn"):
    """Clean single CSV file by chunks"""
    output_path = output_dir / f"{csv_path.stem}_cleaned.csv"
    chunks_processed = []
    
    for chunk in pd.read_csv(csv_path, chunksize=cfg.CHUNK_SIZE, low_memory=False):
        cleaned = clean_chunk(chunk, mode)
        if cleaned is not None and len(cleaned) > 0:
            chunks_processed.append(cleaned)
    
    if chunks_processed:
        df_cleaned = pd.concat(chunks_processed, ignore_index=True)
        df_cleaned = df_cleaned.drop_duplicates()
        df_cleaned.to_csv(output_path, index=False)


def clean_chunk(chunk, mode="flow_gnn"):
    """Clean a single chunk of data"""
    if chunk is None or len(chunk) == 0:
        return None
    
    df = chunk.copy()
    df.columns = df.columns.str.strip()
    
    # Handle label
    label_col = find_label_column(df)
    if label_col:
        df["Label"] = create_binary_label(df[label_col])
        if label_col != "Label":
            df = df.drop(columns=[label_col])
    else:
        df["Label"] = 0
    
    # Determine columns to drop
    if mode == "ip_gnn":
        # Keep IP columns, drop only metadata
        cols_to_drop = [c for c in cfg.DROP_COLS if c in df.columns]
    else:
        # Drop both metadata and IP columns
        cols_to_drop = [c for c in cfg.DROP_COLS + cfg.IP_COLS if c in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Convert to numeric (skip Label and IP columns for ip_gnn)
    numeric_cols = []
    for col in df.columns:
        if col == "Label":
            continue
        if mode == "ip_gnn" and col in cfg.IP_COLS:
            continue
        df[col] = convert_to_numeric_robust(df[col])
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            numeric_cols.append(col)
    
    # Handle inf/nan
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with too many missing values
    missing_ratio = df[numeric_cols].isnull().sum(axis=1) / len(numeric_cols)
    df = df[missing_ratio <= cfg.MISSING_THRESHOLD].reset_index(drop=True)
    
    if len(df) == 0:
        return None
    
    # Impute with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Clip outliers
    for col in numeric_cols:
        q_low = df[col].quantile(cfg.CLIP_QUANTILES[0])
        q_high = df[col].quantile(cfg.CLIP_QUANTILES[1])
        df[col] = df[col].clip(q_low, q_high)
    
    # Reorder columns
    if mode == "ip_gnn":
        # [IP cols, features, Label]
        ip_cols = [c for c in cfg.IP_COLS if c in df.columns]
        feature_cols = [c for c in df.columns if c not in ip_cols and c != "Label"]
        return df[ip_cols + feature_cols + ["Label"]]
    else:
        # [features, Label]
        feature_cols = [c for c in df.columns if c != "Label"]
        return df[feature_cols + ["Label"]]


def find_label_column(df):
    """Find label column name"""
    for name in ["Label", "label", "Labels", "Attack", "Class", "class"]:
        if name in df.columns:
            return name
    return None


def create_binary_label(label_series):
    """Create binary label (0=Benign, 1=Attack)"""
    return (label_series.astype(str).str.strip().str.lower() != "benign").astype(int)


def convert_to_numeric_robust(series):
    """Convert series to numeric, handling infinity values"""
    if series.dtype == object:
        series = series.replace(
            ['Infinity', 'infinity', 'inf', '-Infinity', '-infinity', '-inf'],
            [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
        )
    return pd.to_numeric(series, errors='coerce')


def main():
    """Clean both Flow-GNN and IP-GNN"""
    print("\n" + "="*60)
    print("CLEANING PIPELINE")
    print("="*60)
    
    try:
        # Clean for Flow-GNN
        print("\n[1/2] Flow-GNN")
        clean_all(mode="flow_gnn")
        
        # Clean for IP-GNN
        print("\n[2/2] IP-GNN")
        clean_all(mode="ip_gnn")
        
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
