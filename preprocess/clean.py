"""Clean and preprocess raw CSV files with mode-aware column handling."""

import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_all(
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    output_format: str = "parquet"
) -> Dict[str, Any]:
    """
    Clean all CSV files in input directory with mode-aware processing.
    
    Args:
        input_dir: Directory containing raw CSV files OR path to a single CSV file
        output_dir: Directory to save cleaned files
        config: Configuration dictionary from YAML
        output_format: 'parquet' or 'csv'
        
    Returns:
        Dictionary containing cleaning metadata
    """
    start_time = time.time()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle both directory and single file input
    if input_path.is_file():
        csv_files = [input_path]
    else:
        csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    logger.info("=" * 100)
    logger.info("🚀 STARTING DATA CLEANING PROCESS")
    logger.info("=" * 100)
    logger.info(f"📁 Input:        {input_path.absolute()}")
    logger.info(f"📁 Output:       {output_path.absolute()}")
    logger.info(f"🎯 Mode:         {config['mode']}")
    logger.info(f"📦 Files found:  {len(csv_files)}")
    logger.info(f"🔧 Chunk size:   {config['chunk_processing']['chunk_size']:,} rows")
    logger.info(f"💾 Format:       {output_format}")
    logger.info("=" * 100)
    
    metadata = {
        "mode": config['mode'],
        "input_dir": str(input_path.absolute()),
        "output_dir": str(output_path.absolute()),
        "files_processed": [],
        "total_rows_input": 0,
        "total_rows_output": 0,
        "total_rows_dropped": 0,
        "total_duplicates_removed": 0,
        "schema": None,
        "processing_time": 0
    }
    
    # Process each file with progress bar
    for idx, csv_file in enumerate(tqdm(csv_files, desc="Processing files", unit="file"), 1):
        logger.info(f"\n{'='*100}")
        logger.info(f"📄 File [{idx}/{len(csv_files)}]: {csv_file.name}")
        logger.info(f"{'='*100}")
        
        file_metadata = clean_single_file(
            csv_file,
            output_path,
            config,
            output_format
        )
        
        metadata["files_processed"].append(file_metadata)
        metadata["total_rows_input"] += file_metadata["rows_input"]
        metadata["total_rows_output"] += file_metadata["rows_output"]
        metadata["total_duplicates_removed"] += file_metadata.get("duplicates_removed", 0)
        
        # Use schema from first file
        if metadata["schema"] is None and file_metadata.get("schema"):
            metadata["schema"] = file_metadata["schema"]
    
    metadata["total_rows_dropped"] = metadata["total_rows_input"] - metadata["total_rows_output"]
    metadata["processing_time"] = time.time() - start_time
    
    # Save metadata
    schema_path = output_path / "schema.json"
    with open(schema_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info("✅ CLEANING SUMMARY")
    logger.info(f"{'='*100}")
    logger.info(f"📊 Total rows input:       {metadata['total_rows_input']:,}")
    logger.info(f"✅ Total rows output:      {metadata['total_rows_output']:,}")
    logger.info(f"❌ Rows dropped:           {metadata['total_rows_dropped']:,} ({metadata['total_rows_dropped']/metadata['total_rows_input']*100:.2f}%)")
    logger.info(f"🗑️  Duplicates removed:    {metadata['total_duplicates_removed']:,}")
    logger.info(f"⏱️  Processing time:       {metadata['processing_time']:.2f}s")
    logger.info(f"📝 Schema saved to:       {schema_path}")
    logger.info(f"{'='*100}\n")
    
    return metadata


def clean_single_file(
    csv_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    output_format: str = "parquet"
) -> Dict[str, Any]:
    """
    Clean a single CSV file in chunks.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory
        config: Configuration object
        output_format: Output format ('parquet' or 'csv')
        
    Returns:
        File processing metadata
    """
    file_start_time = time.time()
    
    # Output filename
    output_name = csv_path.stem + f"_cleaned.{output_format}"
    output_path = output_dir / output_name
    
    file_metadata = {
        "input_file": csv_path.name,
        "output_file": output_name,
        "rows_input": 0,
        "rows_output": 0,
        "rows_dropped": 0,
        "duplicates_removed": 0,
        "processing_time": 0,
        "schema": None
    }
    
    # Process in chunks
    chunks_processed = []
    
    try:
        chunk_iter = pd.read_csv(csv_path, chunksize=config['chunk_processing']['chunk_size'], low_memory=False)
        
        # Use tqdm for progress bar
        with tqdm(desc=f"  Processing chunks", unit="chunk") as pbar:
            for chunk_idx, chunk in enumerate(chunk_iter):
                file_metadata["rows_input"] += len(chunk)
                
                # Clean chunk
                cleaned_chunk = clean_chunk(chunk, config, csv_path.name)
                
                if cleaned_chunk is not None and len(cleaned_chunk) > 0:
                    chunks_processed.append(cleaned_chunk)
                    file_metadata["rows_output"] += len(cleaned_chunk)
                    
                    # Save schema from first chunk
                    if file_metadata["schema"] is None:
                        file_metadata["schema"] = extract_schema(cleaned_chunk, config)
                
                pbar.update(1)
                pbar.set_postfix({"rows": file_metadata["rows_input"], "kept": file_metadata["rows_output"]})
        
        # Combine and save all chunks
        if chunks_processed:
            df_cleaned = pd.concat(chunks_processed, ignore_index=True)
            
            # Remove duplicates if enabled
            if config['cleaning'].get('remove_duplicates', False):
                original_len = len(df_cleaned)
                df_cleaned = df_cleaned.drop_duplicates()
                duplicates_removed = original_len - len(df_cleaned)
                file_metadata["duplicates_removed"] = duplicates_removed
                file_metadata["rows_output"] = len(df_cleaned)
                
                if duplicates_removed > 0:
                    logger.info(f"  🗑️  Removed {duplicates_removed:,} duplicate rows")
            
            logger.info(f"  💾 Saving {len(df_cleaned):,} rows to {output_name}...")
            
            if output_format == "parquet":
                df_cleaned.to_parquet(output_path, index=False)
            else:
                df_cleaned.to_csv(output_path, index=False)
            
            logger.info(f"  ✅ Saved: {len(df_cleaned):,} rows")
        else:
            logger.warning(f"  ⚠️  No data remaining after cleaning!")
        
        file_metadata["rows_dropped"] = file_metadata["rows_input"] - file_metadata["rows_output"]
        file_metadata["processing_time"] = time.time() - file_start_time
        
        logger.info(f"  ⏱️  File processing time: {file_metadata['processing_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"  ❌ Error processing {csv_path.name}: {e}")
        file_metadata["error"] = str(e)
    
    return file_metadata


def clean_chunk(chunk: pd.DataFrame, config: Dict[str, Any], source_filename: str = None) -> Optional[pd.DataFrame]:
    """
    Clean a single chunk of data.
    
    Steps:
    1. Normalize column names
    2. Create binary label
    3. Add source file tracking (optional)
    4. Convert numeric columns (handle Infinity)
    5. Handle inf/nan
    6. Drop rows with too many missing
    7. Impute remaining missing
    8. Clip outliers
    9. Keep/drop columns based on mode
    
    Args:
        chunk: Input dataframe chunk
        config: Configuration dictionary
        source_filename: Name of source file (for tracking)
        
    Returns:
        Cleaned dataframe or None if empty
    """
    if chunk is None or len(chunk) == 0:
        return None
    
    df = chunk.copy()
    
    # 1. Normalize column names (strip spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Find and process label column
    label_col = find_label_column(df)
    
    if label_col:
        df["Label"] = create_binary_label(df[label_col])
        if label_col != "Label":
            df = df.drop(columns=[label_col])
    else:
        logger.warning("  No label column found, creating dummy labels")
        df["Label"] = 0
    
    # 3. Add source file tracking (if enabled)
    if config['cleaning'].get('add_source_column', False) and source_filename:
        df['source_file'] = source_filename
    
    # 4. Identify columns to keep/drop based on mode
    keep_cols = config['cleaning'].get('keep_cols_ip_mode', [])
    drop_cols = config['cleaning']['drop_cols_common']
    
    # Save IP/Timestamp columns if in ip_gnn mode
    special_cols = {}
    if config['mode'] == "ip_gnn":
        for col in keep_cols:
            if col in df.columns:
                special_cols[col] = df[col].copy()
    
    # Drop specified columns (if they exist)
    cols_to_drop = [c for c in drop_cols if c in df.columns and c not in keep_cols]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # 5. Convert numeric columns robustly
    numeric_cols = []
    for col in df.columns:
        if col in ["Label", "source_file"] or col in special_cols:
            continue
        
        df[col] = convert_to_numeric_robust(df[col])
        
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            numeric_cols.append(col)
    
    # 6. Handle inf → nan
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 7. Drop rows with too many missing values
    if config['cleaning']['missing_threshold'] > 0:
        missing_ratio = df[numeric_cols].isnull().sum(axis=1) / len(numeric_cols)
        valid_rows = missing_ratio <= config['cleaning']['missing_threshold']
        df = df[valid_rows].reset_index(drop=True)
    
    if len(df) == 0:
        return None
    
    # 8. Impute remaining missing values
    impute_strategy = config['cleaning']['impute_strategy']
    if impute_strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif impute_strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:  # zero
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Handle any remaining NaNs (e.g., if entire column is NaN)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 9. Clip outliers (optional)
    clip_quantiles = config['cleaning'].get('clip_quantiles')
    if clip_quantiles:
        lower_q, upper_q = clip_quantiles
        for col in numeric_cols:
            lower = df[col].quantile(lower_q)
            upper = df[col].quantile(upper_q)
            df[col] = df[col].clip(lower, upper)
    
    # 10. Restore special columns for ip_gnn mode
    for col, values in special_cols.items():
        df[col] = values
    
    # 11. Final column ordering: features + Label + special cols + source_file
    feature_cols = [c for c in df.columns if c not in ["Label", "source_file"] and c not in special_cols]
    ordered_cols = feature_cols + ["Label"]
    
    if config['cleaning'].get('add_source_column', False) and 'source_file' in df.columns:
        ordered_cols.append('source_file')
    
    ordered_cols += list(special_cols.keys())
    df = df[ordered_cols]
    
    return df


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Find the label column in dataframe."""
    possible_names = ["Label", "label", "Labels", "Attack", "Class", "class"]
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    return None


def create_binary_label(label_series: pd.Series) -> pd.Series:
    """Convert label to binary (0=Benign, 1=Attack)."""
    # Normalize to string
    label_str = label_series.astype(str).str.strip().str.lower()
    
    # Binary: benign=0, everything else=1
    binary_label = (label_str != "benign").astype(int)
    
    return binary_label


def convert_to_numeric_robust(series: pd.Series) -> pd.Series:
    """
    Convert series to numeric, handling 'Infinity' strings and other issues.
    """
    # Replace string 'Infinity' and variants
    if series.dtype == object:
        series = series.replace(['Infinity', 'infinity', 'inf', '-Infinity', '-infinity', '-inf'], 
                               [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf])
    
    # Convert to numeric
    series = pd.to_numeric(series, errors='coerce')
    
    return series


def extract_schema(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract schema information from cleaned dataframe."""
    
    keep_cols = config['cleaning'].get('keep_cols_ip_mode', [])
    feature_cols = [c for c in df.columns if c != "Label" and c not in keep_cols]
    
    schema = {
        "mode": config['mode'],
        "total_columns": len(df.columns),
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "label_column": "Label",
        "special_columns": keep_cols,
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    
    return schema


if __name__ == "__main__":
    import argparse
    from config import load_config
    
    parser = argparse.ArgumentParser(description="Clean raw CSV files")
    parser.add_argument("--config", type=str, default="preprocess/config.yaml",
                        help="Path to config file")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Override input directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--format", type=str, default="parquet",
                        choices=["parquet", "csv"],
                        help="Output format")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override paths if provided
    if args.input_dir:
        config['paths']['input_dir'] = args.input_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    logger.info(f"Configuration loaded: mode={config['mode']}")
    
    # Run cleaning
    try:
        # Determine input dir based on mode
        if config['mode'] == 'ip_gnn':
            input_dir = config['paths']['input_file_ip']
        else:
            input_dir = config['paths']['input_dir_flow']
        
        metadata = clean_all(input_dir, config['paths']['output_dir'], config, args.format)
        logger.info("✓ Cleaning completed successfully!")
    except Exception as e:
        logger.error(f"✗ Cleaning failed: {e}")
        raise
