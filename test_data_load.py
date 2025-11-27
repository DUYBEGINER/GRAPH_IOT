"""
Quick test to check data loading and see what columns we have
"""
import pandas as pd
from pathlib import Path

DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"

csv_files = sorted(Path(DATA_DIR).glob("*_TrafficForML_CICFlowMeter.csv"))
print(f"Found {len(csv_files)} files")

# Load just one file to check
if csv_files:
    csv_file = csv_files[0]
    print(f"\nLoading: {csv_file.name}")

    try:
        df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8', nrows=1000)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, low_memory=False, encoding='latin-1', nrows=1000)

    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"{i:3d}. {col:40s} {dtype}")

    # Filter header rows
    if 'Label' in df.columns:
        mask = df['Label'] != 'Label'
        df_clean = df[mask].copy()
        print(f"\nAfter filtering header rows: {len(df_clean)} rows")

        # Check what columns would be dropped
        columns_to_drop = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        print(f"\nColumns that would be dropped: {existing_cols_to_drop}")

        df_clean = df_clean.drop(columns=existing_cols_to_drop)
        print(f"Remaining columns after drop: {len(df_clean.columns)}")

        # Check numeric columns
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        print(f"\nNumeric columns: {len(numeric_cols)}")
        print(f"Sample numeric columns: {numeric_cols[:10].tolist()}")

        # Check non-numeric columns
        non_numeric_cols = df_clean.select_dtypes(exclude=['int64', 'float64']).columns
        print(f"\nNon-numeric columns: {len(non_numeric_cols)}")
        print(f"Non-numeric columns: {non_numeric_cols.tolist()}")

