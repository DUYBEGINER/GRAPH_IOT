"""
Script to merge all CICIDS2018 CSV files into a single merged CSV file
Author: Senior Data Engineer
Date: 2025-11-24
"""

import pandas as pd
import os
from pathlib import Path
import time

# ============================================================================
# CONFIGURATION - Change this to your data directory
# ============================================================================
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"

# Output file settings
OUTPUT_FILENAME = "CICIDS2018_merged.csv"
ADD_SOURCE_COLUMN = True  # Set to False if you don't want the source_file column

# Columns to exclude (these have different schemas across files)
COLUMNS_TO_EXCLUDE = ['Dst IP', 'Flow ID', 'Src IP', 'Src Port']

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def merge_cicids_csv_files(data_directory, output_filename, add_source_column=False, columns_to_exclude=None):
    """
    Merge all CICIDS2018 CSV files into a single CSV file.

    Args:
        data_directory (str): Path to the directory containing CSV files
        output_filename (str): Name of the output merged CSV file
        add_source_column (bool): Whether to add a column with the source filename
        columns_to_exclude (list): List of column names to exclude from the merge

    Returns:
        pd.DataFrame: The merged dataframe
    """

    print("=" * 80)
    print("CICIDS2018 CSV FILES MERGER")
    print("=" * 80)
    print(f"\nData directory: {data_directory}")
    print(f"Output file: {output_filename}")
    print(f"Add source file column: {add_source_column}")
    if columns_to_exclude:
        print(f"Columns to exclude: {', '.join(columns_to_exclude)}")
    print("-" * 80)

    # Check if directory exists
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Directory not found: {data_directory}")

    # Find all CSV files ending with _TrafficForML_CICFlowMeter.csv
    csv_pattern = "*_TrafficForML_CICFlowMeter.csv"
    csv_files = sorted(Path(data_directory).glob(csv_pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching pattern '{csv_pattern}' found in {data_directory}"
        )

    print(f"\nFound {len(csv_files)} CSV file(s) to merge:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file.name}")

    print("\n" + "-" * 80)
    print("Starting merge process...")
    print("-" * 80)

    # List to store individual dataframes
    dfs = []
    total_rows = 0
    start_time = time.time()

    # Read each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Reading: {csv_file.name}")

        try:
            # Read CSV with low_memory=False to avoid dtype warnings
            # Try UTF-8 encoding first, fall back to latin-1 if needed
            try:
                df = pd.read_csv(
                    csv_file,
                    low_memory=False,
                    encoding='utf-8'
                )
            except UnicodeDecodeError:
                print(f"  UTF-8 encoding failed, trying latin-1...")
                df = pd.read_csv(
                    csv_file,
                    low_memory=False,
                    encoding='latin-1'
                )

            rows_count = len(df)
            total_rows += rows_count
            print(f"  ✓ Loaded {rows_count:,} rows, {len(df.columns)} columns")

            # Drop excluded columns if they exist in this dataframe
            if columns_to_exclude:
                cols_to_drop = [col for col in columns_to_exclude if col in df.columns]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    print(f"  ℹ Dropped columns: {', '.join(cols_to_drop)}")
                    print(f"  Remaining columns: {len(df.columns)}")

            # Add source filename column if requested
            if add_source_column:
                df['source_file'] = csv_file.name

            dfs.append(df)

        except Exception as e:
            print(f"  ✗ Error reading {csv_file.name}: {e}")
            print(f"  Skipping this file...")
            continue

    if not dfs:
        raise ValueError("No data frames were successfully loaded!")

    print("\n" + "-" * 80)
    print("Concatenating all dataframes...")
    print("-" * 80)

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    print(f"\n✓ Merge completed!")
    print(f"  Total rows: {len(merged_df):,}")
    print(f"  Total columns: {len(merged_df.columns)}")
    print(f"  Memory usage: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display column names
    print(f"\nColumns in merged dataset:")
    for i, col in enumerate(merged_df.columns, 1):
        print(f"  {i}. {col}")

    # Display label distribution if Label column exists
    if 'Label' in merged_df.columns:
        print("\n" + "-" * 80)
        print("Label distribution:")
        print("-" * 80)
        label_counts = merged_df['Label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {label}: {count:,} ({percentage:.2f}%)")

    # Save to CSV
    output_path = os.path.join(data_directory, output_filename)
    print("\n" + "-" * 80)
    print(f"Saving merged data to: {output_path}")
    print("-" * 80)

    try:
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✓ File saved successfully!")
        print(f"  File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"\n✗ Error saving file: {e}")
        raise

    # Display timing information
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Files processed: {len(dfs)}/{len(csv_files)}")
    print(f"Total rows: {len(merged_df):,}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Output file: {output_filename}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print("=" * 80)

    return merged_df


def main():
    """Main function to execute the merge process"""
    try:
        merged_df = merge_cicids_csv_files(
            data_directory=DATA_DIR,
            output_filename=OUTPUT_FILENAME,
            add_source_column=ADD_SOURCE_COLUMN,
            columns_to_exclude=COLUMNS_TO_EXCLUDE
        )
        print("\n✓ Process completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

