import pandas as pd
import os

# Define the directory containing the dataset files
DATA_DIR = "CICIDS2018-CSV"

# Function to check column similarity across datasets
def check_column_similarity(data_dir):
    print("\nChecking column similarity across datasets...")

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith("_TrafficForML_CICFlowMeter.csv")]
    if not csv_files:
        print("No dataset files found in the directory.")
        return

    print(f"Found {len(csv_files)} dataset files.")

    # Initialize a dictionary to store columns for each file
    file_columns = {}

    # Read each file and store its columns
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path, nrows=1, low_memory=False)  # Read only the header row
            file_columns[file] = set(df.columns)
            print(f"Loaded columns from {file}.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Compare columns between files
    base_file = csv_files[0]
    base_columns = file_columns[base_file]
    print(f"\nBase file for comparison: {base_file}")

    for file, columns in file_columns.items():
        if file == base_file:
            continue

        # Find differences
        extra_columns = columns - base_columns
        missing_columns = base_columns - columns

        if extra_columns or missing_columns:
            print(f"\nDifferences found in {file}:")
            if extra_columns:
                print(f"  Extra columns: {extra_columns}")
            if missing_columns:
                print(f"  Missing columns: {missing_columns}")
        else:
            print(f"\n{file} has identical columns to the base file.")

if __name__ == "__main__":
    check_column_similarity(DATA_DIR)
