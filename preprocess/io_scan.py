"""Scan and analyze raw dataset files."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scan_dir(input_dir: str, output_path: str = "scan_report.json") -> Dict[str, Any]:
    """
    Scan directory for CSV files and generate a comprehensive report.
    
    Args:
        input_dir: Path to directory containing CSV files
        output_path: Path to save JSON report
        
    Returns:
        Dictionary containing scan report
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    report = {
        "scan_timestamp": pd.Timestamp.now().isoformat(),
        "input_directory": str(input_path.absolute()),
        "total_files": len(csv_files),
        "files": []
    }
    
    # Scan each file
    for idx, csv_file in enumerate(csv_files, 1):
        logger.info(f"Scanning [{idx}/{len(csv_files)}]: {csv_file.name}")
        
        file_info = scan_single_file(csv_file)
        report["files"].append(file_info)
    
    # Summary statistics
    report["summary"] = generate_summary(report["files"])
    
    # Save report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Scan report saved to {output_path}")
    
    # Print summary
    print_summary(report)
    
    return report


def scan_single_file(csv_path: Path) -> Dict[str, Any]:
    """
    Scan a single CSV file and extract metadata.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary with file information
    """
    try:
        # File size
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        
        # Read first few rows to get column info
        df_sample = pd.read_csv(csv_path, nrows=1000, low_memory=False)
        
        # Total rows (approximate from file size if too large)
        try:
            total_rows = len(pd.read_csv(csv_path, usecols=[0]))
        except:
            # Estimate based on sample
            total_rows = int(file_size_mb * 1000)  # rough estimate
        
        # Column analysis
        columns = df_sample.columns.tolist()
        
        # Detect IP columns
        ip_columns = []
        for col in columns:
            col_lower = col.lower()
            if 'ip' in col_lower or col in ['Src_IP', 'Dst_IP', 'Source', 'Destination']:
                ip_columns.append(col)
                # Check if column contains IP-like values
                sample_values = df_sample[col].dropna().head(3).tolist()
                logger.info(f"  IP column '{col}' sample: {sample_values}")
        
        # Detect label column
        label_column = None
        for col in ['Label', 'label', 'Labels', 'Attack', 'Class']:
            if col in columns:
                label_column = col
                break
        
        # Label distribution (if found)
        label_distribution = None
        if label_column:
            try:
                df_full = pd.read_csv(csv_path, usecols=[label_column])
                label_counts = df_full[label_column].value_counts().to_dict()
                label_distribution = {str(k): int(v) for k, v in label_counts.items()}
            except:
                label_distribution = None
        
        # Detect numeric columns
        numeric_columns = []
        non_numeric_columns = []
        
        for col in columns:
            if col == label_column or col in ip_columns:
                continue
            
            try:
                pd.to_numeric(df_sample[col], errors='coerce')
                numeric_columns.append(col)
            except:
                non_numeric_columns.append(col)
        
        file_info = {
            "filename": csv_path.name,
            "file_size_mb": round(file_size_mb, 2),
            "total_rows": total_rows,
            "total_columns": len(columns),
            "columns": columns,
            "has_ip_columns": len(ip_columns) > 0,
            "ip_columns": ip_columns,
            "label_column": label_column,
            "label_distribution": label_distribution,
            "numeric_columns_count": len(numeric_columns),
            "non_numeric_columns": non_numeric_columns,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error scanning {csv_path.name}: {e}")
        file_info = {
            "filename": csv_path.name,
            "status": "error",
            "error": str(e)
        }
    
    return file_info


def generate_summary(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from scanned files."""
    
    successful_scans = [f for f in files if f["status"] == "success"]
    
    if not successful_scans:
        return {"error": "No successful scans"}
    
    total_rows = sum(f.get("total_rows", 0) for f in successful_scans)
    total_size_mb = sum(f.get("file_size_mb", 0) for f in successful_scans)
    
    files_with_ip = sum(1 for f in successful_scans if f.get("has_ip_columns", False))
    
    # Aggregate label distribution
    aggregated_labels = {}
    for f in successful_scans:
        if f.get("label_distribution"):
            for label, count in f["label_distribution"].items():
                aggregated_labels[label] = aggregated_labels.get(label, 0) + count
    
    # Common columns (appear in all files)
    if successful_scans:
        column_sets = [set(f.get("columns", [])) for f in successful_scans]
        common_columns = list(set.intersection(*column_sets)) if column_sets else []
    else:
        common_columns = []
    
    summary = {
        "total_rows_all_files": total_rows,
        "total_size_mb": round(total_size_mb, 2),
        "files_with_ip_columns": files_with_ip,
        "aggregated_label_distribution": aggregated_labels,
        "common_columns_count": len(common_columns),
        "common_columns": sorted(common_columns)[:20]  # First 20 for brevity
    }
    
    return summary


def print_summary(report: Dict[str, Any]):
    """Print formatted summary to console."""
    
    print("\n" + "="*80)
    print("DATASET SCAN REPORT")
    print("="*80)
    
    print(f"\nDirectory: {report['input_directory']}")
    print(f"Total files: {report['total_files']}")
    
    if "summary" in report:
        summary = report["summary"]
        print(f"\nTotal rows (all files): {summary.get('total_rows_all_files', 0):,}")
        print(f"Total size: {summary.get('total_size_mb', 0):.2f} MB")
        print(f"Files with IP columns: {summary.get('files_with_ip_columns', 0)}")
        
        if summary.get('aggregated_label_distribution'):
            print("\nAggregated Label Distribution:")
            for label, count in sorted(summary['aggregated_label_distribution'].items()):
                print(f"  {label}: {count:,}")
        
        print(f"\nCommon columns across all files: {summary.get('common_columns_count', 0)}")
    
    print("\nPer-file details:")
    for f in report["files"]:
        status = f.get("status", "unknown")
        if status == "success":
            print(f"\n  ✓ {f['filename']}")
            print(f"    Size: {f['file_size_mb']} MB, Rows: {f['total_rows']:,}, Cols: {f['total_columns']}")
            print(f"    Has IP: {f['has_ip_columns']}, IP columns: {f.get('ip_columns', [])}")
            print(f"    Label column: {f.get('label_column', 'N/A')}")
        else:
            print(f"\n  ✗ {f['filename']} - ERROR: {f.get('error', 'Unknown')}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan dataset directory")
    parser.add_argument("--input_dir", type=str, default="dataset-raw",
                        help="Input directory with CSV files")
    parser.add_argument("--output", type=str, default="scan_report.json",
                        help="Output JSON report path")
    
    args = parser.parse_args()
    
    try:
        report = scan_dir(args.input_dir, args.output)
        logger.info("Scan completed successfully!")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise
