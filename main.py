#!/usr/bin/env python3
"""
GNN-IDS Main CLI

Train GNN models for Intrusion Detection System.

Commands:
  flow      - Train Flow-based GNN (Node = flow, Edge = KNN)
  endpoint  - Train Endpoint-based GNN (Node = endpoint, Edge = flow)
  preprocess - Preprocess data only

Examples:
  python main.py flow --csv data/Tuesday_20_02_exist_ip.csv
  python main.py endpoint --csv data/Tuesday_20_02_exist_ip.csv
  python main.py preprocess --csv data/Tuesday_20_02_exist_ip.csv --output data/preprocessed.pt
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_flow_gnn(args):
    """Run Flow-based GNN pipeline."""
    import yaml
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler
    
    from preprocess import load_cicids_csv, split_and_scale, set_seed
    from flow_gnn import build_knn_graph, train_flow_gnn, get_device
    
    # Load config
    config_path = args.config or "flow_gnn/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    if args.csv:
        config['data'] = {'csv_path': args.csv, 'max_samples': args.max_samples or 200000}
    if args.device:
        config['project']['device'] = args.device
    
    set_seed(config['project']['seed'])
    device = get_device(config['project']['device'])
    
    logger.info("=" * 70)
    logger.info("FLOW-BASED GNN PIPELINE")
    logger.info("=" * 70)
    
    # 1. Load data
    logger.info("\n[1/5] Loading CSV data...")
    csv_path = config.get('data', {}).get('csv_path') or args.csv
    X, y, feature_cols = load_cicids_csv(
        csv_path=csv_path,
        max_samples=config.get('data', {}).get('max_samples'),
        seed=config['project']['seed']
    )
    
    # 2. Split and scale
    logger.info("\n[2/5] Splitting and scaling...")
    preprocess_config = yaml.safe_load(open("preprocess/config.yaml"))
    x_tensor, y_tensor, idx_train, idx_val, idx_test, scaler = split_and_scale(
        X, y,
        val_ratio=preprocess_config['data']['val_split'],
        test_ratio=preprocess_config['data']['test_split'],
        seed=config['project']['seed']
    )
    
    # 3. Build KNN graph
    logger.info("\n[3/5] Building KNN graph...")
    edge_index, edge_weight = build_knn_graph(
        x_tensor.numpy(),
        k=config['graph']['k_neighbors'],
        metric=config['graph']['metric']
    )
    
    # 4. Create masks
    logger.info("\n[4/5] Creating train/val/test masks...")
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    # 5. Train
    logger.info("\n[5/5] Training model...")
    test_metrics = train_flow_gnn(
        x_tensor, y_tensor,
        edge_index, edge_weight,
        train_mask, val_mask, test_mask,
        config, device
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("FLOW-BASED GNN COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")


def run_endpoint_gnn(args):
    """Run Endpoint-based GNN pipeline."""
    import yaml
    import pandas as pd
    import torch
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    from preprocess import set_seed
    from endpoint_gnn import create_endpoint_graph, train_endpoint_gnn, get_device
    
    # Load config
    config_path = args.config or "endpoint_gnn/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    if args.csv:
        config['data']['csv_path'] = args.csv
    if args.device:
        config['project']['device'] = args.device
    
    set_seed(config['project']['seed'])
    device = get_device(config['project']['device'])
    
    logger.info("=" * 70)
    logger.info("ENDPOINT-BASED GNN PIPELINE")
    logger.info("=" * 70)
    
    # 1. Load CSV
    logger.info("\n[1/5] Loading CSV data...")
    csv_path = config['data'].get('csv_path') or args.csv
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records")
    
    # 2. Prepare features
    logger.info("\n[2/5] Preparing features...")
    label_col = config['data']['label_col']
    drop_cols = config['data']['drop_cols']
    
    # Get numeric features
    feature_cols = []
    for col in df.columns:
        if col not in drop_cols and col != label_col:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    feature_cols.append(col)
            except:
                pass
    
    # Clean features
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    logger.info(f"Features: {len(feature_cols)}")
    
    # 3. Build graph
    logger.info("\n[3/5] Building endpoint graph...")
    data, num_nodes = create_endpoint_graph(df, feature_cols, config)
    
    # 4. Split edges
    logger.info("\n[4/5] Splitting edges...")
    num_edges = data.edge_index.shape[1]
    indices = torch.randperm(num_edges)
    
    test_size = int(num_edges * config['training']['test_split'])
    val_size = int(num_edges * config['training']['val_split'])
    
    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]
    
    train_edges = data.edge_index[:, train_idx]
    val_edges = data.edge_index[:, val_idx]
    test_edges = data.edge_index[:, test_idx]
    
    logger.info(f"Train edges: {train_edges.shape[1]}")
    logger.info(f"Val edges: {val_edges.shape[1]}")
    logger.info(f"Test edges: {test_edges.shape[1]}")
    
    # 5. Train
    logger.info("\n[5/5] Training model...")
    test_metrics = train_endpoint_gnn(
        data, train_edges, val_edges, test_edges,
        config, device
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("ENDPOINT-BASED GNN COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")


def run_preprocess(args):
    """Run preprocessing only."""
    import yaml
    import torch
    from preprocess import load_cicids_csv, split_and_scale, set_seed
    
    config_path = args.config or "preprocess/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if args.csv:
        config['data']['csv_path'] = args.csv
    
    set_seed(config['project']['seed'])
    
    logger.info("=" * 70)
    logger.info("PREPROCESSING DATA")
    logger.info("=" * 70)
    
    # Load
    logger.info("\n[1/3] Loading CSV...")
    X, y, feature_cols = load_cicids_csv(
        csv_path=config['data']['csv_path'],
        max_samples=config['data'].get('max_samples'),
        seed=config['project']['seed']
    )
    
    # Split and scale
    logger.info("\n[2/3] Splitting and scaling...")
    x_tensor, y_tensor, idx_train, idx_val, idx_test, scaler = split_and_scale(
        X, y,
        val_ratio=config['data']['val_split'],
        test_ratio=config['data']['test_split'],
        seed=config['project']['seed']
    )
    
    # Save
    logger.info("\n[3/3] Saving preprocessed data...")
    output_path = args.output or f"{config['project']['output_dir']}/preprocessed.pt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'x': x_tensor,
        'y': y_tensor,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
        'feature_cols': feature_cols,
        'scaler': scaler
    }, output_path)
    
    logger.info(f"Saved to: {output_path}")
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETED")
    logger.info("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GNN-IDS: Graph Neural Networks for Intrusion Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Flow command
    flow_parser = subparsers.add_parser('flow', help='Train Flow-based GNN')
    flow_parser.add_argument('--csv', type=str, help='Path to CSV file')
    flow_parser.add_argument('--config', type=str, help='Path to config.yaml')
    flow_parser.add_argument('--max-samples', type=int, help='Max samples to load')
    flow_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], help='Device')
    
    # Endpoint command
    endpoint_parser = subparsers.add_parser('endpoint', help='Train Endpoint-based GNN')
    endpoint_parser.add_argument('--csv', type=str, help='Path to CSV file')
    endpoint_parser.add_argument('--config', type=str, help='Path to config.yaml')
    endpoint_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], help='Device')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data only')
    preprocess_parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    preprocess_parser.add_argument('--config', type=str, help='Path to config.yaml')
    preprocess_parser.add_argument('--output', type=str, help='Output path for preprocessed data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'flow':
            run_flow_gnn(args)
        elif args.command == 'endpoint':
            run_endpoint_gnn(args)
        elif args.command == 'preprocess':
            run_preprocess(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
