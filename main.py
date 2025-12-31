#!/usr/bin/env python3
"""
GNN-IDS Main CLI

Train GNN models for Intrusion Detection System.

Commands:
  flow      - Train Flow-based GNN (Node = flow, Edge = KNN)
  ip        - Train IP-based GNN (Node = endpoint, Edge = flow)

Examples:
  python main.py flow
  python main.py ip
"""

import sys
import argparse
from pathlib import Path


def run_flow_gnn(args):
    """Run Flow-based GNN pipeline."""
    import torch
    import numpy as np
    
    from flow_gnn import build_knn_graph, train_flow_gnn, get_device, set_seed, config as cfg
    
    # Setup
    device_str = args.device or cfg.DEVICE
    set_seed(cfg.SEED)
    device = get_device(device_str)
    
    print("=" * 70)
    print("🔷 FLOW-BASED GNN PIPELINE")
    print("=" * 70)
    
    # Load preprocessed data
    data_dir = Path(cfg.DATA_DIR)
    
    print("\n📂 [1/4] Loading preprocessed data...")
    print(f"   Data directory: {data_dir}")
    
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    idx_train = np.load(data_dir / "idx_train.npy")
    idx_val = np.load(data_dir / "idx_val.npy")
    idx_test = np.load(data_dir / "idx_test.npy")
    
    x_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    print(f"   Features: {x_tensor.shape}")
    print(f"   Train: {len(idx_train):,} | Val: {len(idx_val):,} | Test: {len(idx_test):,}")
    
    # Build KNN graph
    print("\n🔨 [2/4] Building KNN graph...")
    edge_index = build_knn_graph(X, k=cfg.K_NEIGHBORS)
    
    # Create masks
    print("\n🎯 [3/4] Creating train/val/test masks...")
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    # Train
    print("\n🚀 [4/4] Training model...")
    output_dir = args.output or cfg.OUTPUT_DIR
    test_metrics = train_flow_gnn(
        x_tensor, y_tensor,
        edge_index,
        train_mask, val_mask, test_mask,
        device=device,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("✅ FLOW-BASED GNN COMPLETED")
    print("=" * 70)
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   F1 Score: {test_metrics['f1']:.4f}")
    print(f"   Output:   {output_dir}")


def run_ip_gnn(args):
    """Run IP-based GNN pipeline."""
    import numpy as np
    import pandas as pd
    
    from ip_gnn import create_endpoint_graph, train_ip_gnn, get_device, set_seed, config as cfg
    
    # Setup
    device_str = args.device or cfg.DEVICE
    set_seed(cfg.SEED)
    device = get_device(device_str)
    
    print("=" * 70)
    print("🔷 IP-BASED GNN PIPELINE (E-GraphSAGE)")
    print("=" * 70)
    
    # Load preprocessed data
    data_dir = Path(cfg.DATA_DIR)
    
    print("\n📂 [1/4] Loading preprocessed data...")
    print(f"   Data directory: {data_dir}")
    
    # Load data with IPs
    df = pd.read_csv(data_dir / "data_with_ips.csv")
    idx_train = np.load(data_dir / "idx_train.npy")
    idx_val = np.load(data_dir / "idx_val.npy")
    idx_test = np.load(data_dir / "idx_test.npy")
    
    print(f"   Total samples: {len(df):,}")
    print(f"   Train: {len(idx_train):,} | Val: {len(idx_val):,} | Test: {len(idx_test):,}")
    
    # Prepare features
    print("\n🔧 [2/4] Preparing features...")
    ip_cols = [cfg.SRC_IP_COL, cfg.DST_IP_COL]
    feature_cols = [c for c in df.columns if c not in ip_cols and c != cfg.LABEL_COL]
    
    # Ensure numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    print(f"   Features: {len(feature_cols)}")
    
    # Build graph
    print("\n🔨 [3/4] Building endpoint graph...")
    data, num_nodes = create_endpoint_graph(df, feature_cols)
    
    # Use pre-split indices from preprocessing
    print("\n🎯 Using pre-split indices from preprocessing...")
    print(f"   Train edges: {len(idx_train):,}")
    print(f"   Val edges:   {len(idx_val):,}")
    print(f"   Test edges:  {len(idx_test):,}")
    
    # Train
    print("\n🚀 [4/4] Training model...")
    output_dir = args.output or cfg.OUTPUT_DIR
    test_metrics = train_ip_gnn(
        data, idx_train, idx_val, idx_test,
        device=device,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("✅ IP-BASED GNN COMPLETED")
    print("=" * 70)
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   F1 Score: {test_metrics['f1']:.4f}")
    print(f"   Output:   {output_dir}")


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
    flow_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], 
                             help='Device to use')
    flow_parser.add_argument('--output', type=str, help='Output directory')
    
    # IP command
    ip_parser = subparsers.add_parser('ip', help='Train IP-based GNN')
    ip_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], 
                           help='Device to use')
    ip_parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'flow':
            run_flow_gnn(args)
        elif args.command == 'ip':
            run_ip_gnn(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
