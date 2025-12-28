"""Main training script for GNN-IDS with support for both modes.

Modes:
- flow: Node = flow record, Edge = KNN, Task = node classification
- endpoint: Node = endpoint (IP:Port), Edge = flow, Task = edge classification
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_ids.config_loader import load_config, get_device, print_config
from gnn_ids.common import set_seed
from gnn_ids.utils.metrics import compute_class_weights

logger = logging.getLogger(__name__)


def run_flow_mode(config: dict, device: torch.device):
    """Run flow-based node classification pipeline."""
    from gnn_ids.data.load_csv import load_cicids_csv
    from gnn_ids.data.preprocess import split_and_scale
    from gnn_ids.graph.knn_graph import build_knn_graph
    from gnn_ids.model.sage import FlowGraphSAGE
    from gnn_ids.train import train_one_epoch, evaluate
    from gnn_ids.train import EarlyStopping, save_checkpoint
    from torch_geometric.data import Data
    
    logger.info("=" * 70)
    logger.info("FLOW-BASED MODE: Node Classification")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n[1/5] Loading CSV data...")
    X, y, feature_cols = load_cicids_csv(
        csv_path=config['data']['csv_path'],
        max_samples=config['data']['max_samples'],
        seed=config['project']['seed']
    )
    
    # Preprocess
    logger.info("\n[2/5] Preprocessing data...")
    x_tensor, y_tensor, idx_train, idx_val, idx_test, scaler = split_and_scale(
        X=X,
        y=y,
        val_ratio=config['data']['val_split'],
        test_ratio=config['data']['test_split'],
        seed=config['project']['seed']
    )
    
    # Build KNN graph
    logger.info("\n[3/5] Building KNN graph...")
    edge_index, edge_weight = build_knn_graph(
        X_scaled=x_tensor.numpy(),
        k=config['flow_graph']['k_neighbors'],
        metric=config['flow_graph']['metric']
    )
    
    # Create PyG Data
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y_tensor
    )
    
    # Create masks
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Keep data on CPU for NeighborLoader (MPS doesn't support CSC conversion)
    # Batches will be moved to device during training
    if device.type == 'mps':
        logger.info("Note: Keeping graph data on CPU for MPS compatibility. Batches will be moved to MPS during training.")
        data_for_loader = data
    else:
        data_for_loader = data.to(device)
    
    # Initialize model
    logger.info("\n[4/5] Initializing model...")
    model = FlowGraphSAGE(
        in_dim=x_tensor.shape[1],
        hidden_dim=config['flow_model']['hidden_dim'],
        num_classes=config['task']['num_classes'],
        num_layers=config['flow_model']['num_layers'],
        dropout=config['flow_model']['dropout']
    ).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    
    # Class weighting
    use_class_weights = config.get('task', {}).get('class_weight') == 'balanced'
    if use_class_weights:
        class_weights = compute_class_weights(y_tensor, config['task']['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # DataLoader
    train_loader = NeighborLoader(
        data_for_loader,
        num_neighbors=config['training']['num_neighbors'],
        batch_size=config['training']['batch_size'],
        input_nodes=train_mask,
        shuffle=True
    )
    
    # Training loop
    logger.info("\n[5/5] Training model...")
    early_stopping = EarlyStopping(
        patience=config.get('training', {}).get('early_stopping', {}).get('patience', 10),
        min_delta=config.get('training', {}).get('early_stopping', {}).get('min_delta', 0.001),
        mode='max'
    )
    
    best_f1 = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        val_loss, val_metrics = evaluate(model, data_for_loader, data_for_loader.val_mask, device)
        
        logger.info(f"Epoch {epoch:3d} | Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_metrics['accuracy']:.4f} | "
                   f"Val F1: {val_metrics.get('f1', 0):.4f}")
        
        # Save best model
        val_f1 = val_metrics.get('f1', val_metrics['accuracy'])
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = Path(config.get('output', {}).get('checkpoint_dir', 'output/checkpoints')) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, val_metrics, str(save_path))
        
        # Early stopping
        if early_stopping(val_f1):
            break
    
    # Final evaluation
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 70)
    
    from gnn_ids.train import evaluate_logits
    test_loss, test_metrics, y_true, y_pred = evaluate_logits(
        model, data_for_loader, data_for_loader.test_mask, device
    )
    
    from gnn_ids.utils.metrics import print_metrics, get_classification_report
    print_metrics(test_metrics, prefix="Test")
    print("\nTest Classification Report:")
    print(get_classification_report(y_true, y_pred))


def run_endpoint_mode(config: dict, device: torch.device):
    """Run endpoint-based edge classification pipeline."""
    from gnn_ids.data.load_csv import load_cicids_csv
    from gnn_ids.data.graph_build import create_endpoint_graph
    from gnn_ids.model.e_graphsage import EGraphSAGE
    from gnn_ids.train_edge import (
        train_epoch_edge,
        evaluate_epoch_edge,
        EarlyStopping,
        save_checkpoint
    )
    from gnn_ids.eval import evaluate_edge_model, save_predictions
    from sklearn.preprocessing import StandardScaler
    
    logger.info("=" * 70)
    logger.info("ENDPOINT-BASED MODE: Edge Classification")
    logger.info("=" * 70)
    
    # Load CSV
    logger.info("\n[1/5] Loading CSV data...")
    csv_path = config['data']['csv_path']
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from {csv_path}")
    
    # Sample if needed
    max_samples = config['data']['max_samples']
    if len(df) > max_samples:
        logger.info(f"Sampling {max_samples} records...")
        df = df.sample(n=max_samples, random_state=config['project']['seed']).reset_index(drop=True)
    
    # Process features
    logger.info("\n[2/5] Processing features...")
    label_col = config['data']['label_col']
    drop_cols = config['data']['drop_cols'] + [label_col]
    
    # Select numeric features
    feature_cols = [c for c in df.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    logger.info(f"Number of edge features: {len(feature_cols)}")
    
    # Handle inf/nan
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Build graph
    logger.info("\n[3/5] Building endpoint graph...")
    data, endpoint_to_idx, num_nodes = create_endpoint_graph(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        src_ip_col=config['data']['src_ip_col'],
        src_port_col=config['data']['src_port_col'],
        dst_ip_col=config['data']['dst_ip_col'],
        dst_port_col=config['data']['dst_port_col'],
        mapping_mode=config['endpoint_graph']['mapping_mode'],
        ip_random_map=config['endpoint_graph']['anti_leakage']['enabled'],
        map_scope=config['endpoint_graph']['anti_leakage']['map_scope'],
        task_type=config['task']['type'],
        seed=config['project']['seed']
    )
    
    # Split edges
    logger.info("\n[4/5] Splitting edges...")
    num_edges = data.edge_index.shape[1]
    perm = torch.randperm(num_edges)
    
    test_size = int(num_edges * config['data']['test_split'])
    val_size = int(num_edges * config['data']['val_split'])
    train_size = num_edges - test_size - val_size
    
    train_edge_indices = perm[:train_size]
    val_edge_indices = perm[train_size:train_size + val_size]
    test_edge_indices = perm[train_size + val_size:]
    
    # Create masks
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    train_mask[train_edge_indices] = True
    val_mask[val_edge_indices] = True
    test_mask[test_edge_indices] = True
    
    logger.info(f"Train edges: {train_mask.sum()}")
    logger.info(f"Val edges: {val_mask.sum()}")
    logger.info(f"Test edges: {test_mask.sum()}")
    
    # Keep data on CPU for LinkNeighborLoader (MPS doesn't support CSC conversion)
    # Batches will be moved to device during training
    if device.type == 'mps':
        logger.info("Note: Keeping graph data on CPU for MPS compatibility. Batches will be moved to MPS during training.")
        loader_device = torch.device('cpu')
        data_for_loader = data
    else:
        loader_device = device
        data_for_loader = data.to(device)
    
    # Create LinkNeighborLoader
    train_loader = LinkNeighborLoader(
        data_for_loader,
        num_neighbors=config['training']['num_neighbors'],
        edge_label_index=data_for_loader.edge_index[:, train_mask],
        edge_label=data_for_loader.edge_y[train_mask],
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = LinkNeighborLoader(
        data_for_loader,
        num_neighbors=config['training']['num_neighbors'],
        edge_label_index=data_for_loader.edge_index[:, val_mask],
        edge_label=data_for_loader.edge_y[val_mask],
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    logger.info("\n[5/5] Initializing E-GraphSAGE model...")
    model = EGraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=config['endpoint_model']['hidden_dim'],
        num_classes=config['task']['num_classes'],
        num_layers=config['endpoint_model']['num_layers'],
        dropout=config['endpoint_model']['dropout'],
        aggr=config['endpoint_model']['aggregator']
    ).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    
    # Class weighting
    use_class_weights = config.get('task', {}).get('class_weight') == 'balanced'
    if use_class_weights:
        class_weights = compute_class_weights(
            data.edge_y[train_mask],
            config['task']['num_classes']
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING E-GRAPHSAGE")
    logger.info("=" * 70)
    
    early_stopping = EarlyStopping(
        patience=config.get('training', {}).get('early_stopping', {}).get('patience', 10),
        min_delta=config.get('training', {}).get('early_stopping', {}).get('min_delta', 0.001),
        metric=config.get('training', {}).get('early_stopping', {}).get('metric', 'f1')
    )
    
    best_f1 = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch_edge(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        val_loss, val_metrics = evaluate_epoch_edge(
            model, val_loader, criterion, device, config['task']['type']
        )
        
        val_f1 = val_metrics.get('f1', val_metrics.get('f1_macro', val_metrics['accuracy']))
        
        logger.info(f"Epoch {epoch:3d} | Val Loss: {val_loss:.4f} | "
                   f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = Path(config.get('output', {}).get('checkpoint_dir', 'output/checkpoints')) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, val_metrics, str(save_path))
        
        # Early stopping
        if early_stopping(val_f1):
            break
    
    # Final evaluation
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 70)
    
    test_metrics, y_true, y_pred = evaluate_edge_model(
        model, data, test_mask, device, config['task']['type'], prefix="Test"
    )
    
    # Save predictions if enabled
    if config.get('evaluation', {}).get('save_predictions', False):
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / "predictions.csv"
        save_predictions(y_true, y_pred, str(pred_path))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GNN-IDS Training")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--csv_path", type=str, default=None,
                       help="Path to CSV file (overrides config)")
    parser.add_argument("--mode", type=str, default=None, choices=["flow", "endpoint"],
                       help="Mode (overrides config)")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.csv_path:
        config['data']['csv_path'] = args.csv_path
    if args.mode:
        config['mode'] = args.mode
    
    # Validate CSV path
    if not config['data']['csv_path']:
        raise ValueError("CSV path must be provided via config or --csv_path")
    
    # Set seed
    set_seed(config['project']['seed'])
    
    # Get device
    device = get_device(config['project']['device'])
    
    # Print config
    print_config(config)
    
    # Run appropriate mode
    if config['mode'] == "flow":
        run_flow_mode(config, device)
    elif config['mode'] == "endpoint":
        run_endpoint_mode(config, device)
    else:
        raise ValueError(f"Unknown mode: {config['mode']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
