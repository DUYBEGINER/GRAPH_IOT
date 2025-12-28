"""Main training script for GNN-IDS."""

import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_ids import config
from gnn_ids.common import set_seed
from gnn_ids.data.load_csv import load_cicids_csv
from gnn_ids.data.preprocess import split_and_scale
from gnn_ids.graph.knn_graph import build_knn_graph
from gnn_ids.model.sage import FlowGraphSAGE
from gnn_ids.train import (
    train_one_epoch, 
    evaluate, 
    evaluate_logits,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint
)

logger = logging.getLogger(__name__)


def main(args):
    """Main training pipeline."""
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    logger.info(f"Random seed set to {config.SEED}")
    
    # Validate and print configuration
    config.validate_config()
    config.print_config()
    
    # Get device
    device = config.get_device()
    
    # ============================
    # 1. Load Data
    # ============================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)
    
    if args.csv_path:
        csv_path = args.csv_path
    elif config.CSV_PATH:
        csv_path = config.CSV_PATH
    else:
        raise ValueError("CSV path must be provided via --csv_path or config.CSV_PATH")
    
    X, y, feature_cols = load_cicids_csv(
        csv_path=csv_path,
        max_samples=config.MAX_SAMPLES,
        seed=config.SEED
    )
    
    # ============================
    # 2. Preprocess Data
    # ============================
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("=" * 60)
    
    x_tensor, y_tensor, idx_train, idx_val, idx_test, scaler = split_and_scale(
        X=X,
        y=y,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED
    )
    
    # ============================
    # 3. Build Graph
    # ============================
    logger.info("=" * 60)
    logger.info("STEP 3: Building KNN Graph")
    logger.info("=" * 60)
    
    edge_index, edge_weight = build_knn_graph(
        X_scaled=x_tensor.numpy(),
        k=config.K_NEIGHBORS,
        metric=config.GRAPH_METRIC
    )
    
    # Create PyG Data object
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_weight,  # Store edge weights
        y=y_tensor
    )
    
    # Create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    logger.info(f"Graph created: {data}")
    
    # ============================
    # 4. Create DataLoaders
    # ============================
    logger.info("=" * 60)
    logger.info("STEP 4: Creating DataLoaders")
    logger.info("=" * 60)
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=config.NUM_NEIGHBORS,
        batch_size=config.BATCH_SIZE,
        input_nodes=train_mask,
        shuffle=True,
    )
    
    # Full neighbor loader for inference
    full_loader = NeighborLoader(
        data,
        num_neighbors=[-1],  # Use all neighbors for inference
        batch_size=config.BATCH_SIZE,
        input_nodes=None,
        shuffle=False,
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    
    # ============================
    # 5. Initialize Model
    # ============================
    logger.info("=" * 60)
    logger.info("STEP 5: Initializing Model")
    logger.info("=" * 60)
    
    model = FlowGraphSAGE(
        in_dim=data.num_node_features,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # ============================
    # 6. Training Loop
    # ============================
    logger.info("=" * 60)
    logger.info("STEP 6: Training Model")
    logger.info("=" * 60)
    
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.MIN_DELTA,
        mode='max'  # Maximize validation AUC
    )
    
    best_val_auc = 0.0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Evaluate on validation set every epoch
        if epoch % 1 == 0:
            logits = evaluate_logits(data, model, full_loader, device)
            val_acc, val_auc = evaluate(data, logits, val_mask, device, name="VAL")
            
            # Save best model
            if config.SAVE_BEST_MODEL and val_auc > best_val_auc:
                best_val_auc = val_auc
                save_checkpoint(
                    model, optimizer, epoch, val_acc, val_auc,
                    config.MODEL_SAVE_PATH
                )
                logger.info(f"New best model saved! Val AUC: {val_auc:.4f}")
            
            # Early stopping check
            if early_stopping(val_auc):
                logger.info("Early stopping triggered!")
                break
    
    # ============================
    # 7. Final Evaluation
    # ============================
    logger.info("=" * 60)
    logger.info("STEP 7: Final Evaluation on Test Set")
    logger.info("=" * 60)
    
    # Load best model if saved
    if config.SAVE_BEST_MODEL and config.MODEL_SAVE_PATH.exists():
        logger.info("Loading best model for final evaluation...")
        load_checkpoint(model, optimizer, config.MODEL_SAVE_PATH)
    
    # Final test evaluation
    logits = evaluate_logits(data, model, full_loader, device)
    test_acc, test_auc = evaluate(data, logits, test_mask, device, name="TEST")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best Val AUC: {best_val_auc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN-IDS model")
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to CICIDS CSV file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=config.MAX_SAMPLES,
        help="Maximum number of samples to use"
    )
    
    args = parser.parse_args()
    
    # Override config if provided
    if args.max_samples:
        config.MAX_SAMPLES = args.max_samples
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
