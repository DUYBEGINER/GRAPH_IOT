"""Training logic for Flow-based GNN."""

import logging
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import classification_report

from .model import FlowGraphSAGE
from .graph import build_knn_graph
from .utils import compute_metrics, EarlyStopping, get_device

logger = logging.getLogger(__name__)


def train_flow_gnn(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    config: dict,
    device: torch.device
) -> Dict:
    """
    Train Flow-based GNN model.
    
    Args:
        x_tensor: Node features [N, F]
        y_tensor: Labels [N]
        edge_index: Edge indices [2, E]
        edge_weight: Edge weights [E]
        train_mask, val_mask, test_mask: Boolean masks
        config: Configuration dict
        device: Torch device
        
    Returns:
        Dictionary with final test metrics
    """
    logger.info("=" * 70)
    logger.info("TRAINING FLOW-BASED GNN")
    logger.info("=" * 70)
    
    # Create PyG Data
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # MPS compatibility
    if device.type == 'mps':
        logger.info("Using MPS - keeping data on CPU for NeighborLoader")
        data_for_loader = data
    else:
        data_for_loader = data.to(device)
    
    # Model
    model = FlowGraphSAGE(
        in_dim=x_tensor.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
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
    early_stopping = EarlyStopping(
        patience=config['training'].get('patience', 10),
        min_delta=config['training'].get('min_delta', 0.001)
    )
    
    best_f1 = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device) if device.type != 'mps' else batch
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        val_loss, val_metrics = evaluate(model, data_for_loader, val_mask, device)
        
        logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = Path(config.get('output_dir', 'output/flow_gnn')) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, save_path)
        
        # Early stopping
        if early_stopping(val_metrics['f1']):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final test
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 70)
    
    test_loss, test_metrics, y_true, y_pred = evaluate_with_predictions(
        model, data_for_loader, test_mask, device
    )
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
    
    return test_metrics


def evaluate(model, data, mask, device):
    """Evaluate model on given mask."""
    model.eval()
    with torch.no_grad():
        data_device = data.to(device) if device.type != 'mps' else data
        out = model(data_device.x, data_device.edge_index)
        
        masked_out = out[mask].cpu()
        masked_y = data.y[mask].cpu()
        
        loss = nn.functional.cross_entropy(masked_out, masked_y).item()
        
        pred = masked_out.argmax(dim=1).numpy()
        true = masked_y.numpy()
        
        metrics = compute_metrics(true, pred)
        
    return loss, metrics


def evaluate_with_predictions(model, data, mask, device):
    """Evaluate and return predictions."""
    model.eval()
    with torch.no_grad():
        data_device = data.to(device) if device.type != 'mps' else data
        out = model(data_device.x, data_device.edge_index)
        
        masked_out = out[mask].cpu()
        masked_y = data.y[mask].cpu()
        
        loss = nn.functional.cross_entropy(masked_out, masked_y).item()
        
        pred = masked_out.argmax(dim=1).numpy()
        true = masked_y.numpy()
        
        metrics = compute_metrics(true, pred)
        
    return loss, metrics, true, pred
