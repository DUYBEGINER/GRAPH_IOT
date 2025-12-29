"""Training logic for Endpoint-based GNN."""

import logging
import torch
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.metrics import classification_report

from .model import EGraphSAGE
from .utils import compute_metrics, EarlyStopping, get_device

logger = logging.getLogger(__name__)


def train_endpoint_gnn(
    data,
    train_edges,
    val_edges,
    test_edges,
    config: dict,
    device: torch.device
) -> Dict:
    """
    Train Endpoint-based GNN model.
    
    Args:
        data: PyG Data with x, edge_index, edge_attr, edge_y
        train_edges, val_edges, test_edges: Edge indices for each split
        config: Configuration dict
        device: Torch device
        
    Returns:
        Dictionary with final test metrics
    """
    logger.info("=" * 70)
    logger.info("TRAINING ENDPOINT-BASED GNN")
    logger.info("=" * 70)
    
    # Move data to device
    data = data.to(device)
    
    # Model
    model = EGraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        aggr=config['model'].get('aggr', 'mean')
    ).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    criterion = nn.CrossEntropyLoss()
    
    # DataLoader for mini-batch training
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=config['training']['num_neighbors'],
        edge_label_index=train_edges,
        edge_label=data.edge_y[train_edges[0]],  # Assume edge_y indexed by source
        batch_size=config['training']['batch_size'],
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
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(
                batch.x, 
                batch.edge_index, 
                batch.edge_attr,
                batch.edge_label_index
            )
            
            loss = criterion(out, batch.edge_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        val_loss, val_metrics = evaluate_edges(
            model, data, val_edges, device
        )
        
        logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = Path(config.get('output_dir', 'output/endpoint_gnn')) / 'best_model.pt'
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
    
    test_loss, test_metrics, y_true, y_pred = evaluate_edges_with_predictions(
        model, data, test_edges, device
    )
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
    
    return test_metrics


def evaluate_edges(model, data, edge_indices, device):
    """Evaluate model on given edges."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, edge_indices)
        
        # Get edge labels (assuming edge_y is indexed consistently)
        edge_labels = data.edge_y[edge_indices[0]].cpu()
        
        loss = nn.functional.cross_entropy(out.cpu(), edge_labels).item()
        
        pred = out.cpu().argmax(dim=1).numpy()
        true = edge_labels.numpy()
        
        metrics = compute_metrics(true, pred)
        
    return loss, metrics


def evaluate_edges_with_predictions(model, data, edge_indices, device):
    """Evaluate and return predictions."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, edge_indices)
        
        edge_labels = data.edge_y[edge_indices[0]].cpu()
        
        loss = nn.functional.cross_entropy(out.cpu(), edge_labels).item()
        
        pred = out.cpu().argmax(dim=1).numpy()
        true = edge_labels.numpy()
        
        metrics = compute_metrics(true, pred)
        
    return loss, metrics, true, pred
