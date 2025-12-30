"""Training logic for Endpoint-based GNN."""

import logging
import torch
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader
from pathlib import Path
from typing import Dict
import numpy as np
import time
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from .model import EGraphSAGE
from .utils import compute_metrics, EarlyStopping, get_device, save_metrics_plots, save_metrics_report

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
    
    # Calculate pos_weight from TRAINING edges only
    train_edge_y = data.edge_y[train_edges[0]]
    pos = (train_edge_y == 1).sum().item()
    neg = (train_edge_y == 0).sum().item()
    pos_weight = neg / pos if pos > 0 else 1.0
    
    logger.info(f"Training edge distribution: Benign={neg}, Attack={pos}")
    logger.info(f"Calculated pos_weight: {pos_weight:.4f}")
    
    # Model
    model = EGraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        aggr=config['model'].get('aggr', 'mean')
    ).to(device)
    
    # Optimizer & Loss (BCEWithLogitsLoss with pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    
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
            
            logits = model(
                batch.x, 
                batch.edge_index, 
                batch.edge_attr,
                batch.edge_label_index
            )
            
            loss = criterion(logits, batch.edge_label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        val_loss, val_metrics = evaluate_edges(
            model, data, val_edges, criterion, device
        )
        
        if epoch % 5 == 0 or epoch == 1:
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
    
    # Tune threshold on validation edges
    logger.info("\n" + "=" * 70)
    logger.info("TUNING THRESHOLD ON VALIDATION")
    logger.info("=" * 70)
    best_threshold = tune_threshold_edges(model, data, val_edges, device)
    logger.info(f"Best threshold: {best_threshold:.4f}\n")
    
    # Final test
    logger.info("=" * 70)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 70)
    
    start_time = time.time()
    test_loss, test_metrics, y_true, y_pred, y_probs = evaluate_edges_with_predictions(
        model, data, test_edges, criterion, best_threshold, device
    )
    inference_time = time.time() - start_time
    latency_per_sample = inference_time / len(y_true)
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    if 'auc' in test_metrics:
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test FAR: {test_metrics['far']:.4f}")
    logger.info(f"\n⏱️  Inference Performance:")
    logger.info(f"   Total time: {inference_time:.2f}s")
    logger.info(f"   Latency per sample: {latency_per_sample*1000:.4f}ms")
    logger.info(f"   Throughput: {len(y_true)/inference_time:.2f} samples/sec")
    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
    
    # Save visualizations and reports
    output_dir = Path(config.get('output_dir', 'output/endpoint_gnn'))
    logger.info(f"\n💾 Saving results to {output_dir}/")
    
    save_metrics_plots(y_true, y_pred, y_probs, test_metrics, str(output_dir))
    save_metrics_report(test_metrics, str(output_dir), 
                       y_true, y_pred, latency=latency_per_sample)
    
    logger.info("\n" + "="*70)
    logger.info("✨ ALL DONE!")
    logger.info("="*70 + "\n")
    
    return test_metrics


def tune_threshold_edges(model, data, edge_indices, device):
    """Find optimal threshold on validation edges to maximize F1 score."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, edge_indices)
        probs = torch.sigmoid(logits).cpu().numpy()
        true = data.edge_y[edge_indices[0]].cpu().numpy()
    
    # Search for best threshold
    best_t, best_f1 = 0.5, 0.0
    best_precision, best_recall = 0.0, 0.0
    
    for t in np.linspace(0.01, 0.99, 99):
        pred = (probs >= t).astype(int)
        f1 = f1_score(true, pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_precision = precision_score(true, pred, zero_division=0)
            best_recall = recall_score(true, pred, zero_division=0)
    
    logger.info(f"Best threshold = {best_t:.4f}")
    logger.info(f"Val Precision  = {best_precision:.4f}")
    logger.info(f"Val Recall     = {best_recall:.4f}")
    logger.info(f"Val F1         = {best_f1:.4f}")
    
    return best_t


def evaluate_edges(model, data, edge_indices, criterion, device, threshold=0.5):
    """Evaluate model on given edges."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, edge_indices)
        
        # Get edge labels
        edge_labels = data.edge_y[edge_indices[0]]
        
        loss = criterion(logits, edge_labels.float()).item()
        
        # Apply sigmoid + threshold
        probs = torch.sigmoid(logits).cpu().numpy()
        pred = (probs >= threshold).astype(int)
        true = edge_labels.cpu().numpy()
        
        metrics = compute_metrics(true, pred, y_probs=probs)
        
    return loss, metrics


def evaluate_edges_with_predictions(model, data, edge_indices, criterion, threshold, device):
    """Evaluate and return predictions with probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, edge_indices)
        
        edge_labels = data.edge_y[edge_indices[0]]
        
        loss = criterion(logits, edge_labels.float()).item()
        
        # Apply sigmoid + threshold
        probs = torch.sigmoid(logits).cpu().numpy()
        pred = (probs >= threshold).astype(int)
        true = edge_labels.cpu().numpy()
        
        metrics = compute_metrics(true, pred, y_probs=probs)
        
    return loss, metrics, true, pred, probs
