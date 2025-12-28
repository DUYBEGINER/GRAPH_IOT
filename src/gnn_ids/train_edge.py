"""Training utilities for E-GraphSAGE edge classification."""

import time
import logging
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from torch_geometric.loader import LinkNeighborLoader

from .utils.metrics import compute_metrics, compute_class_weights

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when metric doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, metric: str = "f1"):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor (f1, accuracy, loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = "max" if metric != "loss" else "min"
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                self.early_stop = True
                return True
        
        return False


def train_epoch_edge(
    model,
    loader: LinkNeighborLoader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train E-GraphSAGE for one epoch on edge classification.
    
    Args:
        model: E-GraphSAGE model
        loader: LinkNeighborLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        epoch: Current epoch
        
    Returns:
        avg_loss: Average loss
        avg_acc: Average accuracy
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    t0 = time.time()
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        # batch.edge_label_index contains the edges to predict
        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            edge_label_index=batch.edge_label_index
        )
        
        # Loss on edge labels
        loss = criterion(logits, batch.edge_label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * batch.edge_label.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == batch.edge_label).sum().item()
        total_examples += batch.edge_label.size(0)
    
    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    elapsed = time.time() - t0
    
    logger.info(f"Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | "
               f"Train Acc: {avg_acc:.4f} | Time: {elapsed:.2f}s")
    
    return avg_loss, avg_acc


def evaluate_epoch_edge(
    model,
    loader: LinkNeighborLoader,
    criterion,
    device: torch.device,
    task_type: str = "binary"
) -> Tuple[float, dict]:
    """
    Evaluate E-GraphSAGE on edge classification.
    
    Args:
        model: E-GraphSAGE model
        loader: LinkNeighborLoader
        criterion: Loss function
        device: Device
        task_type: "binary" or "multiclass"
        
    Returns:
        avg_loss: Average loss
        metrics: Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total_examples = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                edge_label_index=batch.edge_label_index
            )
            
            # Loss
            loss = criterion(logits, batch.edge_label)
            total_loss += loss.item() * batch.edge_label.size(0)
            total_examples += batch.edge_label.size(0)
            
            # Predictions
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.edge_label.cpu().numpy())
    
    avg_loss = total_loss / total_examples
    
    # Compute metrics
    import numpy as np
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        task_type=task_type
    )
    metrics["loss"] = avg_loss
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded from {path}")
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
