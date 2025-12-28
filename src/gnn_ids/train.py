"""Training and evaluation utilities."""

import time
import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/AUC
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
                return True
        
        return False


def train_one_epoch(
    model, 
    loader, 
    optimizer, 
    criterion, 
    device, 
    epoch: int
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)
        
        # Only compute loss on seed nodes (first batch_size nodes)
        out_seed = out[:batch.batch_size]
        y_seed = batch.y[:batch.batch_size]
        
        loss = criterion(out_seed, y_seed)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.batch_size
        pred = out_seed.argmax(dim=-1)
        total_correct += (pred == y_seed).sum().item()
        total_examples += batch.batch_size
    
    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    t1 = time.time()
    
    logger.info(
        f"[Epoch {epoch:03d}] Train Loss: {avg_loss:.4f}, "
        f"Train Acc: {avg_acc:.4f}, Time: {t1-t0:.1f}s"
    )
    
    return avg_loss, avg_acc


def evaluate_logits(data, model, full_loader, device):
    """
    Run inference to get logits for all nodes.
    
    Args:
        data: PyTorch Geometric Data object
        model: Trained model
        full_loader: DataLoader for inference
        device: Device
        
    Returns:
        Logits tensor [num_nodes, num_classes]
    """
    model.eval()
    logits = torch.zeros((data.num_nodes, 2), device=device)
    
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            logits[batch.n_id] = out
    
    return logits


def evaluate(
    data, 
    logits, 
    mask, 
    device, 
    name: str = "VAL"
) -> Tuple[float, float]:
    """
    Evaluate model performance on a dataset split.
    
    Args:
        data: PyTorch Geometric Data object
        logits: Model output logits
        mask: Boolean mask for the split
        device: Device
        name: Name of the split (for logging)
        
    Returns:
        Accuracy and AUC score
    """
    y_true = data.y.to(device)[mask]
    y_pred = logits[mask].argmax(dim=-1)
    y_prob = F.softmax(logits[mask], dim=-1)[:, 1]
    
    acc = (y_pred == y_true).float().mean().item()
    
    try:
        auc = roc_auc_score(y_true.cpu().numpy(), y_prob.cpu().numpy())
    except ValueError:
        auc = float("nan")
        logger.warning(f"Could not compute AUC for {name} set")
    
    # Confusion matrix
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
    logger.info(f"[{name}] Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Detailed classification report
    report = classification_report(
        y_true.cpu().numpy(), 
        y_pred.cpu().numpy(),
        target_names=['Benign', 'Attack'],
        digits=4
    )
    logger.info(f"Classification Report:\n{report}")
    
    return acc, auc


def save_checkpoint(
    model, 
    optimizer, 
    epoch: int, 
    val_acc: float, 
    val_auc: float,
    save_path: Path
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        val_acc: Validation accuracy
        val_auc: Validation AUC
        save_path: Path to save checkpoint
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_auc': val_auc,
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model, 
    optimizer, 
    checkpoint_path: Path
) -> Tuple[int, float, float]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint
        
    Returns:
        epoch, val_acc, val_auc from checkpoint
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")
    
    return checkpoint['epoch'], checkpoint['val_acc'], checkpoint['val_auc']
