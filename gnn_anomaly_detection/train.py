"""
Training Module for GNN Anomaly Detection
==========================================
Module này chứa các functions để train và validate GNN models.

Bao gồm:
- Training loop với early stopping
- Validation và evaluation
- Class weight balancing cho imbalanced data
- Learning rate scheduling
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, precision_recall_curve, average_precision_score)
import os
import time
from tqdm import tqdm
import joblib

import config
from models import get_model, print_model_summary, count_parameters


class EarlyStopping:
    """
    Early stopping để tránh overfitting.

    Dừng training khi validation metric không cải thiện sau patience epochs.
    """

    def __init__(self, patience: int = None, min_delta: float = None, mode: str = 'min'):
        """
        Khởi tạo early stopping.

        Args:
            patience: Số epochs chờ đợi trước khi dừng
            min_delta: Minimum change để được coi là improvement
            mode: 'min' nếu metric cần giảm, 'max' nếu cần tăng
        """
        if patience is None:
            patience = config.EARLY_STOPPING_PATIENCE
        if min_delta is None:
            min_delta = config.MIN_DELTA

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """
        Check nếu cần early stop.

        Args:
            score: Current validation metric
            epoch: Current epoch number

        Returns:
            True nếu có improvement, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if self.mode == 'min':
            improvement = self.best_score - score > self.min_delta
        else:
            improvement = score - self.best_score > self.min_delta

        if improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Tính class weights cho imbalanced dataset.

    Args:
        labels: Tensor chứa labels

    Returns:
        Tensor chứa weights cho mỗi class
    """
    labels_np = labels.cpu().numpy()
    class_counts = np.bincount(labels_np)
    total_samples = len(labels_np)

    # Inverse frequency weighting
    weights = total_samples / (len(class_counts) * class_counts)

    return torch.FloatTensor(weights)


def train_epoch(model: nn.Module, data: Data, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> tuple:
    """
    Train một epoch.

    Args:
        model: GNN model
        data: Graph data
        optimizer: Optimizer
        criterion: Loss function
        device: Device để train

    Returns:
        Tuple (loss, accuracy)
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data)

    # Compute loss chỉ trên training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute accuracy
    pred = out[data.train_mask].argmax(dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    accuracy = correct / data.train_mask.sum().item()

    return loss.item(), accuracy


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, mask: torch.Tensor,
             criterion: nn.Module = None) -> dict:
    """
    Evaluate model trên một subset của data.

    Args:
        model: GNN model
        data: Graph data
        mask: Boolean mask cho nodes cần evaluate
        criterion: Loss function (optional)

    Returns:
        Dictionary chứa các metrics
    """
    model.eval()

    out = model(data)
    pred = out[mask].argmax(dim=1)
    y_true = data.y[mask]

    # Probabilities cho positive class
    probs = torch.exp(out[mask])[:, 1]

    # Chuyển sang numpy
    y_true_np = y_true.cpu().numpy()
    pred_np = pred.cpu().numpy()
    probs_np = probs.cpu().numpy()

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true_np, pred_np),
        'precision': precision_score(y_true_np, pred_np, zero_division=0),
        'recall': recall_score(y_true_np, pred_np, zero_division=0),
        'f1': f1_score(y_true_np, pred_np, zero_division=0),
    }

    # AUC-ROC (chỉ khi có cả 2 classes)
    if len(np.unique(y_true_np)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true_np, probs_np)
        metrics['auc_pr'] = average_precision_score(y_true_np, probs_np)
    else:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true_np, pred_np)

    # Loss
    if criterion is not None:
        loss = criterion(out[mask], y_true).item()
        metrics['loss'] = loss

    return metrics


def train(model: nn.Module, data: Data, num_epochs: int = None,
          lr: float = None, weight_decay: float = None,
          use_class_weights: bool = None, device: str = None,
          save_best: bool = True, model_path: str = None,
          verbose: bool = True) -> dict:
    """
    Train GNN model.

    Args:
        model: GNN model
        data: Graph data với train/val/test masks
        num_epochs: Số epochs
        lr: Learning rate
        weight_decay: L2 regularization
        use_class_weights: Có sử dụng class weights không
        device: Device để train
        save_best: Có lưu best model không
        model_path: Đường dẫn để lưu model
        verbose: Có in progress không

    Returns:
        Dictionary chứa training history
    """
    # Default values từ config
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY
    if use_class_weights is None:
        use_class_weights = config.USE_CLASS_WEIGHTS
    if device is None:
        device = config.DEVICE
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, 'best_model.pt')

    # Move to device
    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)

    # Loss function với class weights
    if use_class_weights:
        class_weights = compute_class_weights(data.y[data.train_mask]).to(device)
        criterion = nn.NLLLoss(weight=class_weights)
        if verbose:
            print(f"[INFO] Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.NLLLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=verbose)

    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': [],
        'best_epoch': 0,
        'best_val_f1': 0.0
    }

    # Training loop
    if verbose:
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        print(f"Device: {device}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {lr}")
        print(f"Training nodes: {data.train_mask.sum().item()}")
        print(f"Validation nodes: {data.val_mask.sum().item()}")

    start_time = time.time()
    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, device)

        # Validate
        val_metrics = evaluate(model, data, data.val_mask, criterion)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics.get('loss', 0))
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc_roc'])

        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])

        # Check for best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            history['best_epoch'] = epoch
            history['best_val_f1'] = best_val_f1

            if save_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                }, model_path)

        # Early stopping check
        improved = early_stopping(val_metrics['f1'], epoch)

        # Logging
        if verbose and (epoch % config.LOG_INTERVAL == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics.get('loss', 0):.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc_roc']:.4f}")

        # Check early stopping
        if early_stopping.early_stop:
            if verbose:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                print(f"[INFO] Best epoch: {early_stopping.best_epoch}")
            break

    training_time = time.time() - start_time
    history['training_time'] = training_time

    if verbose:
        print(f"\n[INFO] Training completed in {training_time:.2f} seconds")
        print(f"[INFO] Best validation F1: {best_val_f1:.4f} at epoch {history['best_epoch']}")

    return history


def test(model: nn.Module, data: Data, device: str = None,
         model_path: str = None, verbose: bool = True) -> dict:
    """
    Test model trên test set.

    Args:
        model: GNN model
        data: Graph data
        device: Device
        model_path: Đường dẫn đến saved model
        verbose: Có in results không

    Returns:
        Dictionary chứa test metrics
    """
    if device is None:
        device = config.DEVICE
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, 'best_model.pt')

    device = torch.device(device)

    # Load best model nếu có
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if verbose:
            print(f"[INFO] Loaded best model from epoch {checkpoint['epoch']}")

    model = model.to(device)
    data = data.to(device)

    # Evaluate on test set
    test_metrics = evaluate(model, data, data.test_mask)

    if verbose:
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")
        print(f"F1 Score:  {test_metrics['f1']:.4f}")
        print(f"AUC-ROC:   {test_metrics['auc_roc']:.4f}")
        print(f"AUC-PR:    {test_metrics['auc_pr']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Normal  Anomaly")
        cm = test_metrics['confusion_matrix']
        print(f"Actual Normal    {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Anomaly   {cm[1,0]:6d}  {cm[1,1]:6d}")

    return test_metrics


def save_results(history: dict, test_metrics: dict, file_path: str = None):
    """
    Lưu training history và test results.

    Args:
        history: Training history
        test_metrics: Test metrics
        file_path: Đường dẫn để lưu
    """
    if file_path is None:
        file_path = os.path.join(config.OUTPUT_DIR, 'training_results.pkl')

    results = {
        'history': history,
        'test_metrics': test_metrics
    }

    joblib.dump(results, file_path)
    print(f"[INFO] Saved results to: {file_path}")


def load_results(file_path: str = None) -> dict:
    """
    Load saved results.

    Args:
        file_path: Đường dẫn đến file

    Returns:
        Dictionary chứa results
    """
    if file_path is None:
        file_path = os.path.join(config.OUTPUT_DIR, 'training_results.pkl')

    return joblib.load(file_path)

