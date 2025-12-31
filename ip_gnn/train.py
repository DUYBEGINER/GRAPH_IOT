"""Training logic for IP-based GNN (E-GraphSAGE)."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score

from . import config as cfg
from .model import EGraphSAGE
from .utils import compute_metrics, EarlyStopping, get_device, save_metrics_plots, save_metrics_report, set_seed


def train_ip_gnn(
    data,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device = None,
    output_dir: str = None
) -> Dict:
    """Train IP-based GNN (E-GraphSAGE) model.
    
    Args:
        data: PyG Data with x, edge_index, edge_attr, edge_y
        train_idx, val_idx, test_idx: Edge indices for each split (from preprocessing)
        device: Torch device (default: auto-detect)
        output_dir: Output directory (default from config)
        
    Returns:
        Dictionary with test metrics
    """
    # Setup
    if device is None:
        device = get_device(cfg.DEVICE)
    if output_dir is None:
        output_dir = cfg.OUTPUT_DIR
    
    set_seed(cfg.SEED)
    
    print("\n" + "=" * 70)
    print("🚀 TRAINING IP-BASED GNN (E-GraphSAGE)")
    print("=" * 70)
    
    # Move data to device
    data = data.to(device)
    
    # Convert indices to tensors
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)
    
    # Calculate class weights for CrossEntropyLoss
    train_edge_y = data.edge_y[train_idx_t]
    pos = (train_edge_y == 1).sum().item()
    neg = (train_edge_y == 0).sum().item()
    total = pos + neg
    # Weight inversely proportional to class frequency
    class_weights = torch.tensor([total / (2 * neg), total / (2 * pos)], device=device)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training edges:   {len(train_idx):,}")
    print(f"   Validation edges: {len(val_idx):,}")
    print(f"   Test edges:       {len(test_idx):,}")
    print(f"   Class distribution: Benign={neg:,} ({neg/(neg+pos)*100:.1f}%), Attack={pos:,} ({pos/(neg+pos)*100:.1f}%)")
    print(f"   Class weights:      [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")
    
    # Model
    print(f"\n🏗️  Model Configuration:")
    print(f"   Hidden dim:  {cfg.HIDDEN_DIM}")
    print(f"   Num layers:  {cfg.NUM_LAYERS}")
    print(f"   Dropout:     {cfg.DROPOUT}")
    print(f"   Aggregation: {cfg.AGGR}")
    
    model = EGraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        aggr=cfg.AGGR
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")
    print(f"   Device:       {device}")
    
    # Optimizer & Loss (Softmax + CrossEntropy as per paper)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs:         {cfg.EPOCHS}")
    print(f"   Learning rate:  {cfg.LEARNING_RATE}")
    print(f"   Weight decay:   {cfg.WEIGHT_DECAY}")
    print(f"   Early stopping: {cfg.PATIENCE} epochs")
    print(f"   Training mode:  Full-batch (1 forward/epoch)")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA)
    
    # Training history
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_accuracy': []
    }
    
    print(f"\n🔥 Starting Training (Full-batch mode)...")
    print("-" * 70)
    
    epoch_pbar = tqdm(range(1, cfg.EPOCHS + 1), desc="Training", unit="epoch", ncols=100)
    
    for epoch in epoch_pbar:
        # Train: 1 forward pass per epoch
        model.train()
        optimizer.zero_grad()
        
        # Single forward pass on entire graph for all edges
        logits_all = model(data.x, data.edge_index, data.edge_attr)  # [num_edges, num_classes]
        
        # Compute loss only on training edges
        loss = criterion(logits_all[train_idx_t], data.edge_y[train_idx_t])
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # Validate: use logits from full forward pass
        val_loss, val_metrics = evaluate_edges_fullbatch(model, data, val_idx_t, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'loss': f"{train_loss:.4f}",
            'val_f1': f"{val_metrics['f1']:.4f}",
            'val_acc': f"{val_metrics['accuracy']:.4f}"
        })
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = Path(output_dir) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, save_path)
        
        # Early stopping
        if early_stopping(val_metrics['f1']):
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break
    
    print(f"\n✅ Training completed! Best validation F1: {best_f1:.4f}")
    
    # Tune threshold
    print("\n" + "=" * 70)
    print("🎯 TUNING DECISION THRESHOLD")
    print("=" * 70)
    best_threshold = tune_threshold_edges(model, data, val_idx_t, device)
    print(f"✅ Optimal threshold: {best_threshold:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("🧪 FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    start_time = time.time()
    test_loss, test_metrics, y_true, y_pred, y_probs = evaluate_edges_with_predictions(
        model, data, test_idx_t, criterion, best_threshold, device
    )
    inference_time = time.time() - start_time
    latency_per_sample = inference_time / len(y_true)
    
    print(f"\n📈 Test Results:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   AUC:       {test_metrics.get('auc', 0):.4f}")
    print(f"   FAR:       {test_metrics['far']:.4f}")
    print(f"\n⏱️  Inference Performance:")
    print(f"   Total time:   {inference_time:.2f}s")
    print(f"   Latency:      {latency_per_sample*1000:.4f} ms/sample")
    print(f"   Throughput:   {len(y_true)/inference_time:.2f} samples/sec")
    
    # Save results
    output_path = Path(output_dir)
    print(f"\n💾 Saving results to {output_path}/")
    
    save_metrics_plots(y_true, y_pred, y_probs, test_metrics, 
                       str(output_path), history=history,
                       latency_ms=latency_per_sample * 1000)
    save_metrics_report(test_metrics, str(output_path), 
                        y_true, y_pred, latency=latency_per_sample)
    
    print("\n" + "=" * 70)
    print("✨ ALL DONE!")
    print("=" * 70 + "\n")
    
    return test_metrics


def tune_threshold_edges(model, data, edge_indices, device):
    """Find optimal threshold to maximize F1 score.
    
    With softmax output, we tune threshold on P(attack) = softmax(logits)[:, 1]
    """
    model.eval()
    
    with torch.no_grad():
        # Single forward pass for all edges
        logits_all = model(data.x, data.edge_index, data.edge_attr)  # [num_edges, num_classes]
        # Get logits for validation edges
        logits = logits_all[edge_indices]
        # Softmax -> probability of class 1 (attack)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        true = data.edge_y[edge_indices].cpu().numpy()
    
    best_t, best_f1 = 0.5, 0.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in tqdm(thresholds, desc="   Searching threshold", ncols=100, leave=False):
        pred = (probs >= t).astype(int)
        f1 = f1_score(true, pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    
    return best_t


def evaluate_edges_fullbatch(model, data, edge_indices, criterion, device, threshold=0.5):
    """Evaluate model on given edges using full-batch forward pass."""
    model.eval()
    
    with torch.no_grad():
        # Single forward pass for all edges
        logits_all = model(data.x, data.edge_index, data.edge_attr)  # [num_edges, num_classes]
        # Get logits for evaluation edges
        logits = logits_all[edge_indices]
        edge_labels = data.edge_y[edge_indices]
        loss = criterion(logits, edge_labels.long()).item()
        
        # Softmax -> probability of class 1 (attack)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = (probs >= threshold).astype(int)
        true = edge_labels.cpu().numpy()
        
        metrics = compute_metrics(true, pred, y_probs=probs)
        
    return loss, metrics


def evaluate_edges_with_predictions(model, data, edge_indices, criterion, threshold, device):
    """Evaluate and return predictions using full-batch forward pass."""
    model.eval()
    
    with torch.no_grad():
        # Single forward pass for all edges
        logits_all = model(data.x, data.edge_index, data.edge_attr)  # [num_edges, num_classes]
        # Get logits for evaluation edges
        logits = logits_all[edge_indices]
        edge_labels = data.edge_y[edge_indices]
        loss = criterion(logits, edge_labels.long()).item()
        
        # Softmax -> probability of class 1 (attack)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = (probs >= threshold).astype(int)
        true = edge_labels.cpu().numpy()
        
        metrics = compute_metrics(true, pred, y_probs=probs)
        
    return loss, metrics, true, pred, probs
