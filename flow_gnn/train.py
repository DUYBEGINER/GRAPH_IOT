"""Training logic for Flow-based GNN."""

import logging
import torch
import torch.nn as nn
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from . import config as cfg
from .model import FlowGraphSAGE
from .graph import build_knn_graph
from .utils import compute_metrics, EarlyStopping, get_device, save_metrics_plots, save_metrics_report, set_seed


class RandomNodeSampler:
    """Simple random node sampler for mini-batch training on full graph."""
    
    def __init__(self, mask: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.node_indices = mask.nonzero(as_tuple=True)[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = self.node_indices.clone()
        if self.shuffle:
            perm = torch.randperm(len(indices))
            indices = indices[perm]
        
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]
    
    def __len__(self):
        return (len(self.node_indices) + self.batch_size - 1) // self.batch_size


def train_flow_gnn(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device = None,
    output_dir: str = None
) -> Dict:
    """Train Flow-based GNN model.
    
    Args:
        x_tensor: Node features
        y_tensor: Node labels
        edge_index: Graph edges
        train_mask, val_mask, test_mask: Boolean masks for splits
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
    print("🚀 TRAINING FLOW-BASED GNN")
    print("=" * 70)
    
    # Create PyG Data
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    ).to(device)
    
    # Calculate pos_weight from training set
    y_train = y_tensor[train_mask]
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    pos_weight = neg / pos if pos > 0 else 1.0
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples:   {train_mask.sum().item():,}")
    print(f"   Validation samples: {val_mask.sum().item():,}")
    print(f"   Test samples:       {test_mask.sum().item():,}")
    print(f"   Class distribution: Benign={neg:,} ({neg/(neg+pos)*100:.1f}%), Attack={pos:,} ({pos/(neg+pos)*100:.1f}%)")
    print(f"   Positive weight:    {pos_weight:.4f}")
    
    # Model
    print(f"\n🏗️  Model Configuration:")
    print(f"   Hidden dim:  {cfg.HIDDEN_DIM}")
    print(f"   Num layers:  {cfg.NUM_LAYERS}")
    print(f"   Dropout:     {cfg.DROPOUT}")
    
    model = FlowGraphSAGE(
        in_dim=x_tensor.shape[1],
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")
    print(f"   Device:       {device}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs:         {cfg.EPOCHS}")
    print(f"   Batch size:     {cfg.BATCH_SIZE}")
    print(f"   Learning rate:  {cfg.LEARNING_RATE}")
    print(f"   Weight decay:   {cfg.WEIGHT_DECAY}")
    print(f"   Early stopping: {cfg.PATIENCE} epochs")
    
    # Sampler & Early stopping
    train_sampler = RandomNodeSampler(train_mask, batch_size=cfg.BATCH_SIZE, shuffle=True)
    early_stopping = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA)
    
    # Training history
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_accuracy': []
    }
    
    print(f"\n🔥 Starting Training...")
    print("-" * 70)
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(1, cfg.EPOCHS + 1), desc="Training", unit="epoch", ncols=100)
    
    for epoch in epoch_pbar:
        # Train
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_nodes in train_sampler:
            batch_nodes = batch_nodes.to(device)
            optimizer.zero_grad()
            
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[batch_nodes], data.y[batch_nodes].float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        train_loss = total_loss / num_batches
        
        # Validate
        val_loss, val_metrics = evaluate(model, data, val_mask, criterion, device)
        
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
    best_threshold = tune_threshold(model, data, val_mask, device)
    print(f"✅ Optimal threshold: {best_threshold:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("🧪 FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    start_time = time.time()
    test_loss, test_metrics, y_true, y_pred, y_probs = evaluate_with_predictions(
        model, data, test_mask, criterion, best_threshold, device
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


def tune_threshold(model, data, mask, device):
    """Find optimal threshold to maximize F1 score."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        val_logits = logits[mask]
        val_targets = data.y[mask]
    
    val_probs = torch.sigmoid(val_logits).cpu().numpy()
    y_val_np = val_targets.cpu().numpy()
    
    best_t, best_f1 = 0.5, 0.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in tqdm(thresholds, desc="   Searching threshold", ncols=100, leave=False):
        y_pred = (val_probs >= t).astype(int)
        f1 = f1_score(y_val_np, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    
    return best_t


def evaluate(model, data, mask, criterion, device, threshold=0.5):
    """Evaluate model on given mask."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        mask_logits = logits[mask]
        mask_y = data.y[mask]
        loss = criterion(mask_logits, mask_y.float()).item()
    
    probs = torch.sigmoid(mask_logits).cpu().numpy()
    pred = (probs >= threshold).astype(int)
    true = mask_y.cpu().numpy()
    
    metrics = compute_metrics(true, pred, y_probs=probs)
    return loss, metrics


def evaluate_with_predictions(model, data, mask, criterion, threshold, device):
    """Evaluate and return predictions."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        mask_logits = logits[mask]
        mask_y = data.y[mask]
        loss = criterion(mask_logits, mask_y.float()).item()
    
    probs = torch.sigmoid(mask_logits).cpu().numpy()
    pred = (probs >= threshold).astype(int)
    true = mask_y.cpu().numpy()
    
    metrics = compute_metrics(true, pred, y_probs=probs)
    return loss, metrics, true, pred, probs
