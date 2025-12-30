import logging
import torch
import torch.nn as nn
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from .model import FlowGraphSAGE
from .graph import build_knn_graph
from .utils import compute_metrics, EarlyStopping, get_device, save_metrics_plots, save_metrics_report


class RandomNodeSampler:
    """Simple random node sampler for mini-batch training on full graph.
    
    Instead of sampling neighbors (which requires pyg-lib/torch-sparse),
    we sample random nodes and compute loss only on those nodes while
    using the full graph for message passing.
    """
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
    config: dict,
    device: torch.device
) -> Dict:
    """Train Flow-based GNN model with comprehensive logging and progress tracking.
    
    Uses full-graph message passing with mini-batch node sampling for loss computation.
    This approach doesn't require pyg-lib or torch-sparse.
    """
    
    logging.info("\n" + "="*80)
    logging.info("🚀 TRAINING FLOW-BASED GNN")
    logging.info("="*80)
    
    # Create PyG Data and move to device
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    ).to(device)
    
    # Calculate pos_weight from TRAINING set only
    y_train = y_tensor[train_mask]
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    pos_weight = neg / pos if pos > 0 else 1.0
    
    logging.info(f"\n📊 Dataset Statistics:")
    logging.info(f"   Training samples: {train_mask.sum().item():,}")
    logging.info(f"   Validation samples: {val_mask.sum().item():,}")
    logging.info(f"   Test samples: {test_mask.sum().item():,}")
    logging.info(f"   Class distribution (train): Benign={neg:,} ({neg/(neg+pos)*100:.1f}%), Attack={pos:,} ({pos/(neg+pos)*100:.1f}%)")
    logging.info(f"   Positive weight (for loss): {pos_weight:.4f}")
    
    # Model
    logging.info(f"\n🏗️  Building Model:")
    model = FlowGraphSAGE(
        in_dim=x_tensor.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"   Total parameters: {total_params:,}")
    logging.info(f"   Trainable parameters: {trainable_params:,}")
    logging.info(f"   Device: {device}")
    
    # Optimizer & Loss (BCEWithLogitsLoss with pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    
    logging.info(f"\n⚙️  Training Configuration:")
    logging.info(f"   Epochs: {config['training']['epochs']}")
    logging.info(f"   Batch size: {config['training']['batch_size']}")
    logging.info(f"   Learning rate: {config['training']['learning_rate']}")
    logging.info(f"   Weight decay: {config['training'].get('weight_decay', 0)}")
    logging.info(f"   Early stopping patience: {config['training'].get('patience', 10)}")
    logging.info(f"   Mode: Full-graph message passing with mini-batch loss")
    
    # Random node sampler for training
    train_sampler = RandomNodeSampler(
        train_mask,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Training loop with history tracking
    early_stopping = EarlyStopping(
        patience=config['training'].get('patience', 10),
        min_delta=config['training'].get('min_delta', 0.001)
    )
    
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_accuracy': []
    }
    
    logging.info(f"\n🔥 Starting Training...")
    logging.info("-" * 80)
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(1, config['training']['epochs'] + 1), 
                      desc="Training", unit="epoch", ncols=120)
    
    for epoch in epoch_pbar:
        # Train - full graph forward pass, mini-batch loss
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_nodes in train_sampler:
            batch_nodes = batch_nodes.to(device)
            optimizer.zero_grad()
            
            # Full graph forward pass
            logits = model(data.x, data.edge_index)
            
            # Compute loss only on batch nodes
            loss = criterion(logits[batch_nodes], data.y[batch_nodes].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        train_loss = total_loss / num_batches
        
        # Validate - full graph inference
        val_loss, val_metrics = evaluate(model, data, val_mask, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'val_f1': f"{val_metrics['f1']:.4f}",
            'val_acc': f"{val_metrics['accuracy']:.4f}"
        })
        
        # Log important epochs
        if epoch % 10 == 0 or epoch == 1:
            logging.info(
                f"   Epoch {epoch:3d}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
            )

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = Path(config.get('output_dir', 'output/flow_gnn')) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config
            }, save_path)
        
        # Early stopping
        if early_stopping(val_metrics['f1']):
            logging.info(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break
    
    epoch_pbar.close()
    
    logging.info(f"\n✅ Training completed!")
    logging.info(f"   Best validation F1: {best_f1:.4f}")
    
    # Tune threshold on validation set
    logging.info("\n" + "="*80)
    logging.info("🎯 TUNING DECISION THRESHOLD")
    logging.info("="*80)
    best_threshold = tune_threshold(model, data, val_mask, device)
    logging.info(f"✅ Optimal threshold: {best_threshold:.4f}")
    
    # Test with tuned threshold
    logging.info("\n" + "="*80)
    logging.info("🧪 FINAL EVALUATION ON TEST SET")
    logging.info("="*80)
    
    start_time = time.time()
    test_loss, test_metrics, y_true, y_pred, y_probs = evaluate_with_predictions(
        model, data, test_mask, criterion, best_threshold, device
    )
    inference_time = time.time() - start_time
    latency_per_sample = inference_time / len(y_true)
    
    logging.info(f"\n📈 Test Results:")
    logging.info(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    logging.info(f"   Precision: {test_metrics['precision']:.4f}")
    logging.info(f"   Recall:    {test_metrics['recall']:.4f}")
    logging.info(f"   F1 Score:  {test_metrics['f1']:.4f}")
    if 'auc' in test_metrics:
        logging.info(f"   AUC:       {test_metrics['auc']:.4f}")
    logging.info(f"   FAR:       {test_metrics['far']:.4f}")
    logging.info(f"\n⏱️  Inference Performance:")
    logging.info(f"   Total time: {inference_time:.2f}s")
    logging.info(f"   Latency per sample: {latency_per_sample*1000:.4f}ms")
    logging.info(f"   Throughput: {len(y_true)/inference_time:.2f} samples/sec")
    
    # Save visualizations and reports
    output_dir = Path(config.get('output_dir', 'output/flow_gnn'))
    logging.info(f"\n💾 Saving results to {output_dir}/")
    
    save_metrics_plots(y_true, y_pred, y_probs, test_metrics, 
                      str(output_dir), history=history)
    save_metrics_report(test_metrics, str(output_dir), 
                       y_true, y_pred, latency=latency_per_sample)
    
    logging.info("\n" + "="*80)
    logging.info("✨ ALL DONE!")
    logging.info("="*80 + "\n")
    
    return test_metrics


def tune_threshold(model, data, mask, device):
    """Find optimal threshold on validation set to maximize F1 score.
    
    Uses full-graph inference (no neighbor sampling).
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        val_logits = logits[mask]
        val_targets = data.y[mask]
    
    # Convert to probabilities
    val_probs = torch.sigmoid(val_logits).cpu().numpy()
    y_val_np = val_targets.cpu().numpy()
    
    # Search for best threshold with progress bar
    best_t, best_f1 = 0.5, 0.0
    best_precision, best_recall = 0.0, 0.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in tqdm(thresholds, desc="Searching threshold", ncols=100, leave=False):
        y_pred = (val_probs >= t).astype(int)
        f1 = f1_score(y_val_np, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_precision = precision_score(y_val_np, y_pred, zero_division=0)
            best_recall = recall_score(y_val_np, y_pred, zero_division=0)
    
    logging.info(f"   Threshold: {best_t:.4f}")
    logging.info(f"   Precision: {best_precision:.4f}")
    logging.info(f"   Recall:    {best_recall:.4f}")
    logging.info(f"   F1 Score:  {best_f1:.4f}")
    
    return best_t


def evaluate(model, data, mask, criterion, device, threshold=0.5):
    """Evaluate model on given mask using full-graph inference."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        
        # Get predictions for masked nodes
        mask_logits = logits[mask]
        mask_y = data.y[mask]
        
        loss = criterion(mask_logits, mask_y.float()).item()
    
    # Convert logits to probabilities and apply threshold
    probs = torch.sigmoid(mask_logits).cpu().numpy()
    pred = (probs >= threshold).astype(int)
    true = mask_y.cpu().numpy()
    
    metrics = compute_metrics(true, pred, y_probs=probs)
    
    return loss, metrics


def evaluate_with_predictions(model, data, mask, criterion, threshold, device):
    """Evaluate and return predictions using full-graph inference."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        
        # Get predictions for masked nodes
        mask_logits = logits[mask]
        mask_y = data.y[mask]
        
        loss = criterion(mask_logits, mask_y.float()).item()
    
    # Convert logits to probabilities and apply threshold
    probs = torch.sigmoid(mask_logits).cpu().numpy()
    pred = (probs >= threshold).astype(int)
    true = mask_y.cpu().numpy()
    
    metrics = compute_metrics(true, pred, y_probs=probs)
    
    return loss, metrics, true, pred, probs
