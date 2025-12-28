"""Evaluation utilities for E-GraphSAGE."""

import logging
import torch
import numpy as np
from typing import Tuple
from torch_geometric.data import Data

from .utils.metrics import (
    compute_metrics,
    print_metrics,
    get_classification_report,
    get_confusion_matrix
)

logger = logging.getLogger(__name__)


def evaluate_edge_model(
    model,
    data: Data,
    edge_mask: torch.Tensor,
    device: torch.device,
    task_type: str = "binary",
    prefix: str = "Test"
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate edge classification model.
    
    Args:
        model: E-GraphSAGE model
        data: PyG Data object
        edge_mask: Boolean mask for edges to evaluate
        device: Device to use
        task_type: "binary" or "multiclass"
        prefix: Prefix for logging
        
    Returns:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
    """
    model.eval()
    
    with torch.no_grad():
        # Move data to device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        
        # Get edges to evaluate
        eval_edge_index = edge_index[:, edge_mask]
        eval_edge_y = data.edge_y[edge_mask]
        
        # Forward pass
        logits = model(x, edge_index, edge_attr, edge_label_index=eval_edge_index)
        
        # Get predictions
        if task_type == "binary" and logits.size(1) == 2:
            pred = logits.argmax(dim=1)
        elif task_type == "binary" and logits.size(1) == 1:
            pred = (torch.sigmoid(logits.squeeze()) > 0.5).long()
        else:
            pred = logits.argmax(dim=1)
        
        # Convert to numpy
        y_true = eval_edge_y.cpu().numpy()
        y_pred = pred.cpu().numpy()
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, task_type=task_type)
    
    # Print metrics
    print_metrics(metrics, prefix=prefix)
    
    # Print classification report
    print(f"\n{prefix} Classification Report:")
    print(get_classification_report(y_true, y_pred))
    
    # Print confusion matrix
    print(f"\n{prefix} Confusion Matrix:")
    cm = get_confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Compute and print FAR explicitly
    if task_type == "binary":
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"\nDetailed Binary Metrics:")
        print(f"  True Negatives (TN):  {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP):  {tp}")
        print(f"  False Alarm Rate:     {far:.4f}")
        print(f"  Detection Rate:       {metrics['detection_rate']:.4f}")
    
    return metrics, y_true, y_pred


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str
):
    """
    Save predictions to file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save predictions
    """
    import pandas as pd
    
    df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred
    })
    
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
