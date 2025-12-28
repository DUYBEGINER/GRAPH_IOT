"""Utility functions for GNN-IDS."""

import random
import logging
import numpy as np
import torch
from typing import Dict, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_metrics(metrics: Dict[str, Any], save_path: Path):
    """
    Save training metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON file
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {save_path}")


def load_metrics(load_path: Path) -> Dict[str, Any]:
    """
    Load training metrics from JSON file.
    
    Args:
        load_path: Path to JSON file
        
    Returns:
        Dictionary of metrics
    """
    if not load_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"Metrics loaded from {load_path}")
    return metrics


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage() -> str:
    """
    Get current GPU memory usage if CUDA is available.
    
    Returns:
        Memory usage string
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    else:
        return "GPU not available"
