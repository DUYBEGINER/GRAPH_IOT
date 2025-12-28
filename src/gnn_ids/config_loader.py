"""Configuration loader for GNN-IDS."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import torch

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_str: "auto", "cuda", "mps", or "cpu"
        
    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    
    return device


def print_config(config: Dict[str, Any]):
    """Pretty print configuration."""
    print("\n" + "=" * 70)
    print("GNN-IDS CONFIGURATION")
    print("=" * 70)
    
    print(f"\nMode: {config['mode'].upper()}")
    print(f"Task: {config['task']['type']} classification")
    print(f"Seed: {config['project']['seed']}")
    
    print(f"\nData:")
    print(f"  CSV Path: {config['data']['csv_path']}")
    print(f"  Max Samples: {config['data']['max_samples']}")
    print(f"  Test Split: {config['data']['test_split']}")
    print(f"  Val Split: {config['data']['val_split']}")
    
    if config['mode'] == "flow":
        print(f"\nFlow-based Graph:")
        print(f"  K Neighbors: {config['flow_graph']['k_neighbors']}")
        print(f"  Metric: {config['flow_graph']['metric']}")
        
        print(f"\nModel (GraphSAGE):")
        print(f"  Hidden Dim: {config['flow_model']['hidden_dim']}")
        print(f"  Layers: {config['flow_model']['num_layers']}")
        print(f"  Dropout: {config['flow_model']['dropout']}")
    else:
        print(f"\nEndpoint-based Graph:")
        print(f"  Mapping Mode: {config['endpoint_graph']['mapping_mode']}")
        print(f"  Anti-leakage: {config['endpoint_graph']['anti_leakage']['enabled']}")
        
        print(f"\nModel (E-GraphSAGE):")
        print(f"  Hidden Dim: {config['endpoint_model']['hidden_dim']}")
        print(f"  Layers: {config['endpoint_model']['num_layers']}")
        print(f"  Dropout: {config['endpoint_model']['dropout']}")
    
    print(f"\nTraining:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    
    print("=" * 70 + "\n")
