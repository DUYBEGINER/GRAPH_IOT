"""Configuration file for GNN-IDS model training."""

import os
import logging
import torch
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
SEED = 42

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "src" / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
CSV_PATH: Optional[str] = None  # Will be set at runtime or via command line
MAX_SAMPLES = 100_000  # Limit number of flows for building graph

# Graph construction
K_NEIGHBORS = 10  # Number of neighbors in KNN graph
GRAPH_METRIC = "cosine"  # Distance metric: 'cosine', 'euclidean', 'manhattan'

# Data split ratios
VAL_RATIO = 0.1
TEST_RATIO = 0.2
assert VAL_RATIO + TEST_RATIO < 1.0, "Val + Test ratio must be < 1.0"

# Training hyperparameters
BATCH_SIZE = 1024
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 2  # Binary classification: Benign (0) vs Attack (1)
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT = 0.3

# Early stopping
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4

# Neighbor sampling for GraphSAGE (mini-batch training)
NUM_NEIGHBORS = [10, 5]  # Number of neighbors to sample per layer

# Model saving
SAVE_BEST_MODEL = True
MODEL_SAVE_PATH = CHECKPOINT_DIR / "best_model.pth"

# Device configuration
def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def validate_config():
    """Validate configuration parameters."""
    assert MAX_SAMPLES > 0, "MAX_SAMPLES must be positive"
    assert K_NEIGHBORS > 0, "K_NEIGHBORS must be positive"
    assert 0 < VAL_RATIO < 1, "VAL_RATIO must be between 0 and 1"
    assert 0 < TEST_RATIO < 1, "TEST_RATIO must be between 0 and 1"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert HIDDEN_DIM > 0, "HIDDEN_DIM must be positive"
    assert NUM_EPOCHS > 0, "NUM_EPOCHS must be positive"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    assert 0 <= DROPOUT < 1, "DROPOUT must be between 0 and 1"
    logger.info("Configuration validated successfully")


def print_config():
    """Print current configuration."""
    logger.info("=" * 60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Data: MAX_SAMPLES={MAX_SAMPLES}")
    logger.info(f"Graph: K_NEIGHBORS={K_NEIGHBORS}, METRIC={GRAPH_METRIC}")
    logger.info(f"Split: TRAIN={1-VAL_RATIO-TEST_RATIO:.1%}, VAL={VAL_RATIO:.1%}, TEST={TEST_RATIO:.1%}")
    logger.info(f"Model: HIDDEN_DIM={HIDDEN_DIM}, LAYERS={NUM_LAYERS}, DROPOUT={DROPOUT}")
    logger.info(f"Training: EPOCHS={NUM_EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LEARNING_RATE}")
    logger.info(f"Device: {get_device()}")
    logger.info("=" * 60)
