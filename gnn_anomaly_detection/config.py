"""
Configuration file for GNN Anomaly Detection Project
Chứa các cấu hình và hyperparameters cho toàn bộ pipeline
"""

import os

# ============================================================================
# PATH CONFIGURATIONS
# ============================================================================
# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "CICIDS2018-CSV")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data file
DATA_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Create directories if not exist
for dir_path in [OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# DATA PREPROCESSING CONFIGURATIONS
# ============================================================================
# Columns to drop (not useful for classification)
COLUMNS_TO_DROP = [
    'Flow ID',      # Unique identifier
    'Src IP',       # Source IP address (categorical, too many unique values)
    'Dst IP',       # Destination IP address (categorical, too many unique values)
    'Timestamp',    # Time information (not useful for this task)
]

# Label column
LABEL_COLUMN = 'Label'

# Normal label value (for binary classification)
NORMAL_LABEL = 'Benign'

# Sample size for training (None = use all data)
# Set a smaller value if you have memory constraints
SAMPLE_SIZE = 500000  # Use 500k samples to speed up training

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# FEATURE ENGINEERING CONFIGURATIONS
# ============================================================================
# Standard scaler parameters
SCALER_TYPE = 'standard'  # 'standard', 'minmax', 'robust'

# Handle infinite values
REPLACE_INF_WITH = 0

# ============================================================================
# GRAPH CONSTRUCTION CONFIGURATIONS
# ============================================================================
# KNN graph parameters
K_NEIGHBORS = 5  # Number of nearest neighbors for graph construction

# Graph construction method
GRAPH_METHOD = 'knn'  # 'knn', 'radius', 'threshold'

# For threshold-based graph
SIMILARITY_THRESHOLD = 0.8

# Maximum edges per node (to limit memory usage)
MAX_EDGES_PER_NODE = 10

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
# GNN Architecture
INPUT_DIM = None  # Will be set dynamically based on features
HIDDEN_DIM = 64
OUTPUT_DIM = 2  # Binary classification: Normal (0) vs Anomaly (1)
NUM_LAYERS = 3
DROPOUT = 0.3

# GNN Type
GNN_TYPE = 'GCN'  # 'GCN', 'GAT', 'GraphSAGE'

# For GAT
NUM_HEADS = 4

# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================
# Training parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
BATCH_SIZE = 1024  # For mini-batch training

# Train/Val/Test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Early stopping
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 0.001

# Class weight balancing
USE_CLASS_WEIGHTS = True

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# LOGGING CONFIGURATIONS
# ============================================================================
LOG_INTERVAL = 10  # Log every N epochs
SAVE_BEST_MODEL = True

