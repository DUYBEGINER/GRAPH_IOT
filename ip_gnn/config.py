# Endpoint-based GNN Configuration
# Node = endpoint (IP), Edge = flow, Task = edge classification

# ============================================================================
# PROJECT SETTINGS
# ============================================================================
PROJECT_NAME = "IP-GNN"
SEED = 42
DEVICE = "auto"  # auto, cuda, mps, cpu

# ============================================================================
# DATA SETTINGS
# ============================================================================
DATA_DIR = "dataset-processed/ip_gnn"

# Column names
SRC_IP_COL = "Src IP"
SRC_PORT_COL = "Src Port"
DST_IP_COL = "Dst IP"
DST_PORT_COL = "Dst Port"
LABEL_COL = "Label"

# Columns to drop for feature extraction
DROP_COLS = ["Flow ID", "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port"]

# ============================================================================
# MODEL SETTINGS
# ============================================================================
HIDDEN_DIM = 128
NUM_CLASSES = 2
NUM_LAYERS = 2
DROPOUT = 0.2
AGGR = "mean"  # mean or sum

# ============================================================================
# GRAPH SETTINGS
# ============================================================================
MAPPING_MODE = "ip_only" 

# Anti-leakage settings
ANTI_LEAKAGE_ENABLED = True
ANTI_LEAKAGE_SCOPE = "src_ip_only"  # all_ips or src_ip_only

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 10
MIN_DELTA = 0.001

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
OUTPUT_DIR = "output/ip_gnn"
