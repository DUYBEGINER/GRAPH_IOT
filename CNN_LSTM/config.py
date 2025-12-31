"""
Configuration for CNN and LSTM training pipeline
"""
import os

# Environment detection
IS_KAGGLE = os.path.exists('/kaggle/input')

# Data paths
if IS_KAGGLE:
    DATA_DIR = "/kaggle/input/cicids2018-csv"
    OUTPUT_DIR = "/kaggle/working"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "CICIDS2018-CSV")
    OUTPUT_DIR = BASE_DIR

# Excluded file (reserved for local testing)
EXCLUDED_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

# Preprocessing
# Columns to drop (Timestamp kept for sorting, dropped after)
COLS_TO_DROP = [
    'Flow ID', 'Src IP', 'Dst IP', 'Src Port',
    'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count'
]

# Data sampling
TOTAL_SAMPLES = 2_000_000
BENIGN_RATIO = 0.7
ATTACK_RATIO = 0.3

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training
BATCH_SIZE = 512 if IS_KAGGLE else 256
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT = 0.3
PATIENCE = 10
RANDOM_STATE = 42

# LSTM specific
LSTM_UNITS = 128
LSTM_SEQUENCE_LENGTH = 10

# CNN specific
CNN_FILTERS = [32, 64, 64]

